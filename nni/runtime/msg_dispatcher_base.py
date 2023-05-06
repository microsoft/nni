# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading
import logging
from queue import Queue, Empty

from .env_vars import dispatcher_env_vars
from ..common import load
from ..recoverable import Recoverable
from .tuner_command_channel import CommandType, TunerCommandChannel


_logger = logging.getLogger(__name__)

QUEUE_LEN_WARNING_MARK = 20
_worker_fast_exit_on_terminate = True


class MsgDispatcherBase(Recoverable):
    """
    This is where tuners and assessors are not defined yet.
    Inherits this class to make your own advisor.

    .. note::

        The class inheriting MsgDispatcherBase should be instantiated
        after nnimanager (rest server) is started, so that the object
        is ready to use right after its instantiation.
    """

    def __init__(self, command_channel_url=None):
        super().__init__()
        self.stopping = False
        if command_channel_url is None:
            command_channel_url = dispatcher_env_vars.NNI_TUNER_COMMAND_CHANNEL
        self._channel = TunerCommandChannel(command_channel_url)
        # NOTE: `connect()` should be put in __init__. First, this `connect()` affects nnimanager's
        # starting process, without `connect()` nnimanager is blocked in `dispatcher.init()`.
        # Second, nas experiment uses a thread to execute `run()` of this class, thus, there is
        # no way to know when the websocket between nnimanager and dispatcher is built. The following
        # logic may crash is websocket is not built. One example is updating search space. If updating
        # search space too soon, as the websocket has not been built, the rest api of updating search
        # space will timeout.
        # FIXME: this is making unittest happy
        if not command_channel_url.startswith('ws://_unittest_'):
            self._channel.connect()
        self.default_command_queue = Queue()
        self.assessor_command_queue = Queue()
        # here daemon should be True, because their parent thread is configured as daemon to enable smooth exit of NAS experiment.
        # if daemon is not set, these threads will block the daemon effect of their parent thread.
        self.default_worker = threading.Thread(target=self.command_queue_worker, args=(self.default_command_queue,), daemon=True)
        self.assessor_worker = threading.Thread(target=self.command_queue_worker, args=(self.assessor_command_queue,), daemon=True)
        self.worker_exceptions = []

    def run(self):
        """Run the tuner.
        This function will never return unless raise.
        """
        _logger.info('Dispatcher started')

        self.default_worker.start()
        self.assessor_worker.start()

        if dispatcher_env_vars.NNI_MODE == 'resume':
            self.load_checkpoint()

        while not self.stopping:
            command, data = self._channel._receive()
            if data:
                data = load(data)

            if command is None or command is CommandType.Terminate:
                break
            self.enqueue_command(command, data)
            if self.worker_exceptions:
                break

        _logger.info('Dispatcher exiting...')
        self.stopping = True
        self.default_worker.join()
        self.assessor_worker.join()
        self._channel.disconnect()

        _logger.info('Dispatcher terminiated')

    def report_error(self, error: str) -> None:
        '''
        Report dispatcher error to NNI manager.
        '''
        _logger.info(f'Report error to NNI manager: {error}')
        try:
            self.send(CommandType.Error, error)
        except Exception:
            _logger.error('Connection to NNI manager is broken. Failed to report error.')

    def send(self, command, data):
        self._channel._send(command, data)

    def command_queue_worker(self, command_queue):
        """Process commands in command queues.
        """
        while True:
            try:
                # set timeout to ensure self.stopping is checked periodically
                command, data = command_queue.get(timeout=3)
                try:
                    self.process_command(command, data)
                except Exception as e:
                    _logger.exception(e)
                    self.worker_exceptions.append(e)
                    break
            except Empty:
                pass
            if self.stopping and (_worker_fast_exit_on_terminate or command_queue.empty()):
                break

    def enqueue_command(self, command, data):
        """Enqueue command into command queues
        """
        if command == CommandType.TrialEnd or (
                command == CommandType.ReportMetricData and data['type'] == 'PERIODICAL'):
            self.assessor_command_queue.put((command, data))
        else:
            self.default_command_queue.put((command, data))

        qsize = self.default_command_queue.qsize()
        if qsize >= QUEUE_LEN_WARNING_MARK:
            _logger.warning('default queue length: %d', qsize)

        qsize = self.assessor_command_queue.qsize()
        if qsize >= QUEUE_LEN_WARNING_MARK:
            _logger.warning('assessor queue length: %d', qsize)

    def process_command(self, command, data):
        _logger.debug('process_command: command: [%s], data: [%s]', command, data)

        command_handlers = {
            # Tuner commands:
            CommandType.Initialize: self.handle_initialize,
            CommandType.RequestTrialJobs: self.handle_request_trial_jobs,
            CommandType.UpdateSearchSpace: self.handle_update_search_space,
            CommandType.ImportData: self.handle_import_data,
            CommandType.AddCustomizedTrialJob: self.handle_add_customized_trial,

            # Tuner/Assessor commands:
            CommandType.ReportMetricData: self.handle_report_metric_data,

            CommandType.TrialEnd: self.handle_trial_end,
            CommandType.Ping: self.handle_ping,
        }
        if command not in command_handlers:
            raise AssertionError('Unsupported command: {}'.format(command))
        command_handlers[command](data)

    def handle_ping(self, data):
        pass

    def handle_initialize(self, data):
        """Initialize search space and tuner, if any
        This method is meant to be called only once for each experiment, after calling this method,
        dispatcher should `send(CommandType.Initialized, '')`, to set the status of the experiment to be "INITIALIZED".
        Parameters
        ----------
        data: dict
            search space
        """
        raise NotImplementedError('handle_initialize not implemented')

    def handle_request_trial_jobs(self, data):
        """The message dispatcher is demanded to generate ``data`` trial jobs.
        These trial jobs should be sent via ``send(CommandType.NewTrialJob, nni.dump(parameter))``,
        where ``parameter`` will be received by NNI Manager and eventually accessible to trial jobs as "next parameter".
        Semantically, message dispatcher should do this ``send`` exactly ``data`` times.

        The JSON sent by this method should follow the format of

        ::

            {
                "parameter_id": 42
                "parameters": {
                    // this will be received by trial
                },
                "parameter_source": "algorithm" // optional
            }

        Parameters
        ----------
        data: int
            number of trial jobs
        """
        raise NotImplementedError('handle_request_trial_jobs not implemented')

    def handle_update_search_space(self, data):
        """This method will be called when search space is updated.
        It's recommended to call this method in `handle_initialize` to initialize search space.
        *No need to* notify NNI Manager when this update is done.
        Parameters
        ----------
        data: dict
            search space
        """
        raise NotImplementedError('handle_update_search_space not implemented')

    def handle_import_data(self, data):
        """Import previous data when experiment is resumed.
        Parameters
        ----------
        data: list
            a list of dictionaries, each of which has at least two keys, 'parameter' and 'value'
        """
        raise NotImplementedError('handle_import_data not implemented')

    def handle_add_customized_trial(self, data):
        """Experimental API. Not recommended for usage.
        """
        raise NotImplementedError('handle_add_customized_trial not implemented')

    def handle_report_metric_data(self, data):
        """Called when metric data is reported or new parameters are requested (for multiphase).
        When new parameters are requested, this method should send a new parameter.

        Parameters
        ----------
        data: dict
            a dict which contains 'parameter_id', 'value', 'trial_job_id', 'type', 'sequence'.
            type: can be `MetricType.REQUEST_PARAMETER`, `MetricType.FINAL` or `MetricType.PERIODICAL`.
            `REQUEST_PARAMETER` is used to request new parameters for multiphase trial job. In this case,
            the dict will contain additional keys: `trial_job_id`, `parameter_index`. Refer to `msg_dispatcher.py`
            as an example.

        Raises
        ------
        ValueError
            Data type is not supported
        """
        raise NotImplementedError('handle_report_metric_data not implemented')

    def handle_trial_end(self, data):
        """Called when the state of one of the trials is changed

        Parameters
        ----------
        data: dict
            a dict with keys: trial_job_id, event, hyper_params.
            trial_job_id: the id generated by training service.
            event: the job’s state.
            hyper_params: the string that is sent by message dispatcher during the creation of trials.

        """
        raise NotImplementedError('handle_trial_end not implemented')
