# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from collections import defaultdict
from queue import Empty
from nni import NoMoreTrialError
from nni.assessor import AssessResult
from .tuner_command_channel.semantic_command import (AddCustomizedTrialJob, BaseCommand,
ImportData, Initialize, Initialized, KillTrialJob,
NewTrialJob, NoMoreTrialJobs,
ReportMetricData, RequestTrialJobs, SendTrialJobParameter, TrialEnd, UpdateSearchSpace)

from .common import multi_phase_enabled
from .env_vars import dispatcher_env_vars
from .msg_dispatcher_base import MsgDispatcherBase
from ..common.serializer import dump, load
from ..utils import MetricType

_logger = logging.getLogger(__name__)

# Assessor global variables
_trial_history = defaultdict(dict)
'''key: trial job ID; value: intermediate results, mapping from sequence number to data'''

_ended_trials = set()
'''trial_job_id of all ended trials.
We need this because NNI manager may send metrics after reporting a trial ended.
TODO: move this logic to NNI manager
'''

QUEUE_LEN_WARNING_MARK = 20
_worker_fast_exit_on_terminate = True

def _sort_history(history):
    ret = []
    for i, _ in enumerate(history):
        if i in history:
            ret.append(history[i])
        else:
            break
    return ret


# Tuner global variables
_next_parameter_id = 0
_trial_params = {}
'''key: parameter ID; value: parameters'''
_customized_parameter_ids = set()


def _create_parameter_id():
    global _next_parameter_id
    _next_parameter_id += 1
    return _next_parameter_id - 1


def _pack_parameter(parameter_id, params, customized=False, trial_job_id=None, parameter_index=None):
    _trial_params[parameter_id] = params
    ret = {
        'parameter_id': parameter_id,
        'parameter_source': 'customized' if customized else 'algorithm',
        'parameters': params
    }
    if trial_job_id is not None:
        ret['trial_job_id'] = trial_job_id
    if parameter_index is not None:
        ret['parameter_index'] = parameter_index
    else:
        ret['parameter_index'] = 0
    return ret


class MsgDispatcher(MsgDispatcherBase):
    def __init__(self, command_channel_url, tuner, assessor=None):
        super().__init__(command_channel_url)
        self.tuner = tuner
        self.assessor = assessor
        if assessor is None:
            _logger.debug('Assessor is not configured')

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
            command = self._channel.receive()
            if command is None or command.command_type == 'Terminate':
                break
            self.enqueue_command(command)
            if self.worker_exceptions:
                break

        _logger.info('Dispatcher exiting...')
        self.stopping = True
        self.default_worker.join()
        self.assessor_worker.join()
        self._channel.disconnect()

        _logger.info('Dispatcher terminiated')

    def send(self, command: BaseCommand):
        self._channel.send(command)

    def command_queue_worker(self, command_queue):
        """Process commands in command queues.
        """
        while True:
            try:
                # set timeout to ensure self.stopping is checked periodically
                command = command_queue.get(timeout=3)
                try:
                    self.process_command(command)
                except Exception as e:
                    _logger.exception(e)
                    self.worker_exceptions.append(e)
                    break
            except Empty:
                pass
            if self.stopping and (_worker_fast_exit_on_terminate or command_queue.empty()):
                break

    def enqueue_command(self, command: BaseCommand):
        """Enqueue command into command queues
        """
        if command.command_type == 'TrialEnd' or (
                command.command_type == 'ReportMetricData' and command.__dict__['type'] == 'PERIODICAL'):
            self.assessor_command_queue.put((command))
        else:
            self.default_command_queue.put((command))

        qsize = self.default_command_queue.qsize()
        if qsize >= QUEUE_LEN_WARNING_MARK:
            _logger.warning('default queue length: %d', qsize)

        qsize = self.assessor_command_queue.qsize()
        if qsize >= QUEUE_LEN_WARNING_MARK:
            _logger.warning('assessor queue length: %d', qsize)

    def process_command(self, command: BaseCommand):
        _logger.debug('process_command: command: [%s]', command)

        command_handlers = {
            # Tuner commands:
            'Initialize': self.handle_initialize,
            'RequestTrialJobs': self.handle_request_trial_jobs,
            'UpdateSearchSpace': self.handle_update_search_space,
            'ImportData': self.handle_import_data,
            'AddCustomizedTrialJob': self.handle_add_customized_trial,

            # Tuner/Assessor commands:
            'ReportMetricData': self.handle_report_metric_data,

            'TrialEnd': self.handle_trial_end,
            'Ping': self.handle_ping,
        }
        if command.command_type not in command_handlers:
            raise AssertionError('Unsupported command: {}'.format(command))
        command_handlers[command.command_type](command)

    def load_checkpoint(self):
        self.tuner.load_checkpoint()
        if self.assessor is not None:
            self.assessor.load_checkpoint()

    def save_checkpoint(self):
        self.tuner.save_checkpoint()
        if self.assessor is not None:
            self.assessor.save_checkpoint()

    def handle_initialize(self, command: Initialize):
        """Data is search space
        """
        self.tuner.update_search_space(command.search_space)
        new_command = Initialized('Initialized')
        self.send(new_command)

    def send_trial_callback(self, id_, params):
        """For tuner to issue trial config when the config is generated
        """
        command_dict = _pack_parameter(id_, params)
        command_dict['command_type'] = 'NewTrialJob'
        command = NewTrialJob(**command_dict)
        self.send(command)

    def handle_request_trial_jobs(self, command: RequestTrialJobs):
        # data: number or trial jobs
        ids = [_create_parameter_id() for _ in range(command.job_num)]
        _logger.debug("requesting for generating params of %s", ids)
        params_list = self.tuner.generate_multiple_parameters(ids, st_callback=self.send_trial_callback)

        for i, _ in enumerate(params_list):
            command_dict ={}
            command_dict = _pack_parameter(ids[i], params_list[i])
            command_dict['command_type'] = 'NewTrialJob'
            self.send(NewTrialJob(**command_dict))
        # when parameters is None.
        if len(params_list) < len(ids):
            command_dict = {}
            command_dict = _pack_parameter(ids[0], '')
            command_dict['command_type'] = 'NoMoreTrialJobs'
            self.send(NoMoreTrialJobs(**command_dict))

    def handle_update_search_space(self, command: UpdateSearchSpace):
        self.tuner.update_search_space(command.search_space)

    def handle_import_data(self, command: ImportData):
        """Import additional data for tuning
        data: a list of dictionaries, each of which has at least two keys, 'parameter' and 'value'
        """
        for entry in command.data:
            entry['value'] = entry['value'] if type(entry['value']) is str else dump(entry['value'])
            entry['value'] = load(entry['value'])
        self.tuner.import_data(command.data)

    def handle_add_customized_trial(self, command: AddCustomizedTrialJob):
        # data: parameters
        id_ = _create_parameter_id()
        _customized_parameter_ids.add(id_)

    def handle_report_metric_data(self, command: ReportMetricData):
        """
        data: a dict received from nni_manager, which contains:
              - 'parameter_id': id of the trial
              - 'value': metric value reported by nni.report_final_result()
              - 'type': report type, support {'FINAL', 'PERIODICAL'}
        """
        # metrics value is dumped as json string in trial, so we need to decode it here
        if command.value is not None:
            command.value = load(command.value)
        if command.type == MetricType.FINAL:
            self._handle_final_metric_data(command)
        elif command.type == MetricType.PERIODICAL:
            if self.assessor is not None:
                self._handle_intermediate_metric_data(command)
        elif command.type == MetricType.REQUEST_PARAMETER:
            assert multi_phase_enabled()
            assert command.trial_job_id is not None
            assert command.parameter_index is not None
            param_id = _create_parameter_id()
            try:
                param = self.tuner.generate_parameters(param_id, trial_job_id=command.trial_job_id)
            except NoMoreTrialError:
                param = None
            command_dict = _pack_parameter(param_id, param, trial_job_id=command.trial_job_id,
                                                                    parameter_index=command.parameter_index)
            command_dict['command_type'] = 'SendTrialJobParameter'
            new_command = SendTrialJobParameter(**command_dict)
            self.send(new_command)
        else:
            raise ValueError('Data type not supported: {}'.format(command.type))

    def handle_trial_end(self, command: TrialEnd):
        """
        data: it has three keys: trial_job_id, event, hyper_params
             - trial_job_id: the id generated by training service
             - event: the job's state
             - hyper_params: the hyperparameters generated and returned by tuner
        """
        trial_job_id = command.trial_job_id
        _ended_trials.add(trial_job_id)
        if trial_job_id in _trial_history:
            _trial_history.pop(trial_job_id)
            if self.assessor is not None:
                self.assessor.trial_end(trial_job_id, command.event == 'SUCCEEDED')
        if self.tuner is not None:
            self.tuner.trial_end(command.hyper_params['parameter_id'], command.event == 'SUCCEEDED')

    def _handle_final_metric_data(self, command: ReportMetricData):
        """Call tuner to process final results
        """
        id_ = command.parameter_id
        value = command.value
        if id_ is None or id_ in _customized_parameter_ids:
            if not hasattr(self.tuner, '_accept_customized'):
                self.tuner._accept_customized = False
            if not self.tuner._accept_customized:
                _logger.info('Customized trial job %s ignored by tuner', id_)
                return
            customized = True
        else:
            customized = False
        if id_ in _trial_params:
            self.tuner.receive_trial_result(id_, _trial_params[id_], value, customized=customized,
                                            trial_job_id=command.trial_job_id)
        else:
            _logger.warning('Find unknown job parameter id %s, maybe something goes wrong.', _trial_params[id_])

    def _handle_intermediate_metric_data(self, command: ReportMetricData):
        """Call assessor to process intermediate results
        """
        if command.type != MetricType.PERIODICAL:
            return
        if self.assessor is None:
            return

        trial_job_id = command.trial_job_id
        if trial_job_id in _ended_trials:
            return

        history = _trial_history[trial_job_id]
        history[command.sequence] = command.value
        ordered_history = _sort_history(history)
        if len(ordered_history) < command.sequence:  # no user-visible update since last time
            return

        try:
            result = self.assessor.assess_trial(trial_job_id, ordered_history)
        except Exception as e:
            _logger.error('Assessor error')
            _logger.exception(e)
            raise

        if isinstance(result, bool):
            result = AssessResult.Good if result else AssessResult.Bad
        elif not isinstance(result, AssessResult):
            msg = 'Result of Assessor.assess_trial must be an object of AssessResult, not %s'
            raise RuntimeError(msg % type(result))

        if result is AssessResult.Bad:
            _logger.debug('BAD, kill %s', trial_job_id)
            command_dict = {}
            command_dict['command_type'] = 'KillTrialJob'
            command_dict['trial_job_id'] = trial_job_id
            new_command = KillTrialJob(**command_dict)
            self.send(new_command)
            # notify tuner
            _logger.debug('env var: NNI_INCLUDE_INTERMEDIATE_RESULTS: [%s]',
                          dispatcher_env_vars.NNI_INCLUDE_INTERMEDIATE_RESULTS)
            if dispatcher_env_vars.NNI_INCLUDE_INTERMEDIATE_RESULTS == 'true':
                self._earlystop_notify_tuner(command)
        else:
            _logger.debug('GOOD')

    def _earlystop_notify_tuner(self, command: ReportMetricData):
        """Send last intermediate result as final result to tuner in case the
        trial is early stopped.
        """
        _logger.debug('Early stop notify tuner data: [%s]', command)
        command.type = MetricType.FINAL
        command.value = dump(command.value)
        self.enqueue_command(command)
