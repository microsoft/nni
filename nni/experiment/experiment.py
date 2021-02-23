import atexit
import logging
from pathlib import Path
import socket
from subprocess import Popen
from threading import Thread
import time
from typing import Optional, Union, List, overload, Any

import colorama
import psutil

import nni.runtime.log
from nni.runtime.msg_dispatcher import MsgDispatcher
from nni.tuner import Tuner

from .config import ExperimentConfig
from .data import TrialJob, TrialMetricData, TrialResult
from . import launcher
from . import management
from .pipe import Pipe
from . import rest
from ..tools.nnictl.command_utils import kill_command

nni.runtime.log.init_logger_experiment()
_logger = logging.getLogger('nni.experiment')


class Experiment:
    """
    Create and stop an NNI experiment.

    Attributes
    ----------
    config
        Experiment configuration.
    port
        Web UI port of the experiment, or `None` if it is not running.
    """

    @overload
    def __init__(self, tuner: Tuner, config: ExperimentConfig) -> None:
        """
        Prepare an experiment.

        Use `Experiment.start()` to launch it.

        Parameters
        ----------
        tuner
            A tuner instance.
        config
            Experiment configuration.
        """
        ...

    @overload
    def __init__(self, tuner: Tuner, training_service: Union[str, List[str]]) -> None:
        """
        Prepare an experiment, leaving configuration fields to be set later.

        Example usage::

            experiment = Experiment(my_tuner, 'remote')
            experiment.config.trial_command = 'python3 trial.py'
            experiment.config.machines.append(RemoteMachineConfig(ip=..., user_name=...))
            ...
            experiment.start(8080)

        Parameters
        ----------
        tuner
            A tuner instance.
        training_service
            Name of training service.
            Supported value: "local", "remote", "openpai", "aml", "kubeflow", "frameworkcontroller", "adl" and hybrid training service.
        """
        ...

    @overload
    def __init__(self) -> None:
        """
        Prepare an empty experiment, for `connect_experiment`.

        Use `Experiment.connect_experiment` to manage experiment.

        """
        ...

    def __init__(self, tuner=None, config=None, training_service=None):
        self.config: Optional[ExperimentConfig] = None
        self.id: Optional[str] = None
        self.port: Optional[int] = None
        self.tuner: Optional[Tuner] = None
        self._proc: Optional[Popen] = None
        self._pipe: Optional[Pipe] = None
        self._dispatcher: Optional[MsgDispatcher] = None
        self._dispatcher_thread: Optional[Thread] = None

        if isinstance(tuner, Tuner):
            self.tuner = tuner
            if isinstance(config, (str, list)):
                config, training_service = None, config

            if config is None:
                self.config = ExperimentConfig(training_service)
            else:
                self.config = config
        else:
            _logger.warning('Tuner not set, wait for connect...')

    def start(self, port: int = 8080, debug: bool = False) -> None:
        """
        Start the experiment in background.

        This method will raise exception on failure.
        If it returns, the experiment should have been successfully started.

        Parameters
        ----------
        port
            The port of web UI.
        debug
            Whether to start in debug mode.
        """
        atexit.register(self.stop)

        self.id = management.generate_experiment_id()

        if self.config.experiment_working_directory is not None:
            log_dir = Path(self.config.experiment_working_directory, self.id, 'log')
        else:
            log_dir = Path.home() / f'nni-experiments/{self.id}/log'
        nni.runtime.log.start_experiment_log(self.id, log_dir, debug)

        self._proc, self._pipe = launcher.start_experiment(self.id, self.config, port, debug)
        assert self._proc is not None
        assert self._pipe is not None

        self.port = port  # port will be None if start up failed

        # dispatcher must be launched after pipe initialized
        # the logic to launch dispatcher in background should be refactored into dispatcher api
        self._dispatcher = self._create_dispatcher()
        self._dispatcher_thread = Thread(target=self._dispatcher.run)
        self._dispatcher_thread.start()

        ips = [self.config.nni_manager_ip]
        for interfaces in psutil.net_if_addrs().values():
            for interface in interfaces:
                if interface.family == socket.AF_INET:
                    ips.append(interface.address)
        ips = [f'http://{ip}:{port}' for ip in ips if ip]
        msg = 'Web UI URLs: ' + colorama.Fore.CYAN + ' '.join(ips) + colorama.Style.RESET_ALL
        _logger.info(msg)

    def _create_dispatcher(self):  # overrided by retiarii, temporary solution
        return MsgDispatcher(self.tuner, None)

    def stop(self) -> None:
        """
        Stop background experiment.
        """
        _logger.info('Stopping experiment, please wait...')
        atexit.unregister(self.stop)

        if self.id is not None:
            nni.runtime.log.stop_experiment_log(self.id)
        if self._proc is not None:
            kill_command(self._proc.pid)
        if self._pipe is not None:
            self._pipe.close()
        if self._dispatcher_thread is not None:
            self._dispatcher.stopping = True
            self._dispatcher_thread.join(timeout=1)

        self.id = None
        self.port = None
        self._proc = None
        self._pipe = None
        self._dispatcher = None
        self._dispatcher_thread = None
        _logger.info('Experiment stopped')

    def run(self, port: int = 8080, debug: bool = False) -> bool:
        """
        Run the experiment.

        This function will block until experiment finish or error.

        Return `True` when experiment done; or return `False` when experiment failed.
        """
        self.start(port, debug)
        try:
            while True:
                time.sleep(10)
                status = self.get_status()
                if status == 'DONE' or status == 'STOPPED':
                    return True
                if status == 'ERROR':
                    return False
        except KeyboardInterrupt:
            _logger.warning('KeyboardInterrupt detected')
        finally:
            self.stop()

    def connect(self, port: int):
        """
        Connect to an existing experiment.

        Parameters
        ----------
        port
            The port of web UI.
        """
        self.port = port
        self.id = self.get_experiment_profile().get('id')
        status = self.get_status()
        pid = self.get_experiment_metadata(self.id).get('pid')
        if pid is None:
            _logger.warning('Get experiment pid failed, can not stop experiment by stop().')
        else:
            self._proc = psutil.Process(pid)
        _logger.info('Connect to port %d success, experiment id is %s, status is %s.' % (port, self.id, status))

    def _experiment_rest_get(self, port: int, api: str) -> Any:
        if self.port is None:
            raise RuntimeError('Experiment is not running')
        return rest.get(self.port, api)

    def _experiment_rest_put(self, port: int, api: str, data: Any):
        if self.port is None:
            raise RuntimeError('Experiment is not running')
        rest.put(self.port, api, data)

    def get_status(self) -> str:
        """
        Return experiment status as a str.

        Returns
        -------
        str
            Experiment status.
        """
        resp = self._experiment_rest_get(self.port, '/check-status')
        return resp['status']

    def get_trial_job(self, trial_job_id: str):
        """
        Return a trial job.

        Parameters
        ----------
        trial_job_id: str
            Trial job id.

        Returns
        ----------
        TrialJob
            A `TrialJob` instance corresponding to `trial_job_id`.
        """
        resp = self._experiment_rest_get(self.port, '/trial-jobs/{}'.format(trial_job_id))
        return TrialJob(**resp)

    def list_trial_jobs(self):
        """
        Return information for all trial jobs as a list.

        Returns
        ----------
        list
            List of `TrialJob`.
        """
        resp = self._experiment_rest_get(self.port, '/trial-jobs')
        return [TrialJob(**trial_job) for trial_job in resp]

    def get_job_statistics(self):
        """
        Return trial job statistics information as a dict.

        Returns
        ----------
        dict
            Job statistics information.
        """
        resp = self._experiment_rest_get(self.port, '/job-statistics')
        return resp

    def get_job_metrics(self, trial_job_id=None):
        """
        Return trial job metrics.

        Parameters
        ----------
        trial_job_id: str
            trial job id. if this parameter is None, all trail jobs' metrics will be returned.

        Returns
        ----------
        dict
            Each key is a trialJobId, the corresponding value is a list of `TrialMetricData`.
        """
        api = '/metric-data/{}'.format(trial_job_id) if trial_job_id else '/metric-data'
        resp = self._experiment_rest_get(self.port, api)
        metric_dict = {}
        for metric in resp:
            trial_id = metric["trialJobId"]
            if trial_id not in metric_dict:
                metric_dict[trial_id] = [TrialMetricData(**metric)]
            else:
                metric_dict[trial_id].append(TrialMetricData(**metric))
        return metric_dict

    def get_experiment_profile(self):
        """
        Return experiment profile as a dict.

        Returns
        ----------
        dict
            The profile of the experiment.
        """
        resp = self._experiment_rest_get(self.port, '/experiment')
        return resp

    def get_experiment_metadata(self, exp_id: str):
        """
        Return experiment metadata with specified exp_id as a dict.

        Returns
        ----------
        dict
            The specified experiment metadata.
        """
        experiments_metadata = self.get_all_experiments_metadata()
        for metadata in experiments_metadata:
            if metadata['id'] == exp_id:
                return metadata
        return {}

    def get_all_experiments_metadata(self):
        """
        Return all experiments metadata as a list.

        Returns
        ----------
        list
            The experiments metadata.
            Example format: [
                                {
                                    "id": str,
                                    "port": int,
                                    "startTime": int,
                                    "endTime": int,
                                    "status": str,
                                    "platform": str,
                                    "experimentName": str,
                                    "tag": [str],
                                    "pid": int,
                                    "webuiUrl": [str],
                                    "logDir": str
                                },
                                {...}
                            ]
        """
        resp = self._experiment_rest_get(self.port, '/experiments-info')
        return resp

    def export_data(self):
        """
        Return exported information for all trial jobs.

        Returns
        ----------
        list
            List of `TrialResult`.
        """
        resp = self._experiment_rest_get(self.port, '/export-data')
        return [TrialResult(**trial_result) for trial_result in resp]

    def _get_query_type(self, key: str):
        if key == 'trialConcurrency':
            return '?update_type=TRIAL_CONCURRENCY'
        if key == 'maxExecDuration':
            return '?update_type=MAX_EXEC_DURATION'
        if key == 'searchSpace':
            return '?update_type=SEARCH_SPACE'
        if key == 'maxTrialNum':
            return '?update_type=MAX_TRIAL_NUM'

    def _update_experiment_profile(self, key: str, value: Any):
        """
        Update an experiment's profile

        Parameters
        ----------
        key: str
            One of `['trial_concurrency', 'max_experiment_duration', 'search_space', 'max_trial_number']`.
        value: Any
            New value of the key.
        """
        api = '/experiment{}'.format(self._get_query_type(key))
        experiment_profile = self.get_experiment_profile()
        experiment_profile['params'][key] = value
        self._experiment_rest_put(self.port, api, experiment_profile)

    def update_trial_concurrency(self, value: int):
        """
        Update an experiment's trial_concurrency

        Parameters
        ----------
        value: int
            New trial_concurrency value.
        """
        self._update_experiment_profile('trialConcurrency', value)

    def update_max_experiment_duration(self, value: str):
        """
        Update an experiment's max_experiment_duration

        Parameters
        ----------
        value: str
            Strings like '1m' for one minute or '2h' for two hours.
            SUFFIX may be 's' for seconds, 'm' for minutes, 'h' for hours or 'd' for days.
        """
        self._update_experiment_profile('maxExecDuration', value)

    def update_search_space(self, value: dict):
        """
        Update the experiment's search_space.
        TODO: support searchspace file.

        Parameters
        ----------
        value: dict
            New search_space.
        """
        self._update_experiment_profile('searchSpace', value)

    def update_max_trial_number(self, value: int):
        """
        Update an experiment's max_trial_number

        Parameters
        ----------
        value: int
            New max_trial_number value.
        """
        self._update_experiment_profile('maxTrialNum', value)
