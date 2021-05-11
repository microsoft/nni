import atexit
import logging
from pathlib import Path
import socket
from subprocess import Popen
import time
from typing import Optional, Union, List, overload, Any

import json_tricks
import colorama
import psutil

import nni.runtime.log

from .config import ExperimentConfig, AlgorithmConfig
from .data import TrialJob, TrialMetricData, TrialResult
from . import launcher
from . import management
from . import rest
from ..tools.nnictl.command_utils import kill_command

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
    def __init__(self, config: ExperimentConfig) -> None:
        """
        Prepare an experiment.

        Use `Experiment.run()` to launch it.

        Parameters
        ----------
        config
            Experiment configuration.
        """
        ...

    @overload
    def __init__(self, training_service: Union[str, List[str]]) -> None:
        """
        Prepare an experiment, leaving configuration fields to be set later.

        Example usage::

            experiment = Experiment('remote')
            experiment.config.trial_command = 'python3 trial.py'
            experiment.config.machines.append(RemoteMachineConfig(ip=..., user_name=...))
            ...
            experiment.run(8080)

        Parameters
        ----------
        training_service
            Name of training service.
            Supported value: "local", "remote", "openpai", "aml", "kubeflow", "frameworkcontroller", "adl" and hybrid training service.
        """
        ...

    def __init__(self, config=None, training_service=None):
        nni.runtime.log.init_logger_experiment()

        self.config: Optional[ExperimentConfig] = None
        self.id: Optional[str] = None
        self.port: Optional[int] = None
        self._proc: Optional[Popen] = None
        self.mode = 'new'

        args = [config, training_service]  # deal with overloading
        if isinstance(args[0], (str, list)):
            self.config = ExperimentConfig(args[0])
            self.config.tuner = AlgorithmConfig(name='_none_', class_args={})
            self.config.assessor = AlgorithmConfig(name='_none_', class_args={})
            self.config.advisor = AlgorithmConfig(name='_none_', class_args={})
        else:
            self.config = args[0]

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

        if self.mode == 'new':
            self.id = management.generate_experiment_id()
        else:
            self.config = launcher.get_stopped_experiment_config(self.id, self.mode)

        if self.config.experiment_working_directory is not None:
            log_dir = Path(self.config.experiment_working_directory, self.id, 'log')
        else:
            log_dir = Path.home() / f'nni-experiments/{self.id}/log'
        nni.runtime.log.start_experiment_log(self.id, log_dir, debug)

        self._proc = launcher.start_experiment(self.id, self.config, port, debug, mode=self.mode)
        assert self._proc is not None

        self.port = port  # port will be None if start up failed

        ips = [self.config.nni_manager_ip]
        for interfaces in psutil.net_if_addrs().values():
            for interface in interfaces:
                if interface.family == socket.AF_INET:
                    ips.append(interface.address)
        ips = [f'http://{ip}:{port}' for ip in ips if ip]
        msg = 'Web UI URLs: ' + colorama.Fore.CYAN + ' '.join(ips) + colorama.Style.RESET_ALL
        _logger.info(msg)

    def stop(self) -> None:
        """
        Stop background experiment.
        """
        _logger.info('Stopping experiment, please wait...')
        atexit.unregister(self.stop)

        if self.id is not None:
            nni.runtime.log.stop_experiment_log(self.id)
        if self._proc is not None:
            try:
                rest.delete(self.port, '/experiment')
            except Exception as e:
                _logger.exception(e)
                _logger.warning('Cannot gracefully stop experiment, killing NNI process...')
                kill_command(self._proc.pid)

        self.id = None
        self.port = None
        self._proc = None
        _logger.info('Experiment stopped')

    def run(self, port: int = 8080, wait_completion: bool = True, debug: bool = False) -> bool:
        """
        Run the experiment.

        If wait_completion is True, this function will block until experiment finish or error.

        Return `True` when experiment done; or return `False` when experiment failed.

        Else if wait_completion is False, this function will non-block and return None immediately.
        """
        self.start(port, debug)
        if wait_completion:
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

    @classmethod
    def connect(cls, port: int):
        """
        Connect to an existing experiment.

        Parameters
        ----------
        port
            The port of web UI.
        """
        experiment = Experiment()
        experiment.port = port
        experiment.id = experiment.get_experiment_profile().get('id')
        status = experiment.get_status()
        pid = experiment.get_experiment_metadata(experiment.id).get('pid')
        if pid is None:
            _logger.warning('Get experiment pid failed, can not stop experiment by stop().')
        else:
            experiment._proc = psutil.Process(pid)
        _logger.info('Connect to port %d success, experiment id is %s, status is %s.', port, experiment.id, status)
        return experiment

    @classmethod
    def resume(cls, experiment_id: str, port: int = 8080, wait_completion: bool = True, debug: bool = False):
        """
        Resume a stopped experiment.

        Parameters
        ----------
        experiment_id
            The stopped experiment id.
        port
            The port of web UI.
        wait_completion
            If true, run in the foreground. If false, run in the background.
        debug
            Whether to start in debug mode.
        """
        experiment = Experiment()
        experiment.id = experiment_id
        experiment.mode = 'resume'
        experiment.run(port=port, wait_completion=wait_completion, debug=debug)
        if not wait_completion:
            return experiment

    @classmethod
    def view(cls, experiment_id: str, port: int = 8080, non_blocking: bool = False):
        """
        View a stopped experiment.

        Parameters
        ----------
        experiment_id
            The stopped experiment id.
        port
            The port of web UI.
        non_blocking
            If false, run in the foreground. If true, run in the background.
        """
        debug = False
        experiment = Experiment()
        experiment.id = experiment_id
        experiment.mode = 'view'
        experiment.start(port=port, debug=debug)
        if non_blocking:
            return experiment
        else:
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                _logger.warning('KeyboardInterrupt detected')
            finally:
                experiment.stop()

    def get_status(self) -> str:
        """
        Return experiment status as a str.

        Returns
        -------
        str
            Experiment status.
        """
        resp = rest.get(self.port, '/check-status')
        return resp['status']

    def get_trial_job(self, trial_job_id: str):
        """
        Return a trial job.

        Parameters
        ----------
        trial_job_id: str
            Trial job id.

        Returns
        -------
        TrialJob
            A `TrialJob` instance corresponding to `trial_job_id`.
        """
        resp = rest.get(self.port, '/trial-jobs/{}'.format(trial_job_id))
        return TrialJob(**resp)

    def list_trial_jobs(self):
        """
        Return information for all trial jobs as a list.

        Returns
        -------
        list
            List of `TrialJob`.
        """
        resp = rest.get(self.port, '/trial-jobs')
        return [TrialJob(**trial_job) for trial_job in resp]

    def get_job_statistics(self):
        """
        Return trial job statistics information as a dict.

        Returns
        -------
        dict
            Job statistics information.
        """
        resp = rest.get(self.port, '/job-statistics')
        return resp

    def get_job_metrics(self, trial_job_id=None):
        """
        Return trial job metrics.

        Parameters
        ----------
        trial_job_id: str
            trial job id. if this parameter is None, all trail jobs' metrics will be returned.

        Returns
        -------
        dict
            Each key is a trialJobId, the corresponding value is a list of `TrialMetricData`.
        """
        api = '/metric-data/{}'.format(trial_job_id) if trial_job_id else '/metric-data'
        resp = rest.get(self.port, api)
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
        -------
        dict
            The profile of the experiment.
        """
        resp = rest.get(self.port, '/experiment')
        return resp

    def get_experiment_metadata(self, exp_id: str):
        """
        Return experiment metadata with specified exp_id as a dict.

        Returns
        -------
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
        -------
        list
            The experiments metadata.
        """
        resp = rest.get(self.port, '/experiments-info')
        return resp

    def export_data(self):
        """
        Return exported information for all trial jobs.

        Returns
        -------
        list
            List of `TrialResult`.
        """
        resp = rest.get(self.port, '/export-data')
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
        rest.put(self.port, api, experiment_profile)
        logging.info('Successfully update %s.', key)

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
        value = json_tricks.dumps(value)
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
