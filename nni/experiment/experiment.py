# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import atexit
from enum import Enum
import logging
from pathlib import Path
import socket
from subprocess import Popen
import time
from typing import Any, cast

import psutil
from typing_extensions import Literal

from nni.runtime.log import start_experiment_logging, stop_experiment_logging
from nni.tools.nnictl.config_utils import Experiments

from .config import ExperimentConfig
from .data import TrialJob, TrialMetricData, TrialResult
from . import launcher
from . import management
from . import rest
from ..tools.nnictl.command_utils import kill_command

_logger = logging.getLogger('nni.experiment')

class RunMode(Enum):
    """
    Config lifecycle and ouput redirection of NNI manager process.

    - Background: stop NNI manager when Python script exits; do not print NNI manager log. (default)
    - Foreground: stop NNI manager when Python script exits; print NNI manager log to stdout.
    - Detach: do not stop NNI manager when Python script exits.

    NOTE: This API is non-stable and is likely to get refactored in upcoming release.
    """
    # TODO:
    # NNI manager should treat log level more seriously so we can default to "foreground" without being too verbose.
    Background = 'background'
    Foreground = 'foreground'
    Detach = 'detach'

class Experiment:
    """
    Manage NNI experiment.

    You can either specify an :class:`ExperimentConfig` object, or a training service name.
    If a platform name is used, a blank config template for that training service will be generated.

    When configuration is completed, use :meth:`Experiment.run` to launch the experiment.

    Parameters
    ----------
    config_or_platform
        See :class:`~nni.experiment.config.ExperimentConfig`.
    id
        Experiment ID. If not specified, a random ID will be generated.
        If specified, the ID should be unique to avoid conflict with existing experiments.
        The only case when you need to specify an existing ID is when you want to resume an experiment.

    Example
    -------
    .. code-block::

        experiment = Experiment('remote')
        experiment.config.trial_command = 'python3 trial.py'
        experiment.config.machines.append(RemoteMachineConfig(ip=..., user_name=...))
        ...
        experiment.run(8080)

    Attributes
    ----------
    config
        Experiment configuration.
    id
        Experiment ID.
    port
        Web portal port. Or ``None`` if the experiment is not running.
    """

    def __init__(
        self,
        config_or_platform: ExperimentConfig | str | list[str] | None,
        id: str | None = None  # pylint: disable=redefined-builtin
    ):
        self.config: ExperimentConfig | None = None
        if id is not None:
            if not management.is_valid_experiment_id(id):
                raise ValueError(f'Invalid experiment ID: {id}. Experiment ID should only contain digits, alphanumeric characters, '
                                 'hyphens, and underscores, and should be no longer than 32 characters.')
            self.id = id
        else:
            self.id = management.generate_experiment_id()
        self.port: int | None = None
        self._proc: Popen | psutil.Process | None = None
        self._action: Literal['create', 'resume', 'view'] = 'create'
        self.url_prefix: str | None = None

        if isinstance(config_or_platform, (str, list)):
            self.config = ExperimentConfig(config_or_platform)
        else:
            self.config = config_or_platform

    def _start_logging(self, debug: bool) -> None:
        assert self.config is not None

        config = self.config.canonical_copy()

        log_file = Path(config.experiment_working_directory, self.id, 'log', 'experiment.log')
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_level = 'debug' if (debug or config.log_level == 'trace') else config.log_level
        start_experiment_logging(self.id, log_file, cast(str, log_level))

    def _start_nni_manager(self, port: int, debug: bool, run_mode: RunMode = RunMode.Background,
                           tuner_command_channel: str | None = None,
                           tags: list[str] = []) -> None:
        assert self.config is not None
        config = self.config.canonical_copy()
        if config.use_annotation:
            raise RuntimeError('NNI annotation is not supported by Python experiment API.')

        self._proc = launcher.start_experiment(self._action, self.id, config, port, debug, run_mode,
                                               self.url_prefix, tuner_command_channel, tags)
        assert self._proc is not None

        self.port = port  # port will be None if start up failed

        ips = [config.nni_manager_ip]
        for interfaces in psutil.net_if_addrs().values():
            for interface in interfaces:
                if interface.family == socket.AF_INET:
                    ips.append(interface.address)
        ips = [f'http://{ip}:{port}' for ip in ips if ip]
        msg = 'Web portal URLs: ${CYAN}' + ' '.join(ips)
        _logger.info(msg)

    def start(self, port: int = 8080, debug: bool = False, run_mode: RunMode = RunMode.Background) -> None:
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
        run_mode
            Running the experiment in foreground or background
        """
        if run_mode is not RunMode.Detach:
            # If the experiment ends normally without KeyboardInterrupt, stop won't be automatically called.
            # As a result, NNI manager will continue to run in the background, even after run() exits.
            # To kill it, either call stop() manually, or atexit will clean it up at process exit.
            atexit.register(self.stop)

        self._start_logging(debug)
        self._start_nni_manager(port, debug, run_mode, None, [])

    def _stop_logging(self) -> None:
        stop_experiment_logging(self.id)

    def _stop_nni_manager(self) -> None:
        if self._proc is not None:
            try:
                rest.delete(self.port, '/experiment', self.url_prefix)
                self._proc.wait()
            except Exception as e:
                _logger.exception(e)
                _logger.warning('Cannot gracefully stop experiment, killing NNI process...')
                kill_command(self._proc.pid)

        self.port = None
        self._proc = None

    def stop(self) -> None:
        """
        Stop the experiment.
        """
        _logger.info('Stopping experiment, please wait...')
        atexit.unregister(self.stop)
        _logger.info('Saving experiment checkpoint...')
        self.save_checkpoint()
        _logger.info('Stopping NNI manager, if any...')
        self._stop_nni_manager()
        self._stop_logging()
        _logger.info('Experiment stopped.')

    def _wait_completion(self) -> bool:
        while True:
            status = self.get_status()
            if status == 'DONE' or status == 'STOPPED':
                return True
            if status == 'ERROR':
                return False
            time.sleep(10)

    def _run_impl(self, port: int, wait_completion: bool, debug: bool) -> bool | None:
        try:
            self.start(port, debug)
            if wait_completion:
                return self._wait_completion()
        except KeyboardInterrupt:
            _logger.warning('KeyboardInterrupt detected')
            self.stop()
        # NOTE: stop is not called if wait is successful without interrupt.
        return None

    def run(self, port: int = 8080, wait_completion: bool = True, debug: bool = False) -> bool | None:
        """
        Run the experiment.

        Using Ctrl-C will :meth:`stop` the experiment.
        Otherwise the experiment won't be :meth:`stop`ped even if the method returns.
        It has to be manually :meth:`stop`ped, or atexit will :meth:`stop` it at process exit.

        Parameters
        ----------
        port
            The port on which NNI manager will run. It will also be the port of web portal.
        wait_completion
            If ``wait_completion`` is ``True``, this function will block until experiment finish or error.
        debug
            Set log level to debug.

        Returns
        -------
        If ``wait_completion`` is ``False``, this function will non-block and return None immediately.
        Otherwise, return ``True`` when experiment done; or return ``False`` when experiment failed.
        """
        return self._run_impl(port, wait_completion, debug)

    def run_or_resume(self, port: int = 8080, wait_completion: bool = True, debug: bool = False) -> bool | None:
        """
        Call :meth:`run` or :meth:`resume` based on the return value of :meth:`has_checkpoint`.

        Parameters are return values are same as :meth:`run`.
        """
        if self.has_checkpoint():
            _logger.info('Checkpoint is found. Resume the experiment: %s', self.id)
            return self.resume(port, wait_completion, debug)
        else:
            _logger.info('No checkpoint is found. Start a new experiment: %s', self.id)
            return self.run(port, wait_completion, debug)

    def has_checkpoint(self) -> bool:
        """
        Check whether a checkpoint of current experiment ID exists.

        Returns
        -------
        ``True`` if checkpoint is found; ``False`` otherwise.
        """
        # First check whether a checkpoint exists.
        experiments_dict = Experiments().get_all_experiments()
        if self.id in experiments_dict:
            _logger.debug('Checkpoint is found in experiment manifest. The experiment can be resumed: %r', experiments_dict[self.id])
            return True
        else:
            _logger.debug('No checkpoint with %s is found in experiment manifest.', self.id)
            return False

    def load_checkpoint(self) -> None:
        """
        Load checkpoint from local file system.
        Restores the status of the experiment instance.
        """
        # HPO basically only needs to load the config.

        # In case the current experiment already has a config,
        # respect the new config's working directory.
        if self.config is not None:
            experiment_working_directory = self.config.canonical_copy().experiment_working_directory
        else:
            experiment_working_directory = None

        # Load the config regardless of whether current config is provided or not.
        config = launcher.get_stopped_experiment_config(self.id, exp_dir=experiment_working_directory)

        if self.config is not None:
            # If `self.config` is set, do some validation.
            from .config.utils import diff
            config_diff = diff(self.config, config, 'Current', 'Loaded')
            if config_diff:
                _logger.warning('Config is found but does not match the current config:\n%s', config_diff)
                _logger.warning('Current config will NOT be overridden by the loaded config.')
            else:
                _logger.info('Current config matches the loaded config.')
        else:
            # If `self.config` is not set, use the loaded config.
            _logger.debug('Current config is None. Loaded config will be used: %r', config)
            self.config = config

    def save_checkpoint(self) -> None:
        """
        Save the experiment status to local file system.
        """
        # HPO experiment doesn't need to do this because the state has already been saved by underlying components.
        pass

    @classmethod
    def connect(cls, port: int):
        """
        Connect to an existing experiment.

        Parameters
        ----------
        port
            The port of web UI.
        """
        experiment = cls(None)
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

    def resume(self, port: int = 8080, wait_completion: bool = True, debug: bool = False) -> bool | None:
        """
        Resume a stopped experiment.

        Parameters
        ----------
        port
            The port of web UI.
        wait_completion
            If true, run in the foreground. If false, run in the background.
        debug
            Whether to start in debug mode.

        Returns
        -------
        See :meth:`run`.
        """
        # Backward compatibility:
        # We will stop supporting experiment_id as keyword arguments instantly right now,
        # because keeping it compatible will be very tricky and not worth the effort.
        # But experiment_id as positional argument is still supported for now.
        if isinstance(self, str):
            _logger.warning('Experiment.resume(id) is deprecated (and has already stopped working for non-HPO experiments). '
                            'Use Experiment(id).resume() instead.')
            # Assumes the type is `Experiment`, self is experiment_id.
            self = Experiment(None, id=self)

        if not self.has_checkpoint():
            raise RuntimeError(f'Experiment {self.id} does not exist thus cannot be resumed.')

        self.load_checkpoint()

        self._action = 'resume'

        return self._run_impl(port, wait_completion, debug)

    def view(self, port: int = 8080, non_blocking: bool = False) -> Experiment:
        """
        View a stopped experiment.

        Parameters
        ----------
        port
            The port of web UI.
        non_blocking
            If false, run in the foreground. If true, run in the background.

        Returns
        -------
        Return self instance.
        """
        # Backward compatibility
        if isinstance(self, str):
            _logger.warning('Experiment.view(id) is deprecated (and has already stopped working for non-HPO experiments). '
                            'Use Experiment(id).view() instead.')
            # Assumes the type is `Experiment`, self is experiment_id.
            self = Experiment(None, id=self)

        self._action = 'view'

        if not self.has_checkpoint():
            raise RuntimeError(f'Experiment {self.id} does not exist thus cannot be viewed.')

        self.load_checkpoint()

        self.start(port=port, debug=False, run_mode=RunMode.Detach)
        if not non_blocking:
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                _logger.warning('KeyboardInterrupt detected')
            finally:
                self.stop()
        return self

    def get_status(self) -> str:
        """
        Return experiment status as a str.

        Returns
        -------
        str
            Experiment status.
        """
        resp = rest.get(self.port, '/check-status', self.url_prefix)
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
        resp = rest.get(self.port, '/trial-jobs/{}'.format(trial_job_id), self.url_prefix)
        return TrialJob(**resp)

    def list_trial_jobs(self):
        """
        Return information for all trial jobs as a list.

        Returns
        -------
        list
            List of `TrialJob`.
        """
        resp = rest.get(self.port, '/trial-jobs', self.url_prefix)
        return [TrialJob(**trial_job) for trial_job in resp]

    def get_job_statistics(self):
        """
        Return trial job statistics information as a dict.

        Returns
        -------
        dict
            Job statistics information.
        """
        resp = rest.get(self.port, '/job-statistics', self.url_prefix)
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
        resp = rest.get(self.port, api, self.url_prefix)
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
        resp = rest.get(self.port, '/experiment', self.url_prefix)
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
        resp = rest.get(self.port, '/experiments-info', self.url_prefix)
        return resp

    def export_data(self):
        """
        Return exported information for all trial jobs.

        Returns
        -------
        list
            List of `TrialResult`.
        """
        resp = rest.get(self.port, '/export-data', self.url_prefix)
        return [TrialResult(**trial_result) for trial_result in resp]

    def _get_query_type(self, key: str):
        if key == 'trialConcurrency':
            return '?update_type=TRIAL_CONCURRENCY'
        if key == 'maxExperimentDuration':
            return '?update_type=MAX_EXEC_DURATION'
        if key == 'searchSpace':
            return '?update_type=SEARCH_SPACE'
        if key == 'maxTrialNumber':
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
        rest.put(self.port, api, experiment_profile, self.url_prefix)
        _logger.info('Successfully update %s.', key)

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
        self._update_experiment_profile('maxExperimentDuration', value)

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
        self._update_experiment_profile('maxTrialNumber', value)

    def kill_trial_job(self, trial_job_id: str):
        """
        Kill a trial job.

        Parameters
        ----------
        trial_job_id: str
            Trial job id.

        """
        rest.delete(self.port, '/trial-jobs/{}'.format(trial_job_id), self.url_prefix)

