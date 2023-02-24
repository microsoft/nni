# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import contextlib
from dataclasses import dataclass, fields
from datetime import datetime
import logging
import os.path
from pathlib import Path
import socket
from subprocess import Popen
import sys
import time
from typing import Any, TYPE_CHECKING, cast

from typing_extensions import Literal

from .config import ExperimentConfig
from . import rest
from ..tools.nnictl.config_utils import Experiments, Config
from ..tools.nnictl.nnictl_utils import update_experiment

if TYPE_CHECKING:
    from .experiment import RunMode

_logger = logging.getLogger('nni.experiment')

@dataclass(init=False)
class NniManagerArgs:
    # argv sent to "ts/nni_manager/main.js"

    port: int
    experiment_id: str
    action: Literal['create', 'resume', 'view']
    mode: str  # training service platform, to be removed
    experiments_directory: str  # renamed "config.nni_experiments_directory", must be absolute
    log_level: str
    foreground: bool = False
    url_prefix: str | None = None  # leading and trailing "/" must be stripped
    tuner_command_channel: str | None = None
    python_interpreter: str

    def __init__(self,
            action: Literal['create', 'resume', 'view'],
            exp_id: str,
            config: ExperimentConfig,
            port: int,
            debug: bool,
            foreground: bool,
            url_prefix: str | None,
            tuner_command_channel: str | None):
        self.port = port
        self.experiment_id = exp_id
        self.action = action
        self.foreground = foreground
        self.url_prefix = url_prefix
        # config field name "experiment_working_directory" is a mistake
        # see "ts/nni_manager/common/globals/arguments.ts" for details
        self.experiments_directory = cast(str, config.experiment_working_directory)
        self.tuner_command_channel = tuner_command_channel
        self.python_interpreter = sys.executable

        if isinstance(config.training_service, list):
            self.mode = 'hybrid'
        else:
            self.mode = config.training_service.platform

        self.log_level = cast(str, config.log_level)
        if debug and self.log_level not in ['debug', 'trace']:
            self.log_level = 'debug'

    def to_command_line_args(self) -> list[str]:
        # reformat fields to meet yargs library's format
        # see "ts/nni_manager/common/globals/arguments.ts" for details
        ret = []
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                ret.append('--' + field.name.replace('_', '-'))
                if isinstance(value, bool):
                    ret.append(str(value).lower())
                else:
                    ret.append(str(value))
        return ret

def start_experiment(
        action: Literal['create', 'resume', 'view'],
        exp_id: str,
        config: ExperimentConfig,
        port: int,
        debug: bool,
        run_mode: RunMode,
        url_prefix: str | None,
        tuner_command_channel: str | None = None,
        tags: list[str] = []) -> Popen:

    foreground = run_mode.value == 'foreground'
    if url_prefix is not None:
        url_prefix = url_prefix.strip('/')
    nni_manager_args = NniManagerArgs(action, exp_id, config, port, debug, foreground, url_prefix, tuner_command_channel)

    _ensure_port_idle(port)
    websocket_platforms = ['hybrid', 'remote', 'openpai', 'kubeflow', 'frameworkcontroller', 'adl']
    if action != 'view' and nni_manager_args.mode in websocket_platforms:
        _ensure_port_idle(port + 1, f'{nni_manager_args.mode} requires an additional port')

    link = Path(config.experiment_working_directory, '_latest')
    try:
        if link.exists():
            link.unlink()
        link.symlink_to(exp_id, target_is_directory=True)
    except Exception:
        if sys.platform != 'win32':
            _logger.warning(f'Failed to create link {link}')

    proc = None
    try:
        _logger.info('Creating experiment, Experiment ID: ${CYAN}%s', exp_id)
        proc = _start_rest_server(nni_manager_args, run_mode)
        start_time = int(time.time() * 1000)

        _logger.info('Starting web server...')
        _check_rest_server(port, url_prefix=url_prefix)

        Experiments().add_experiment(
            exp_id,
            port,
            start_time,
            nni_manager_args.mode,
            config.experiment_name,
            pid=proc.pid,
            logDir=cast(str, config.experiment_working_directory),
            tag=tags,
            prefixUrl=url_prefix
        )

        _logger.info('Setting up...')
        rest.post(port, '/experiment', config.json(), url_prefix)

    except Exception as e:
        _logger.error('Create experiment failed: %s', e)
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.kill()

        log = Path(nni_manager_args.experiments_directory, nni_manager_args.experiment_id, 'log', 'nnictl_stderr.log')
        if log.exists():
            _logger.warning('NNI manager stderr:')
            _logger.warning(log.read_text())

        raise

    return proc

def _start_rest_server(nni_manager_args: NniManagerArgs, run_mode: RunMode) -> Popen:
    import nni_node
    node_dir = Path(nni_node.__path__[0])  # type: ignore
    node = str(node_dir / ('node.exe' if sys.platform == 'win32' else 'node'))
    main_js = str(node_dir / 'main.js')
    cmd = [node, '--max-old-space-size=4096', '--trace-uncaught', main_js]
    cmd += nni_manager_args.to_command_line_args()

    if run_mode.value == 'detach':
        log = Path(nni_manager_args.experiments_directory, nni_manager_args.experiment_id, 'log')
        out = (log / 'nnictl_stdout.log').open('a')
        err = (log / 'nnictl_stderr.log').open('a')
        header = f'Experiment {nni_manager_args.experiment_id} start: {datetime.now()}'
        header = '-' * 80 + '\n' + header + '\n' + '-' * 80 + '\n'
        out.write(header)
        err.write(header)

    else:
        out = None
        err = None

    if sys.platform == 'win32':
        from subprocess import CREATE_NEW_PROCESS_GROUP
        return Popen(cmd, stdout=out, stderr=err, cwd=node_dir, creationflags=CREATE_NEW_PROCESS_GROUP)
    else:
        return Popen(cmd, stdout=out, stderr=err, cwd=node_dir, preexec_fn=os.setpgrp)  # type: ignore


def _ensure_port_idle(port: int, message: str | None = None) -> None:
    sock = socket.socket()
    if sock.connect_ex(('localhost', port)) == 0:
        sock.close()
        message = f'(message)' if message else ''
        raise RuntimeError(f'Port {port} is not idle {message}')


def _check_rest_server(port: int, retry: int = 3, url_prefix: str | None = None) -> None:
    for i in range(retry):
        with contextlib.suppress(Exception):
            rest.get(port, '/check-status', url_prefix)
            return
        if i > 0:
            _logger.warning('Timeout, retry...')
        time.sleep(1)
    rest.get(port, '/check-status', url_prefix)


def _save_experiment_information(experiment_id: str, port: int, start_time: int, platform: str,
                                 name: str, pid: int, logDir: str, tag: list[Any]) -> None:
    experiments_config = Experiments()
    experiments_config.add_experiment(experiment_id, port, start_time, platform, name, pid=pid, logDir=logDir, tag=tag)


def get_stopped_experiment_config(exp_id: str, exp_dir: str | Path | None = None) -> ExperimentConfig:
    """Get the experiment config of a stopped experiment.

    Parameters
    ----------
    exp_id
        The experiment ID.
    exp_dir
        The experiment working directory which is expected to contain a folder named ``exp_id``.

    Returns
    -------
    The config.
    It's the config returned by :func:`get_stopped_experiment_config_json`,
    loaded by :class:`ExperimentConfig`.
    """
    if isinstance(exp_dir, Path):
        exp_dir = str(exp_dir)
    config_json = get_stopped_experiment_config_json(exp_id, exp_dir)  # type: ignore
    if config_json is None:
        raise ValueError(f'Config of {exp_id} (under {exp_dir}) failed to be loaded.')
    config = ExperimentConfig(**config_json)  # type: ignore
    if exp_dir and not os.path.samefile(exp_dir, config.experiment_working_directory):
        msg = 'Experiment working directory provided in command line (%s) is different from experiment config (%s)'
        _logger.warning(msg, exp_dir, config.experiment_working_directory)
        config.experiment_working_directory = exp_dir
    return config


def get_stopped_experiment_config_json(exp_id: str, exp_dir: str | None = None) -> dict | None:
    """Get the experiment config, in JSON format, of a stopped experiment.

    Different from :func:`get_stopped_experiment_config`,
    this function does not load the config into an :class:`ExperimentConfig` object.
    It doesn't check the experiment directory contained inside the config JSON either.

    NOTE: The config is retrieved from SQL database, and should be written by NNI manager in current implementation.

    Parameters
    ----------
    exp_id
        The experiment ID.
    exp_dir
        The experiment working directory which is expected to contain a folder named ``exp_id``.
        If ``exp_dir`` is not provided, the directory will be retrieved from the manifest of all experiments.

    Returns
    -------
    The config JSON.
    """
    if exp_dir:
        return Config(exp_id, exp_dir).get_config()
    else:
        update_experiment()
        experiments_config = Experiments()
        experiments_dict = experiments_config.get_all_experiments()
        experiment_metadata = experiments_dict.get(exp_id)
        if experiment_metadata is None:
            _logger.error('Id %s not exist!', exp_id)
            return None
        if experiment_metadata['status'] != 'STOPPED':
            _logger.error(
                'Only stopped experiments can be resumed or viewed! But retrieved metadata for %s is:\n%s',
                exp_id, experiment_metadata
            )
            return None
        return Config(exp_id, experiment_metadata['logDir']).get_config()
