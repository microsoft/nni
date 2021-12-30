# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
from typing import Optional, Tuple, List, Any

import colorama

import nni_node  # pylint: disable=wrong-import-order, import-error
import nni.runtime.protocol

from .config import ExperimentConfig
from .pipe import Pipe
from . import rest
from ..tools.nnictl.config_utils import Experiments, Config
from ..tools.nnictl.nnictl_utils import update_experiment

_logger = logging.getLogger('nni.experiment')

@dataclass(init=False)
class NniManagerArgs:
    port: int
    experiment_id: int
    start_mode: str  # new or resume
    mode: str  # training service platform
    log_dir: str
    log_level: str
    readonly: bool = False
    foreground: bool = False
    url_prefix: Optional[str] = None
    dispatcher_pipe: Optional[str] = None

    def __init__(self, action, exp_id, config, port, debug, foreground, url_prefix):
        self.port = port
        self.experiment_id = exp_id
        self.foreground = foreground
        self.url_prefix = url_prefix
        self.log_dir = config.experiment_working_directory

        if isinstance(config.training_service, list):
            self.mode = 'hybrid'
        else:
            self.mode = config.training_service.platform

        self.log_level = config.log_level
        if debug and self.log_level not in ['debug', 'trace']:
            self.log_level = 'debug'

        if action == 'resume':
            self.start_mode = 'resume'
        elif action == 'view':
            self.start_mode = 'resume'
            self.readonly = True
        else:
            self.start_mode = 'new'

    def to_command_line_args(self):
        ret = []
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                ret.append('--' + field.name)
                if isinstance(value, bool):
                    ret.append(str(value).lower())
                else:
                    ret.append(str(value))
        return ret

def start_experiment(action, exp_id, config, port, debug, run_mode, url_prefix):
    foreground = run_mode.value == 'foreground'
    nni_manager_args = NniManagerArgs(action, exp_id, config, port, debug, foreground, url_prefix)

    _ensure_port_idle(port)
    websocket_platforms = ['hybrid', 'remote', 'openpai', 'kubeflow', 'frameworkcontroller', 'adl']
    if action != 'view' and nni_manager_args.mode in websocket_platforms:
        _ensure_port_idle(port + 1, f'{nni_manager_args.mode} requires an additional port')

    proc = None
    try:
        _logger.info(
            'Creating experiment, Experiment ID: %s', colorama.Fore.CYAN + exp_id + colorama.Style.RESET_ALL
        )
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
            logDir=config.experiment_working_directory,
            tag=[],
        )

        _logger.info('Setting up...')
        rest.post(port, '/experiment', config.json(), url_prefix)

        return proc

    except Exception as e:
        _logger.error('Create experiment failed')
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.kill()
        raise e

def _start_rest_server(nni_manager_args, run_mode) -> Tuple[int, Popen]:
    node_dir = Path(nni_node.__path__[0])
    node = str(node_dir / ('node.exe' if sys.platform == 'win32' else 'node'))
    main_js = str(node_dir / 'main.js')
    cmd = [node, '--max-old-space-size=4096', main_js]
    cmd += nni_manager_args.to_command_line_args()

    if run_mode.value == 'detach':
        log = Path(nni_manager_args.log_dir, nni_manager_args.experiment_id, 'log')
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
        return Popen(cmd, stdout=out, stderr=err, cwd=node_dir, preexec_fn=os.setpgrp)


def start_experiment_retiarii(exp_id: str, config: ExperimentConfig, port: int, debug: bool) -> Popen:
    pipe = None
    proc = None

    config.validate(initialized_tuner=True)
    _ensure_port_idle(port)
    if isinstance(config.training_service, list): # hybrid training service
        _ensure_port_idle(port + 1, 'Hybrid training service requires an additional port')
    elif config.training_service.platform in ['remote', 'openpai', 'kubeflow', 'frameworkcontroller', 'adl']:
        _ensure_port_idle(port + 1, f'{config.training_service.platform} requires an additional port')

    try:
        _logger.info('Creating experiment, Experiment ID: %s', colorama.Fore.CYAN + exp_id + colorama.Style.RESET_ALL)
        pipe = Pipe(exp_id)
        start_time, proc = _start_rest_server_retiarii(config, port, debug, exp_id, pipe.path)
        _logger.info('Connecting IPC pipe...')
        pipe_file = pipe.connect()
        nni.runtime.protocol._in_file = pipe_file
        nni.runtime.protocol._out_file = pipe_file
        _logger.info('Starting web server...')
        _check_rest_server(port)
        platform = 'hybrid' if isinstance(config.training_service, list) else config.training_service.platform
        _save_experiment_information(exp_id, port, start_time, platform,
                                     config.experiment_name, proc.pid, config.experiment_working_directory, ['retiarii'])
        _logger.info('Setting up...')
        rest.post(port, '/experiment', config.json())
        return proc, pipe

    except Exception as e:
        _logger.error('Create experiment failed')
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.kill()
        if pipe is not None:
            with contextlib.suppress(Exception):
                pipe.close()
        raise e

def _ensure_port_idle(port: int, message: Optional[str] = None) -> None:
    sock = socket.socket()
    if sock.connect_ex(('localhost', port)) == 0:
        sock.close()
        message = f'(message)' if message else ''
        raise RuntimeError(f'Port {port} is not idle {message}')


def _start_rest_server_retiarii(config: ExperimentConfig, port: int, debug: bool, experiment_id: str,
                                pipe_path: str = None, mode: str = 'new') -> Tuple[int, Popen]:
    if isinstance(config.training_service, list):
        ts = 'hybrid'
    else:
        ts = config.training_service.platform
        if ts == 'openpai':
            ts = 'pai'

    args = {
        'port': port,
        'mode': ts,
        'experiment_id': experiment_id,
        'start_mode': mode,
        'log_dir': config.experiment_working_directory,
        'log_level': 'debug' if debug else 'info'
    }
    if pipe_path is not None:
        args['dispatcher_pipe'] = pipe_path

    if mode == 'view':
        args['start_mode'] = 'resume'
        args['readonly'] = 'true'

    node_dir = Path(nni_node.__path__[0])
    node = str(node_dir / ('node.exe' if sys.platform == 'win32' else 'node'))
    main_js = str(node_dir / 'main.js')
    cmd = [node, '--max-old-space-size=4096', main_js]
    for arg_key, arg_value in args.items():
        cmd.append('--' + arg_key)
        cmd.append(str(arg_value))

    if sys.platform == 'win32':
        from subprocess import CREATE_NEW_PROCESS_GROUP
        proc = Popen(cmd, cwd=node_dir, creationflags=CREATE_NEW_PROCESS_GROUP)
    else:
        if pipe_path is None:
            import os
            proc = Popen(cmd, cwd=node_dir, preexec_fn=os.setpgrp)
        else:
            proc = Popen(cmd, cwd=node_dir)
    return int(time.time() * 1000), proc


def _check_rest_server(port: int, retry: int = 3, url_prefix: Optional[str] = None) -> None:
    for i in range(retry):
        with contextlib.suppress(Exception):
            rest.get(port, '/check-status', url_prefix)
            return
        if i > 0:
            _logger.warning('Timeout, retry...')
        time.sleep(1)
    rest.get(port, '/check-status', url_prefix)


def _save_experiment_information(experiment_id: str, port: int, start_time: int, platform: str,
                                 name: str, pid: int, logDir: str, tag: List[Any]) -> None:
    experiments_config = Experiments()
    experiments_config.add_experiment(experiment_id, port, start_time, platform, name, pid=pid, logDir=logDir, tag=tag)


def get_stopped_experiment_config(exp_id, exp_dir=None):
    config_json = get_stopped_experiment_config_json(exp_id, exp_dir)
    config = ExperimentConfig(**config_json)
    if exp_dir and not os.path.samefile(exp_dir, config.experiment_working_directory):
        msg = 'Experiment working directory provided in command line (%s) is different from experiment config (%s)'
        _logger.warning(msg, exp_dir, config.experiment_working_directory)
        config.experiment_working_directory = exp_dir
    return config

def get_stopped_experiment_config_json(exp_id, exp_dir=None):
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
            _logger.error('Only stopped experiments can be resumed or viewed!')
            return None
        return Config(exp_id, experiment_metadata['logDir']).get_config()
