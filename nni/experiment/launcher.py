# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import contextlib
import logging
from pathlib import Path
import socket
from subprocess import Popen
import sys
import time
from typing import Optional, Tuple

import colorama

import nni_node  # pylint: disable=import-error
import nni.runtime.protocol

from .config import ExperimentConfig
from .pipe import Pipe
from . import rest
from ..tools.nnictl.config_utils import Experiments, Config
from ..tools.nnictl.nnictl_utils import update_experiment

_logger = logging.getLogger('nni.experiment')


def start_experiment(exp_id: str, config: ExperimentConfig, port: int, debug: bool, mode: str = 'new') -> Popen:
    proc = None

    config.validate(initialized_tuner=False)
    _ensure_port_idle(port)

    if mode != 'view':
        if isinstance(config.training_service, list): # hybrid training service
            _ensure_port_idle(port + 1, 'Hybrid training service requires an additional port')
        elif config.training_service.platform in ['remote', 'openpai', 'kubeflow', 'frameworkcontroller', 'adl']:
            _ensure_port_idle(port + 1, f'{config.training_service.platform} requires an additional port')

    try:
        _logger.info('Creating experiment, Experiment ID: %s', colorama.Fore.CYAN + exp_id + colorama.Style.RESET_ALL)
        start_time, proc = _start_rest_server(config, port, debug, exp_id, mode=mode)
        _logger.info('Starting web server...')
        _check_rest_server(port)
        platform = 'hybrid' if isinstance(config.training_service, list) else config.training_service.platform
        _save_experiment_information(exp_id, port, start_time, platform,
                                     config.experiment_name, proc.pid, str(config.experiment_working_directory))
        _logger.info('Setting up...')
        rest.post(port, '/experiment', config.json())
        return proc

    except Exception as e:
        _logger.error('Create experiment failed')
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.kill()
        raise e

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
        start_time, proc = _start_rest_server(config, port, debug, exp_id, pipe.path)
        _logger.info('Connecting IPC pipe...')
        pipe_file = pipe.connect()
        nni.runtime.protocol._in_file = pipe_file
        nni.runtime.protocol._out_file = pipe_file
        _logger.info('Starting web server...')
        _check_rest_server(port)
        platform = 'hybrid' if isinstance(config.training_service, list) else config.training_service.platform
        _save_experiment_information(exp_id, port, start_time, platform,
                                     config.experiment_name, proc.pid, config.experiment_working_directory)
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


def _start_rest_server(config: ExperimentConfig, port: int, debug: bool, experiment_id: str, pipe_path: str = None,
                       mode: str = 'new') -> Tuple[int, Popen]:
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


def _check_rest_server(port: int, retry: int = 3) -> None:
    for i in range(retry):
        with contextlib.suppress(Exception):
            rest.get(port, '/check-status')
            return
        if i > 0:
            _logger.warning('Timeout, retry...')
        time.sleep(1)
    rest.get(port, '/check-status')


def _save_experiment_information(experiment_id: str, port: int, start_time: int, platform: str, name: str, pid: int, logDir: str) -> None:
    experiments_config = Experiments()
    experiments_config.add_experiment(experiment_id, port, start_time, platform, name, pid=pid, logDir=logDir)


def get_stopped_experiment_config(exp_id: str, mode: str) -> None:
    update_experiment()
    experiments_config = Experiments()
    experiments_dict = experiments_config.get_all_experiments()
    experiment_metadata = experiments_dict.get(exp_id)
    if experiment_metadata is None:
        _logger.error('Id %s not exist!', exp_id)
        return
    if experiment_metadata['status'] != 'STOPPED':
        _logger.error('Only stopped experiments can be %sed!', mode)
        return
    experiment_config = Config(exp_id, experiment_metadata['logDir']).get_config()
    config = ExperimentConfig(**experiment_config)
    return config
