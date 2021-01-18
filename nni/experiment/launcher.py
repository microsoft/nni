import contextlib
import logging
from pathlib import Path
import socket
from subprocess import Popen
import sys
import time
from typing import Optional, Tuple

import colorama

import nni.runtime.protocol
import nni_node

from .config import ExperimentConfig
from .config import convert
from .pipe import Pipe
from . import rest
from ..tools.nnictl.config_utils import Experiments

_logger = logging.getLogger('nni.experiment')


def start_experiment(exp_id: str, config: ExperimentConfig, port: int, debug: bool) -> Tuple[Popen, Pipe]:
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
        _logger.info('Statring web server...')
        _check_rest_server(port)
        platform = 'hybrid' if isinstance(config.training_service, list) else config.training_service.platform
        _save_experiment_information(exp_id, port, start_time, platform,
                                     config.experiment_name, proc.pid, config.experiment_working_directory)
        _logger.info('Setting up...')
        _init_experiment(config, port, debug)
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


def _start_rest_server(config: ExperimentConfig, port: int, debug: bool, experiment_id: str, pipe_path: str) -> Tuple[int, Popen]:
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
        'start_mode': 'new',
        'log_level': 'debug' if debug else 'info',
        'dispatcher_pipe': pipe_path,
    }

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


def _init_experiment(config: ExperimentConfig, port: int, debug: bool) -> None:
    if config.training_service.platform == 'local':
        rest.post(port, '/experiment', config.json())
        return

    for cluster_metadata in convert.to_cluster_metadata(config):
        rest.put(port, '/experiment/cluster-metadata', cluster_metadata)
    rest.post(port, '/experiment', convert.to_rest_json(config))


def _save_experiment_information(experiment_id: str, port: int, start_time: int, platform: str, name: str, pid: int, logDir: str) -> None:
    experiment_config = Experiments()
    experiment_config.add_experiment(experiment_id, port, start_time, platform, name, pid=pid, logDir=logDir)
