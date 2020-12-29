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
from . import management
from .pipe import Pipe
from . import rest

_logger = logging.getLogger('nni.experiment')


def start_experiment(config: ExperimentConfig, port: int, debug: bool) -> Tuple[Popen, Pipe]:
    pipe = None
    proc = None

    config.validate(initialized_tuner=True)
    _ensure_port_idle(port)
    if config.training_service.platform == 'openpai':
        _ensure_port_idle(port + 1, 'OpenPAI requires an additional port')
    exp_id = management.generate_experiment_id()

    try:
        _logger.info('Creating experiment %s%s', colorama.Fore.CYAN, exp_id)
        pipe = Pipe(exp_id)
        proc = _start_rest_server(config, port, debug, exp_id, pipe.path)
        _logger.info('Connecting IPC pipe...')
        pipe_file = pipe.connect()
        nni.runtime.protocol._in_file = pipe_file
        nni.runtime.protocol._out_file = pipe_file
        _logger.info('Statring web server...')
        _check_rest_server(port)
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


def _start_rest_server(config: ExperimentConfig, port: int, debug: bool, experiment_id: str, pipe_path: str) -> Popen:
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
    return Popen(cmd, cwd=node_dir)


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
    for cluster_metadata in convert.to_cluster_metadata(config):
        rest.put(port, '/experiment/cluster-metadata', cluster_metadata)
    rest.post(port, '/experiment', convert.to_rest_json(config))
