import contextlib
from pathlib import Path
import socket
from subprocess import Popen
import sys
import time
from typing import Optional, Tuple

import nni.runtime.protocol
import nni_node

from .config import ExperimentConfig
from . import management
from .pipe import Pipe
from . import rest


def start_experiment(config: ExperimentConfig, port: int, debug: bool) -> Tuple[Popen, Pipe]:
    pipe = None
    proc = None

    config.validate()
    _ensure_port_idle(port)
    if config._training_service == 'pai':
        _ensure_port_idle(port + 1, 'OpenPAI requires an additional port')
    exp_id = management.generate_experiment_id()

    try:
        print(f'Creating experiment {exp_id}...')
        pipe = Pipe(exp_id)
        proc = _start_rest_server(config, port, debug, exp_id, pipe.path)
        pipe_file = pipe.connect()
        nni.runtime.protocol._in_file = pipe_file
        nni.runtime.protocol._out_file = pipe_file
        print('Statring web server...')
        _check_rest_server(port)
        print('Setting up...')
        _init_experiment(config, port, debug)  # todo: kill on fail
        return proc, pipe

    except Exception as e:
        print('Create experiment failed')
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
    args = {
        'port': port,
        'mode': config._training_service,
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


def _check_rest_server(port: int, retry: int = 10) -> None:
    for _ in range(retry):
        with contextlib.suppress(Exception):
            rest.get(port, '/check-status')
            return
        time.sleep(1)
    rest.get(port, '/check-status')


def _init_experiment(config: ExperimentConfig, port: int, debug: bool) -> None:
    rest.put(port, '/experiment/cluster-metadata', config.cluster_metadata_json())
    rest.post(port, '/experiment', config.experiment_config_json())
