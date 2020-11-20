from pathlib import Path
import random
import socket
from subprocess import Popen
import sys
import tempfile
from threading import Thread

import nni.runtime.protocol
import nni_node

from .config import ExperimentConfig


def _start_rest_server(config: ExperimentConfig, port: int) -> Popen:
    _check_port_idle(port)
    if config._training_service == 'pai':
        _check_port_idle(port + 1)

    uid = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    pipe_path = Path(tempfile.gettempdir(), 'nni-pipe-' + uid)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(pipe_path)
    sock.listen(1)

    # TODO: check if a new thread is really necessary
    Thread(target=lambda: _wait_connection(sock)).start()

    print('Starting NNI manager...')

    node_dir = Path(nni_node.__path__[0])
    node = node_dir / ('node.exe' if sys.platform == 'win32' else 'node')
    cmd = [
        str(node), '--max-old-space-size=4096',
        str(node_dir / 'main.js'),
        '--port', str(port), 
        '--mode', platform,
        '--start_mode', 'new',
        '--dispatcher-pipe', pipe_path
    ]

    # TODO: logging
    return Popen(cmd, cwd=node_dir)

def _wait_connection(sock):
    conn, addr = sock.accept()
    nni.runtime.protocol._in_file = conn
    nni.runtime.protocol._out_file = conn


def _init_experiment(proc: Popen, config: ExperimentConfig, port: int, debug: bool) -> None:
    _check_rest_server(config, port, debug)

    print('Initializing experiment...')

    config_json = config._to_json()
    config_json['debug'] = debug
    url = _url_template.format(port=port, api='/experiment')
    resp = requests.post(url, data=config_json, timeout=20)
    resp.raise_for_status()

    print('Experiment started')
    print('Experiment ID:', resp.json()['experimentId'])



_url_template = 'http://localhost:{port}/api/v1/nni{api}'

def _check_port_idle(port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('localhost', port))
        sock.close()
    except ConnectionRefusedError:
        raise RuntimeError(f'port {port} is not idle')


def _check_rest_server(port: int, retry_count: int = 20, timeout: int = 20) -> None:
    url = _url_template.format(port=port, api='/check-status')
    while True:
        try:
            requests.get(url, timeout=timeout).raise_for_status()
        except Exception:
            retry_count -= 1
            if retry_count <= 0:
                raise RuntimeError('NNI manager start failed')
        time.sleep(1)
