# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import atexit
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
from subprocess import Popen, PIPE
import sys
import time
import threading

import pytest

from nni.runtime.tuner_command_channel.websocket import WebSocket

# A helper server that connects its stdio to incoming WebSocket.
_server = None
_client = None

_command1 = 'T_hello world'
_command2 = 'T_你好'

## test cases ##

#@pytest.mark.skipif(sys.platform == 'win32', reason='debug')
def test_connect():
    global _client
    port = _init()
    _client = WebSocket(f'ws://localhost:{port}')
    _client.connect()

#@pytest.mark.skipif(sys.platform == 'win32', reason='debug')
def test_send():
    # Send commands to server via channel, and get them back via server's stdout.
    _client.send(_command1)
    _client.send(_command2)
    time.sleep(0.01)

    sent1 = _server.stdout.readline().strip()
    assert sent1 == _command1, sent1

    sent2 = _server.stdout.readline().strip()
    assert sent2 == _command2, sent2

#@pytest.mark.skipif(sys.platform == 'win32', reason='debug')
def test_receive():
    # Send commands to server via stdin, and get them back via channel.
    _server.stdin.write(_command1 + '\n')
    _server.stdin.write(_command2 + '\n')
    _server.stdin.flush()

    received1 = _client.receive()
    assert received1 == _command1, received1

    received2 = _client.receive()
    assert received2 == _command2, received2

#@pytest.mark.skipif(sys.platform == 'win32', reason='debug')
def test_disconnect():
    _client.disconnect()

    # release the port
    global _server
    _server.stdin.write('_close_\n')
    _server.stdin.flush()
    _server.terminate()
    #_server = None

def test_debug():
    time.sleep(10)
    threads = '|'.join([t.name for t in threading.enumerate()])
    threads = '@@@ ' + threads + ' @@@\n'
    code = _server.poll()
    threads += f'@@@ {code} @@@\n'
    sys.stderr.write(threads)
    sys.stderr.flush()
    with open('tmp_threads.txt', 'w') as f:
        f.write(threads)
    exit()

## helper ##

def _init():
    global _server

    # launch a server that connects websocket to stdio
    script = (Path(__file__).parent / 'helper/websocket_server.py').resolve()
    _server = Popen([sys.executable, str(script)], stdin=PIPE, stdout=PIPE, encoding='utf_8')
    time.sleep(0.1)

    # if a test fails, make sure to stop the server
    atexit.register(lambda: _server is None or _server.terminate())

    return int(_server.stdout.readline().strip())

if __name__ == '__main__':
    test_connect()
    test_send()
    test_receive()
    test_disconnect()
    print('pass')
