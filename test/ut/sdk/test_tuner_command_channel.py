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

from nni.runtime import tuner_command_channel

# A helper server that connects its stdio to incoming WebSocket.
_server = None

_command1 = 'T_hello world'
_command2 = 'T_你好'

## test cases ##

def test_send():
    # Send commands to server via channel, and get them back via server's stdout.
    tuner_command_channel.send(_command1)
    tuner_command_channel.send(_command2)
    time.sleep(0.01)

    sent1 = _server.stdout.readline().strip()
    assert sent1 == _command1, sent1

    sent2 = _server.stdout.readline().strip()
    assert sent2 == _command2, sent2

def test_receive():
    # Send commands to server via stdin, and get them back via channel.
    _server.stdin.write(_command1 + '\n')
    _server.stdin.write(_command2 + '\n')
    _server.stdin.flush()

    received1 = tuner_command_channel.receive()
    assert received1 == _command1, received1

    received2 = tuner_command_channel.receive()
    assert received2 == _command2, received2

def test_shutdown():
    # If python process exited normally later, then this test succeeds.
    # Because a running event loop will prevent exiting.
    tuner_command_channel.disconnect()

    # release the port
    global _server
    _server.terminate()
    _server = None

## helpers ##

def _init():
    global _server
    # launch a server that connects websocket to stdio
    script = (Path(__file__).parent / 'helper/web_socket_server.py').resolve()
    _server = Popen([sys.executable, str(script)], stdin=PIPE, stdout=PIPE, encoding='utf8')
    time.sleep(0.1)

    # if a test fails, make sure to stop the server
    atexit.register(lambda: _server is None or _server.terminate())

    # this is not public API, but we have no better choice util refactoring __main__.py
    port = int(_server.stdout.readline().strip())
    tuner_command_channel.connect(f'ws://localhost:{port}')

_init()

if __name__ == '__main__':
    test_send()
    test_receive()
    test_shutdown()
