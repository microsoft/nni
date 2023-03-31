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

from nni.runtime.command_channel.websocket import WsChannelClient

# A helper server that connects its stdio to incoming WebSocket.
_server = None
_client = None

_command1 = {'type': 'ut_command', 'value': 123}
_command2 = {'type': 'ut_command', 'value': '你好'}

## test cases ##

def test_connect():
    global _client
    port = _init()
    _client = WsChannelClient(f'ws://localhost:{port}')
    _client.connect()

def test_send():
    # Send commands to server via channel, and get them back via server's stdout.
    _client.send(_command1)
    _client.send(_command2)
    time.sleep(0.01)

    sent1 = json.loads(_server.stdout.readline())
    assert sent1 == _command1, sent1

    sent2 = json.loads(_server.stdout.readline().strip())
    assert sent2 == _command2, sent2

def test_receive():
    # Send commands to server via stdin, and get them back via channel.
    _server.stdin.write(json.dumps(_command1) + '\n')
    _server.stdin.write(json.dumps(_command2) + '\n')
    _server.stdin.flush()

    received1 = _client.receive()
    assert received1 == _command1, received1

    received2 = _client.receive()
    assert received2 == _command2, received2

def test_disconnect():
    _client.disconnect()

    # release the port
    global _server
    _server.stdin.write('_close_\n')
    _server.stdin.flush()
    time.sleep(0.1)
    _server.terminate()
    _server = None

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
