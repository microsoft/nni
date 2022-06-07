# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest import TestCase, main
from click import command

from numpy import reciprocal
from nni.common.serializer import load

from nni.runtime import msg_dispatcher_base
from nni.runtime.msg_dispatcher import MsgDispatcher
from nni.runtime.tuner_command_channel.legacy import *
from nni.tuner import Tuner
from nni.utils import extract_scalar_reward
import atexit
from pathlib import Path
from subprocess import Popen, PIPE
import sys
import time

_server = None

class NaiveTuner(Tuner):
    def __init__(self):
        self.param = 0
        self.trial_results = []
        self.search_space = None
        self._accept_customized_trials()

    def generate_parameters(self, parameter_id, **kwargs):
        # report Tuner's internal states to generated parameters,
        # so we don't need to pause the main loop
        self.param += 2
        return {
            'param': self.param,
            'trial_results': self.trial_results,
            'search_space': self.search_space
        }

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        reward = extract_scalar_reward(value)
        self.trial_results.append((parameter_id, parameters['param'], reward, kwargs.get("customized")))

    def update_search_space(self, search_space):
        self.search_space = search_space

_command1 = 'KIqQKkW'
_command2 = 'IN{"features":{"_type":"choice","_value":[128,256,512,1024]},"lr":{"_type":"loguniform","_value":[0.0001,0.1]},"momentum":{"_type":"uniform","_value":[0,1]}}'
_command3 = 'GE2'
_command4 = 'TR{"parameter_id": 0, "parameter_source": "algorithm", "parameters": {"features": 1024, "lr": 0.0006827054029247927, "momentum": 0.22591105939004463}, "parameter_index": 0}'
_command5 = 'EN{"trial_job_id":"H6KgS","event":"FAILED","hyper_params":"{\"parameter_id\": 0, \"parameter_source\": \"algorithm\", \"parameters\": {\"features\": 1024, \"lr\": 0.0006827054029247927, \"momentum\": 0.22591105939004463}, \"parameter_index\": 0}"}'
_command6 = 'TE'
class MsgDispatcherTestCase(TestCase):
    def test_msg_dispatcher(self):
        tuner = NaiveTuner()
        port = _init()
        dispatcher = MsgDispatcher(f'ws://localhost:{port}', tuner)
        msg_dispatcher_base._worker_fast_exit_on_terminate = False
        dispatcher._channel.connect()
        _server.stdin.write(_command1 + '\n')
        _server.stdin.flush()
        received1 = dispatcher._channel.receive()
        _server.stdin.write(_command2 + '\n')
        _server.stdin.flush()
        received2 = dispatcher._channel.receive()
        _server.stdin.write(_command3 + '\n')
        _server.stdin.flush()
        received3 = dispatcher._channel.receive()
        _server.stdin.write(_command4 + '\n')
        _server.stdin.flush()
        received4 = dispatcher._channel.receive()
        _server.stdin.write(_command5 + '\n')
        _server.stdin.flush()
        received5 = dispatcher._channel.receive()
        _server.stdin.write(_command6 + '\n')
        _server.stdin.flush()
        received6 = dispatcher._channel.receive()

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
    main()
