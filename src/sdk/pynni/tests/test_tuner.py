# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================


import nni.protocol
from nni.protocol import CommandType, send, receive
from nni.tuner import Tuner
from nni.msg_dispatcher import MsgDispatcher
from nni.utils import extract_scalar_reward
from io import BytesIO
import json
from unittest import TestCase, main


class NaiveTuner(Tuner):
    def __init__(self):
        self.param = 0
        self.trial_results = [ ]
        self.search_space = None

    def generate_parameters(self, parameter_id):
        # report Tuner's internal states to generated parameters,
        # so we don't need to pause the main loop
        self.param += 2
        return {
            'param': self.param,
            'trial_results': self.trial_results,
            'search_space': self.search_space
        }

    def receive_trial_result(self, parameter_id, parameters, value):
        reward = extract_scalar_reward(value)
        self.trial_results.append((parameter_id, parameters['param'], reward, False))

    def receive_customized_trial_result(self, parameter_id, parameters, value):
        reward = extract_scalar_reward(value)
        self.trial_results.append((parameter_id, parameters['param'], reward, True))

    def update_search_space(self, search_space):
        self.search_space = search_space


_in_buf = BytesIO()
_out_buf = BytesIO()

def _reverse_io():
    _in_buf.seek(0)
    _out_buf.seek(0)
    nni.protocol._out_file = _in_buf
    nni.protocol._in_file = _out_buf

def _restore_io():
    _in_buf.seek(0)
    _out_buf.seek(0)
    nni.protocol._in_file = _in_buf
    nni.protocol._out_file = _out_buf



class TunerTestCase(TestCase):
    def test_tuner(self):
        _reverse_io()  # now we are sending to Tuner's incoming stream
        send(CommandType.RequestTrialJobs, '2')
        send(CommandType.ReportMetricData, '{"parameter_id":0,"type":"PERIODICAL","value":10}')
        send(CommandType.ReportMetricData, '{"parameter_id":1,"type":"FINAL","value":11}')
        send(CommandType.UpdateSearchSpace, '{"name":"SS0"}')
        send(CommandType.AddCustomizedTrialJob, '{"param":-1}')
        send(CommandType.ReportMetricData, '{"parameter_id":2,"type":"FINAL","value":22}')
        send(CommandType.RequestTrialJobs, '1')
        send(CommandType.KillTrialJob, 'null')
        _restore_io()

        tuner = NaiveTuner()
        dispatcher = MsgDispatcher(tuner)
        nni.msg_dispatcher_base._worker_fast_exit_on_terminate = False

        dispatcher.run()
        e = dispatcher.worker_exceptions[0]
        self.assertIs(type(e), AssertionError)
        self.assertEqual(e.args[0], 'Unsupported command: CommandType.KillTrialJob')

        _reverse_io()  # now we are receiving from Tuner's outgoing stream
        self._assert_params(0, 2, [ ], None)
        self._assert_params(1, 4, [ ], None)

        command, data = receive()  # this one is customized
        data = json.loads(data)
        self.assertIs(command, CommandType.NewTrialJob)
        self.assertEqual(data, {
            'parameter_id': 2,
            'parameter_source': 'customized',
            'parameters': { 'param': -1 }
        })

        self._assert_params(3, 6, [[1,4,11,False], [2,-1,22,True]], {'name':'SS0'})

        self.assertEqual(len(_out_buf.read()), 0)  # no more commands


    def _assert_params(self, parameter_id, param, trial_results, search_space):
        command, data = receive()
        self.assertIs(command, CommandType.NewTrialJob)
        data = json.loads(data)
        self.assertEqual(data['parameter_id'], parameter_id)
        self.assertEqual(data['parameter_source'], 'algorithm')
        self.assertEqual(data['parameters']['param'], param)
        self.assertEqual(data['parameters']['trial_results'], trial_results)
        self.assertEqual(data['parameters']['search_space'], search_space)


if __name__ == '__main__':
    main()
