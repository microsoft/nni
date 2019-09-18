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


from io import BytesIO
from unittest import TestCase, main

import nni.protocol
from nni.assessor import Assessor, AssessResult
from nni.msg_dispatcher import MsgDispatcher
from nni.protocol import CommandType, send, receive

_trials = []
_end_trials = []


class NaiveAssessor(Assessor):
    def assess_trial(self, trial_job_id, trial_history):
        _trials.append(trial_job_id)
        if sum(trial_history) % 2 == 0:
            return AssessResult.Good
        else:
            return AssessResult.Bad

    def trial_end(self, trial_job_id, success):
        _end_trials.append((trial_job_id, success))


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


class AdvisorTestCase(TestCase):

    def test_hyperband_advisor(self):
        _reverse_io()
        send(CommandType.RequestTrialJobs, '2')
        send(CommandType.ReportMetricData, '{"parameter_id":0,"type":"PERIODICAL","value":10}')
        send(CommandType.ReportMetricData, '{"parameter_id":1,"type":"FINAL","value":11}')
        send(CommandType.UpdateSearchSpace, '{"name":"SS0"}')
        send(CommandType.AddCustomizedTrialJob, '{"param":-1}')
        send(CommandType.ReportMetricData, '{"parameter_id":2,"type":"FINAL","value":22}')
        send(CommandType.RequestTrialJobs, '1')
        send(CommandType.KillTrialJob, 'null')
        _restore_io()

        assessor = NaiveAssessor()
        dispatcher = MsgDispatcher(None, assessor)
        nni.msg_dispatcher_base._worker_fast_exit_on_terminate = False

        dispatcher.run()
        e = dispatcher.worker_exceptions[0]
        self.assertIs(type(e), AssertionError)
        self.assertEqual(e.args[0], 'Unsupported command: CommandType.NewTrialJob')

        self.assertEqual(_trials, ['A', 'B', 'A'])
        self.assertEqual(_end_trials, [('A', False), ('B', True)])

        _reverse_io()
        command, data = receive()
        self.assertIs(command, CommandType.KillTrialJob)
        self.assertEqual(data, '"A"')
        self.assertEqual(len(_out_buf.read()), 0)


if __name__ == '__main__':
    main()
