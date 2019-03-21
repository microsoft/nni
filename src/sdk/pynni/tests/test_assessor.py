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
from nni.assessor import Assessor, AssessResult
from nni.msg_dispatcher import MsgDispatcher

from io import BytesIO
import json
from unittest import TestCase, main


_trials = [ ]
_end_trials = [ ]

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


class AssessorTestCase(TestCase):
    def test_assessor(self):
        _reverse_io()
        send(CommandType.ReportMetricData, '{"trial_job_id":"A","type":"PERIODICAL","sequence":0,"value":2}')
        send(CommandType.ReportMetricData, '{"trial_job_id":"B","type":"PERIODICAL","sequence":0,"value":2}')
        send(CommandType.ReportMetricData, '{"trial_job_id":"A","type":"PERIODICAL","sequence":1,"value":3}')
        send(CommandType.TrialEnd, '{"trial_job_id":"A","event":"SYS_CANCELED"}')
        send(CommandType.TrialEnd, '{"trial_job_id":"B","event":"SUCCEEDED"}')
        send(CommandType.NewTrialJob, 'null')
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
