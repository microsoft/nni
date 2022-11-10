# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from io import BytesIO
import json
from unittest import TestCase, main

from nni.assessor import Assessor, AssessResult
from nni.runtime import msg_dispatcher_base as msg_dispatcher_base
from nni.runtime.msg_dispatcher import MsgDispatcher
from nni.runtime.tuner_command_channel.legacy import *

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
    _set_out_file(_in_buf)
    _set_in_file(_out_buf)


def _restore_io():
    _in_buf.seek(0)
    _out_buf.seek(0)
    _set_in_file(_in_buf)
    _set_out_file(_out_buf)


class AssessorTestCase(TestCase):
    def test_assessor(self):
        pass
        _reverse_io()
        send(CommandType.ReportMetricData, '{"parameter_id": 0,"trial_job_id":"A","type":"PERIODICAL","sequence":0,"value":"2"}')
        send(CommandType.ReportMetricData, '{"parameter_id": 1,"trial_job_id":"B","type":"PERIODICAL","sequence":0,"value":"2"}')
        send(CommandType.ReportMetricData, '{"parameter_id": 0,"trial_job_id":"A","type":"PERIODICAL","sequence":1,"value":"3"}')
        send(CommandType.TrialEnd, '{"trial_job_id":"A","event":"SYS_CANCELED","hyper_params":"{\\"parameter_id\\": 0}"}')
        send(CommandType.TrialEnd, '{"trial_job_id":"B","event":"SUCCEEDED","hyper_params":"{\\"parameter_id\\": 1}"}')
        send(CommandType.NewTrialJob, 'null')
        _restore_io()

        assessor = NaiveAssessor()
        dispatcher = MsgDispatcher('ws://_unittest_placeholder_', None, assessor)
        dispatcher._channel = LegacyCommandChannel()
        msg_dispatcher_base._worker_fast_exit_on_terminate = False

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
