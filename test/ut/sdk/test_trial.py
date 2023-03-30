# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import nni
import nni.trial
from nni.runtime.trial_command_channel import get_default_trial_command_channel, set_default_trial_command_channel

import numpy as np
from unittest import TestCase, main

from .helper.trial_command_channel import TestHelperTrialCommandChannel


class TrialTestCase(TestCase):
    def setUp(self):
        self._default_channel = get_default_trial_command_channel()
        self.channel = TestHelperTrialCommandChannel()
        set_default_trial_command_channel(self.channel)
        nni.trial.overwrite_intermediate_seq(0)

        self._trial_params = { 'msg': 'hi', 'x': 123, 'dict': { 'key': 'value', 'y': None } }
        self.channel.init_params({
            'parameter_id': 'test_param',
            'parameters': self._trial_params
        })

    def tearDown(self):
        set_default_trial_command_channel(self._default_channel)

    def test_get_next_parameter(self):
        self.assertEqual(nni.get_next_parameter(), self._trial_params)

    def test_get_current_parameter(self):
        nni.get_next_parameter()
        self.assertEqual(nni.get_current_parameter('x'), 123)

    def test_get_experiment_id(self):
        self.assertEqual(nni.get_experiment_id(), 'test_experiment')

    def test_get_trial_id(self):
        self.assertEqual(nni.get_trial_id(), 'test_trial_job_id')
    
    def test_get_sequence_id(self):
        self.assertEqual(nni.get_sequence_id(), 0)

    def test_report_intermediate_result(self):
        nni.report_intermediate_result(123)
        self.assertEqual(self.channel.get_last_metric(), {
            'parameter_id': 'test_param',
            'trial_job_id': 'test_trial_job_id',
            'type': 'PERIODICAL',
            'sequence': 0,
            'value': 123
        })

    def test_report_final_result_simple(self):
        self._test_report_final_result(123, 123)

    def test_report_final_result_object(self):
        obj = ['obj1', {'key1': 'v1', 'k2': None}, 233, 0.456]
        self._test_report_final_result(obj, obj)

    def test_report_final_result_numpy(self):
        self._test_report_final_result(np.float32(0.25), 0.25)

    def test_report_final_result_nparray(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        nni.report_final_result(arr)
        out = self.channel.get_last_metric()
        self.assertEqual(len(arr), 2)
        self.assertEqual(len(arr[0]), 3)
        self.assertEqual(len(arr[1]), 3)
        self.assertEqual(arr[0][0], 1)
        self.assertEqual(arr[0][1], 2)
        self.assertEqual(arr[0][2], 3)
        self.assertEqual(arr[1][0], 4)
        self.assertEqual(arr[1][1], 5)
        self.assertEqual(arr[1][2], 6)

    def _test_report_final_result(self, in_, out):
        nni.report_final_result(in_)
        self.assertEqual(self.channel.get_last_metric(), {
            'parameter_id': 'test_param',
            'trial_job_id': 'test_trial_job_id',
            'type': 'FINAL',
            'sequence': 0,
            'value': out
        })


if __name__ == '__main__':
    main()
