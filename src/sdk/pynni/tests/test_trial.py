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

import nni
import nni.platform.test as test_platform
import nni.trial

import numpy as np
from unittest import TestCase, main


class TrialTestCase(TestCase):
    def setUp(self):
        self._trial_params = { 'msg': 'hi', 'x': 123, 'dict': { 'key': 'value', 'y': None } }
        test_platform._params = { 'parameter_id': 'test_param', 'parameters': self._trial_params }

    def test_get_next_parameter(self):
        self.assertEqual(nni.get_next_parameter(), self._trial_params)

    def test_get_current_parameter(self):
        nni.get_next_parameter()
        self.assertEqual(nni.get_current_parameter('x'), 123)

    def test_get_sequence_id(self):
        self.assertEqual(nni.get_sequence_id(), 0)

    def test_report_intermediate_result(self):
        nni.report_intermediate_result(123)
        self.assertEqual(test_platform.get_last_metric(), {
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
        out = test_platform.get_last_metric()
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
        self.assertEqual(test_platform.get_last_metric(), {
            'parameter_id': 'test_param',
            'trial_job_id': 'test_trial_job_id',
            'type': 'FINAL',
            'sequence': 0,
            'value': out
        })


if __name__ == '__main__':
    main()