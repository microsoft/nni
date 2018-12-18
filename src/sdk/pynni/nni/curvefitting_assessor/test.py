# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import unittest

from .curvefitting_assessor import CurvefittingAssessor
from nni.assessor import AssessResult

class TestCurveFittingAssessor(unittest.TestCase):
    def test_init(self):
        new_assessor = CurvefittingAssessor(20)
        self.assertEquals(new_assessor.start_step, 6)
        self.assertEquals(new_assessor.target_pos, 20)
        self.assertEquals(new_assessor.completed_best_performance, 0.0001)

    def test_insufficient_point(self):
        new_assessor = CurvefittingAssessor(20)
        ret = new_assessor.assess_trial(1, [1])
        self.assertEquals(ret, AssessResult.Good)

    def test_not_converged(self):
        new_assessor = CurvefittingAssessor(20)
        with self.assertRaises(TypeError):
            ret = new_assessor.assess_trial([1, 199, 0, 199, 1, 209, 2])
        ret = new_assessor.assess_trial(1, [1, 199, 0, 199, 1, 209, 2])
        self.assertEquals(ret, AssessResult.Good)
        models = CurveModel(21)
        self.assertEquals(models.predict([1, 199, 0, 199, 1, 209, 2]), -1)

    def test_curve_model(self):
        test_model = CurveModel(21)
        test_model.effective_model = ['vap', 'pow3', 'linear', 'logx_linear', 'dr_hill_zero_background', 'log_power', 'pow4', 'mmf', 'exp4', 'ilog2', 'weibull', 'janoschek']
        test_model.effective_model_num = 12
        test_model.point_num = 9
        test_model.target_pos = 20
        test_model.trial_history = ([1, 1, 1, 1, 1, 1, 1, 1, 1])
        test_model.weight_samples = np.ones((test_model.effective_model_num), dtype=np.float) / test_model.effective_model_num
        self.assertAlmostEquals(test_model.predict_y('vap', 9), 0.5591906328335763)
        self.assertAlmostEquals(test_model.predict_y('logx_linear', 15), 1.0704360293379522)
        self.assertAlmostEquals(test_model.f_comb(9, test_model.weight_samples), 1.1543379521172443)
        self.assertAlmostEquals(test_model.f_comb(15, test_model.weight_samples), 1.6949395581692737)

if __name__ == '__main__':
    unittest.main()
