# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import unittest

from nni.algorithms.hpo.curvefitting_assessor import CurvefittingAssessor
from nni.algorithms.hpo.curvefitting_assessor.model_factory import CurveModel
from nni.assessor import AssessResult

class TestCurveFittingAssessor(unittest.TestCase):
    def test_init(self):
        new_assessor = CurvefittingAssessor(20)
        self.assertEqual(new_assessor.start_step, 6)
        self.assertEqual(new_assessor.target_pos, 20)

    def test_insufficient_point(self):
        new_assessor = CurvefittingAssessor(20)
        ret = new_assessor.assess_trial(1, [1])
        self.assertEqual(ret, AssessResult.Good)

    def test_not_converged(self):
        new_assessor = CurvefittingAssessor(20)
        with self.assertRaises(TypeError):
            ret = new_assessor.assess_trial([1, 199, 0, 199, 1, 209, 2])
        ret = new_assessor.assess_trial(1, [1, 199, 0, 199, 1, 209, 2])
        self.assertEqual(ret, AssessResult.Good)
        models = CurveModel(21)
        self.assertEqual(models.predict([1, 199, 0, 199, 1, 209, 2]), None)

    def test_curve_model(self):
        test_model = CurveModel(21)
        test_model.effective_model = ['vap', 'pow3', 'linear', 'logx_linear', 'dr_hill_zero_background', 'log_power', 'pow4', 'mmf', 'exp4', 'ilog2', 'weibull', 'janoschek']
        test_model.effective_model_num = 12
        test_model.point_num = 9
        test_model.target_pos = 20
        test_model.trial_history = ([1, 1, 1, 1, 1, 1, 1, 1, 1])
        test_model.weight_samples = np.ones((test_model.effective_model_num), dtype=float) / test_model.effective_model_num
        self.assertAlmostEqual(test_model.predict_y('vap', 9), 0.5591906328335763)
        self.assertAlmostEqual(test_model.predict_y('logx_linear', 15), 1.0704360293379522)
        self.assertAlmostEqual(test_model.f_comb(9, test_model.weight_samples), 1.1543379521172443)
        self.assertAlmostEqual(test_model.f_comb(15, test_model.weight_samples), 1.6949395581692737)

if __name__ == '__main__':
    unittest.main()
