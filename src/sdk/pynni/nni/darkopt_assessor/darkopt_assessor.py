# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from nni.assessor import Assessor, AssessResult

from .darkopt2 import EnsembleSamplingPredictor

from enum import Enum
import logging

logger = logging.getLogger('nni.contribution.darkopt_assessor')


class OptimizeMode(Enum):
    Maximize = 'maximize'
    Minimize = 'minimize'


class DarkoptAssessor(Assessor):
    def __init__(self, best_score, period, threshold, optimize_mode):
        self.best_score = best_score
        self.period = period
        self.threshold = threshold
        self.optimize_mode = optimize_mode

        self.predictor = EnsembleSamplingPredictor()
        if self.optimize_mode is OptimizeMode.Minimize:
            self.best_score = -self.best_score


    def assess_trial(self, trial_job_id, history):
        '''
        assess_trial
        '''
        logger.debug('assess_trial %s' % history)
        if self.optimize_mode is OptimizeMode.Minimize:
            history = [ -x for x in history ]

        max_ = max(history)
        if max_ > self.best_score:
            self.best_score = max_
            return AssessResult.Good

        self.predictor.fit(list(range(len(history))), history)
        proba_worse = self.predictor.predict_proba_less_than(self.period, self.best_score)

        if proba_worse > self.threshold:
            return AssessResult.Bad
        else:
            return AssessResult.Good