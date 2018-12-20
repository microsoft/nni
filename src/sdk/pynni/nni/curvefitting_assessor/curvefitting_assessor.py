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

import logging
from nni.assessor import Assessor, AssessResult
from .model_factory import CurveModel

logger = logging.getLogger('curvefitting_Assessor')

class CurvefittingAssessor(Assessor):
    '''
    CurvefittingAssessor uses learning curve fitting algorithm to predict the learning curve performance in the future.
    It stops a pending trial X at step S if the trial's forecast result at target step is convergence and lower than the
    best performance in the history.
    '''
    def __init__(self, epoch_num=20, optimize_mode='maximize', start_step=6, threshold=0.95):
        if start_step <= 0:
            logger.warning('It\'s recommended to set start_step to a positive number')
        # Record the target position we predict
        self.target_pos = epoch_num
        # Record the optimize_mode
        if optimize_mode == 'maximize':
            self.higher_better = True
        elif optimize_mode == 'minimize':
            self.higher_better = False
        else:
            self.higher_better = True
            logger.warning('unrecognized optimize_mode', optimize_mode)
        # Start forecasting when historical data reaches start step
        self.start_step = start_step
        # Record the compared threshold
        self.threshold = threshold
        # Record the best performance
        self.set_best_performance = False
        self.completed_best_performance = None
        self.trial_history = []
        logger.info('Successfully initials the curvefitting assessor')

    def trial_end(self, trial_job_id, success):
        '''
        trial end: update the best performance of completed trial job
        '''
        if success:
            if self.set_best_performance:
                self.completed_best_performance = max(self.completed_best_performance, self.trial_history[-1])
            else:
                self.set_best_performance = True
                self.completed_best_performance = self.trial_history[-1]
            logger.info('Updated complted best performance, trial job id:', trial_job_id)
        else:
            logger.info('No need to update, trial job id: ', trial_job_id)

    def assess_trial(self, trial_job_id, trial_history):
        '''
        assess whether a trial should be early stop by curve fitting algorithm
        return AssessResult.Good or AssessResult.Bad
        '''
        self.trial_history = trial_history
        curr_step = len(trial_history)
        if curr_step < self.start_step:
            return AssessResult.Good
        if not self.set_best_performance:
            return AssessResult.Good

        try:
            curvemodel = CurveModel(self.target_pos)
            predict_y = curvemodel.predict(trial_history)
            logger.info('Prediction done. Trial job id = ', trial_job_id, '. Predict value = ', predict_y)
            if predict_y is None:
                logger.info('wait for more information to predict precisely')
                return AssessResult.Good
            standard_performance = self.completed_best_performance * self.threshold
            if self.higher_better:
                if predict_y > standard_performance:
                    return AssessResult.Good
                return AssessResult.Bad
            else:
                if predict_y < standard_performance:
                    return AssessResult.Good
                return AssessResult.Bad

        except Exception as exception:
            logger.exception('unrecognize exception in curvefitting_asserssor', exception)
