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
import numpy as np
from nni.assessor import Assessor, AssessResult
from .modelfactory import CurveModel

logger = logging.getLogger('curvefitting_Assessor')
THRESHOLD = 0.95

class CurvefittingAssessor(Assessor):
    '''
    CurvefittingAssessor uses learning curve fitting algorithm to predict the learning curve performance in the future.
    It stops a pending trial X at step S if the trial's forecast result at target step is convergence and lower than the best performance in the history.
    '''
    def __init__(self, start_step=6, epoch_num=20, threshold=THRESHOLD):
        if not start_step > 0:
            logger.warning('start step should be a positive number')
        if not epoch_num > 0:
            logger.warning('number of epoch should be a positive number')
        # Record the target position we predict
        self.target_pos = epoch_num
        # Record the best performance so far
        self.completed_best_performance = 0.0001
        # Start forecasting when historical data reaches start step
        self.start_step = max(start_step, int(epoch_num / 4))
        THRESHOLD = threshold
        logger.info('Successfully initials the curvefitting assessor')

    def trial_end(self, trial_job_id, success):
        '''
        trial end: update the best performance of completed trial job
        '''
        if success:
            self.completed_best_performance = max(self.completed_best_performance, np.max(self.trial_history))
            logger.info('Successully update complted best performance, trial job id:', trial_job_id)
        else:
            logger.info('No need to update, trial job id: ', trial_job_id)

    def assess_trial(self, trial_job_id, trial_history):
        '''
        assess whether a trial should be early stop by curve fitting
        return AssessResult.Good or AssessResult.Bad
        '''
        self.trial_history = trial_history
        curr_step = len(trial_history)
        if curr_step < self.start_step:
            return AssessResult.Good
        
        try:
            curvemodel = CurveModel(self.target_pos)
            predict_y = curvemodel.predict(trial_history)
            print ("predict y ", predict_y)
            if predict_y == -1:
                '''wait for more information to predict precisely'''
                return AssessResult.Good
            elif predict_y / self.completed_best_performance > THRESHOLD:
                return AssessResult.Good
            else:
                return AssessResult.Bad

        except Exception as e:
            logger.warning('unrecognize exception in curvefitting_asserssor', e)
