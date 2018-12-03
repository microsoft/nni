# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the 'Software'), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]) 
import logging
import numpy as np
from assessor import Assessor, AssessResult
from curvemodelfactory import MLCurveModel
from mcmc_modelfactory import MCMCCurveModelCombination
from curvefunctions import curve_combination_models, model_defaults, all_models

logger = logging.getLogger('curvefitting_Assessor')

IMPROVEMENT_PROB_THERSHOLD = 0.05
LOOK_AHEAD = 2
XLIM = 20
PREDICTION_THINNING = 2
# Belows are by default:
# threads = 1
# mode = conservative
# greater_type =posterior_prob_x_greater_than


def cut_beginning(y, threshold=IMPROVEMENT_PROB_THERSHOLD, look_ahead=LOOK_AHEAD):
    '''
    choose the start point where is bigger than the initial value for look_ahead steps
    '''
    if len(y) < look_ahead:
        return y
    num_cut = 0
    for idx in range(len(y) - look_ahead):
        start_here = True
        for idx_ahead in range(idx, idx + look_ahead):
            if not (y[idx_ahead] - y[0] > threshold):
                start_here = False
        if start_here:
            num_cut = idx
            break
    return y[num_cut:]

# recency_weight = False
# normalize_weights = True
# monotonicity_constraint = False
# soft_monotonicity_constraint = True

def setup_model_combination(xlim, models=curve_combination_models):
    curve_models = []
    for model_name in models:
        m = MLCurveModel(function=all_models[model_name], recency_weighting=False)
        curve_models.append(m)

    model_combination = MCMCCurveModelCombination(curve_models, xlim=xlim)
    return model_combination

class TerminationCriterion(object):
    def __init__(self, trial_history, completed_best_history=None, xlim=XLIM):
        models = ['vap', 'pow3', 'loglog_linear', 'dr_hill_zero_background', 'log_power', 'pow4', 'mmf', 'exp4', 'ilog2', 'weibull', 'janoschek']
        self.trial_history = trial_history
        self.completed_best_history = completed_best_history
        self.xlim = xlim
        model = setup_model_combination(xlim, models=models)
        self.model = model

    def run(self):
        y = self.trial_history
        y_curr_best = np.max(y)
        if y_curr_best > self.completed_best_history:
            return AssessResult.Good

        y_avg = np.sum(y)/len(y)
        if y_avg < self.completed_best_history / 3:
            return AssessResult.Bad

        y = cut_beginning(y)
        x = np.asarray(range(1, len(y) + 1))
        print ("Purity: Under Termination.run, prepare to fit!")

        if not self.model.fit(x, y):
            return AssessResult.Good
        print ("Purity: Under Termination.run, now y = ", y)

        prob_gt_ybest_xlast = self.model.posterior_prob_x_greater_than(self.xlim, self.completed_best_history, thin=PREDICTION_THINNING)
        print (prob_gt_ybest_xlast)
        if prob_gt_ybest_xlast < IMPROVEMENT_PROB_THERSHOLD:
            return self.predict()
        else:
            return AssessResult.Good

    def predict(self):
        '''
        predict f(x)
        '''
        y_predict = self.model.predict(self.xlim, thin=PREDICTION_THINNING)
        print ("y_predict = ", y_predict)
        if y_predict >= 0. and y_predict <= 1.0:
            return AssessResult.Bad
        else:
            return AssessResult.Good


class CurvefittingAssessor(Assessor):
    '''
    CurvefittingAssessor uses learning curve fitting to predict the future learning curvem, and stops a pending trial X at step S 
    if the trial's predict learning curve was IMPROVEMENT_PROB_THERSHOLD (95% by default) certain that it would not improve over 
    the best known performance trial.

    Referencr paper: Tobias Domhan, Jost Tobias Springenberg, Frank Hutter. Speeding up Automatic Hyperparameter Optimization of 
    Deep Neural Networks by Extrapolation of Learning Curves. IJCAI, 2015.
    '''
    def __init__(self, start_step=0, xlim=XLIM):
        self.start_step = start_step
        self.xlim = xlim
        self.running_history = dict()
        self.completed_best_history = 0.
    
    def _update_data(self, trial_job_id, trial_history):
        if trial_job_id not in self.running_history: 
            self.running_history[trial_job_id] = []
        self.running_history[trial_job_id].extend(trial_history[len(self.running_history[trial_job_id]):])

    def trial_end(self, trial_job_id, success):
        '''
        trial end: record the best performance of completed trial job
        '''
        if trial_job_id in self.running_history:
            if success:
                self.completed_best_history = np.max(self.running_history[trial_job_id])
        else:
            logger.warning('trial_end: trial_job_id does not in running_history')

    def assess_trial(self, trial_job_id, trial_history):
        '''
        assess whether a trial should be early stop by curve fitting
        return AssessResult.Good or AssessResult.Bad
        '''
        curr_step = len(trial_history)
        trial_history = np.argsort(trial_history)
        if curr_step < self.start_step:
            return AssessResult.Good
        
        self._update_data(trial_job_id, trial_history)
        
        # predict and assess
        #try:
        term_crit = TerminationCriterion(trial_history=trial_history, completed_best_history=self.completed_best_history)
        return term_crit.run()
        #except Exception as e:
        #    logger.warning('unrecognize exception in curvefitting_asserssor')
        # return ret
