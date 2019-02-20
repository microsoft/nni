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

logger = logging.getLogger('medianstop_Assessor')

class MedianstopAssessor(Assessor):
    """MedianstopAssessor is The median stopping rule stops a pending trial X at step S 
    if the trial’s best objective value by step S is strictly worse than the median value 
    of the running averages of all completed trials’ objectives reported up to step S
    
    Parameters
    ----------
    optimize_mode: str
        optimize mode, 'maximize' or 'minimize'
    start_step: int
        only after receiving start_step number of reported intermediate results
    """
    def __init__(self, optimize_mode='maximize', start_step=0):
        self.start_step = start_step
        self.running_history = dict()
        self.completed_avg_history = dict()
        if optimize_mode == 'maximize':
            self.high_better = True
        elif optimize_mode == 'minimize':
            self.high_better = False
        else:
            self.high_better = True
            logger.warning('unrecognized optimize_mode', optimize_mode)

    def _update_data(self, trial_job_id, trial_history):
        """update data

        Parameters
        ----------
        trial_job_id: int
            trial job id
        trial_history: list
            The history performance matrix of each trial
        """
        if trial_job_id not in self.running_history:
            self.running_history[trial_job_id] = []
        self.running_history[trial_job_id].extend(trial_history[len(self.running_history[trial_job_id]):])

    def trial_end(self, trial_job_id, success):
        """trial_end
        
        Parameters
        ----------
        trial_job_id: int
            trial job id
        success: bool
            True if succssfully finish the experiment, False otherwise
        """
        if trial_job_id in self.running_history:
            if success:
                cnt = 0
                history_sum = 0
                self.completed_avg_history[trial_job_id] = []
                for each in self.running_history[trial_job_id]:
                    cnt += 1
                    history_sum += each
                    self.completed_avg_history[trial_job_id].append(history_sum / cnt)
            self.running_history.pop(trial_job_id)
        else:
            logger.warning('trial_end: trial_job_id does not in running_history')

    def assess_trial(self, trial_job_id, trial_history):
        """assess_trial
        
        Parameters
        ----------
        trial_job_id: int
            trial job id
        trial_history: list
            The history performance matrix of each trial

        Returns
        -------
        bool
            AssessResult.Good or AssessResult.Bad

        Raises
        ------
        Exception
            unrecognize exception in medianstop_assessor
        """
        curr_step = len(trial_history)
        if curr_step < self.start_step:
            return AssessResult.Good

        try:
            num_trial_history = [float(ele) for ele in trial_history]
        except (TypeError, ValueError) as error:
            logger.warning('incorrect data type or value:')
            logger.exception(error)
        except Exception as error:
            logger.warning('unrecognized exception in medianstop_assessor:')
            logger.excpetion(error)

        self._update_data(trial_job_id, num_trial_history)
        if self.high_better:
            best_history = max(trial_history)
        else:
            best_history = min(trial_history)

        avg_array = []
        for id in self.completed_avg_history:
            if len(self.completed_avg_history[id]) >= curr_step:
                avg_array.append(self.completed_avg_history[id][curr_step - 1])
        if len(avg_array) > 0:
            avg_array.sort()
            if self.high_better:
                median = avg_array[(len(avg_array)-1) // 2]
                return AssessResult.Bad if best_history < median else AssessResult.Good
            else:
                median = avg_array[len(avg_array) // 2]
                return AssessResult.Bad if best_history > median else AssessResult.Good
        else:
            return AssessResult.Good
