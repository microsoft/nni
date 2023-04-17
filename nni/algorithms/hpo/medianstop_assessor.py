# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging

from schema import Schema, Optional
from typing_extensions import Literal

from nni import ClassArgsValidator
from nni.assessor import Assessor, AssessResult
from nni.utils import extract_scalar_history

logger = logging.getLogger('medianstop_Assessor')

class MedianstopClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            Optional('optimize_mode'): self.choices('optimize_mode', 'maximize', 'minimize'),
            Optional('start_step'): self.range('start_step', int, 0, 9999),
        }).validate(kwargs)

class MedianstopAssessor(Assessor):
    """
    The median stopping rule stops a pending trial X at step S
    if the trial’s best objective value by step S is strictly worse than the median value
    of the running averages of all completed trials’ objectives reported up to step S

    Paper: `Google Vizer: A Service for Black-Box Optimization
    <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf>`__

    Examples
    --------

    .. code-block::

        config.assessor.name = 'Medianstop'
        config.assessor.class_args = {
            'optimize_mode': 'maximize',
            'start_step': 5
        }

    Parameters
    ----------
    optimize_mode
        Whether optimize to minimize or maximize trial result.
    start_step
        A trial is determined to be stopped or not
        only after receiving start_step number of reported intermediate results.
    """

    def __init__(self, optimize_mode: Literal['minimize', 'maximize'] = 'maximize', start_step: int = 0):
        self._start_step = start_step
        self._running_history = dict()
        self._completed_avg_history = dict()
        if optimize_mode == 'maximize':
            self._high_better = True
        elif optimize_mode == 'minimize':
            self._high_better = False
        else:
            self._high_better = True
            logger.warning('unrecognized optimize_mode %s', optimize_mode)

    def _update_data(self, trial_job_id, trial_history):
        """update data

        Parameters
        ----------
        trial_job_id : int
            trial job id
        trial_history : list
            The history performance matrix of each trial
        """
        if trial_job_id not in self._running_history:
            self._running_history[trial_job_id] = []
        self._running_history[trial_job_id].extend(trial_history[len(self._running_history[trial_job_id]):])

    def trial_end(self, trial_job_id, success):
        if trial_job_id in self._running_history:
            if success:
                cnt = 0
                history_sum = 0
                self._completed_avg_history[trial_job_id] = []
                for each in self._running_history[trial_job_id]:
                    cnt += 1
                    history_sum += each
                    self._completed_avg_history[trial_job_id].append(history_sum / cnt)
            self._running_history.pop(trial_job_id)
        else:
            logger.warning('trial_end: trial_job_id does not exist in running_history')

    def assess_trial(self, trial_job_id, trial_history):
        curr_step = len(trial_history)
        if curr_step < self._start_step:
            return AssessResult.Good

        scalar_trial_history = extract_scalar_history(trial_history)
        self._update_data(trial_job_id, scalar_trial_history)
        if self._high_better:
            best_history = max(scalar_trial_history)
        else:
            best_history = min(scalar_trial_history)

        avg_array = []
        for id_ in self._completed_avg_history:
            if len(self._completed_avg_history[id_]) >= curr_step:
                avg_array.append(self._completed_avg_history[id_][curr_step - 1])
        if avg_array:
            avg_array.sort()
            if self._high_better:
                median = avg_array[(len(avg_array)-1) // 2]
                return AssessResult.Bad if best_history < median else AssessResult.Good
            else:
                median = avg_array[len(avg_array) // 2]
                return AssessResult.Bad if best_history > median else AssessResult.Good
        else:
            return AssessResult.Good
