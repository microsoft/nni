# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import datetime
from schema import Schema, Optional

from nni import ClassArgsValidator
from nni.assessor import Assessor, AssessResult
from nni.utils import extract_scalar_history
from .model_factory import CurveModel

logger = logging.getLogger('curvefitting_Assessor')

class CurvefittingClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            'epoch_num': self.range('epoch_num', int, 0, 9999),
            Optional('start_step'): self.range('start_step', int, 0, 9999),
            Optional('threshold'): self.range('threshold', float, 0, 9999),
            Optional('gap'): self.range('gap', int, 1, 9999),
        }).validate(kwargs)

class CurvefittingAssessor(Assessor):
    """CurvefittingAssessor uses learning curve fitting algorithm to predict the learning curve performance in the future.
    It stops a pending trial X at step S if the trial's forecast result at target step is convergence and lower than the
    best performance in the history.

    Parameters
    ----------
    epoch_num : int
        The total number of epoch
    start_step : int
        only after receiving start_step number of reported intermediate results
    threshold : float
        The threshold that we decide to early stop the worse performance curve.
    """

    def __init__(self, epoch_num=20, start_step=6, threshold=0.95, gap=1):
        if start_step <= 0:
            logger.warning('It\'s recommended to set start_step to a positive number')
        # Record the target position we predict
        self.target_pos = epoch_num
        # Start forecasting when historical data reaches start step
        self.start_step = start_step
        # Record the compared threshold
        self.threshold = threshold
        # Record the number of gap
        self.gap = gap
        # Record the number of intermediate result in the lastest judgment
        self.last_judgment_num = dict()
        # Record the best performance
        self.set_best_performance = False
        self.completed_best_performance = None
        self.trial_history = []
        logger.info('Successfully initials the curvefitting assessor')

    def trial_end(self, trial_job_id, success):
        """update the best performance of completed trial job

        Parameters
        ----------
        trial_job_id : int
            trial job id
        success : bool
            True if succssfully finish the experiment, False otherwise
        """
        if success:
            if self.set_best_performance:
                self.completed_best_performance = max(self.completed_best_performance, self.trial_history[-1])
            else:
                self.set_best_performance = True
                self.completed_best_performance = self.trial_history[-1]
            logger.info('Updated complted best performance, trial job id: %s', trial_job_id)
        else:
            logger.info('No need to update, trial job id: %s', trial_job_id)

    def assess_trial(self, trial_job_id, trial_history):
        """assess whether a trial should be early stop by curve fitting algorithm

        Parameters
        ----------
        trial_job_id : int
            trial job id
        trial_history : list
            The history performance matrix of each trial

        Returns
        -------
        bool
            AssessResult.Good or AssessResult.Bad

        Raises
        ------
        Exception
            unrecognize exception in curvefitting_assessor
        """
        scalar_trial_history = extract_scalar_history(trial_history)
        self.trial_history = scalar_trial_history
        if not self.set_best_performance:
            return AssessResult.Good
        curr_step = len(scalar_trial_history)
        if curr_step < self.start_step:
            return AssessResult.Good

        if trial_job_id in self.last_judgment_num.keys() and curr_step - self.last_judgment_num[trial_job_id] < self.gap:
            return AssessResult.Good
        self.last_judgment_num[trial_job_id] = curr_step

        try:
            start_time = datetime.datetime.now()
            # Predict the final result
            curvemodel = CurveModel(self.target_pos)
            predict_y = curvemodel.predict(scalar_trial_history)
            log_message = "Prediction done. Trial job id = {}, Predict value = {}".format(trial_job_id, predict_y)
            if predict_y is None:
                logger.info('%s, wait for more information to predict precisely', log_message)
                return AssessResult.Good
            else:
                logger.info(log_message)
            standard_performance = self.completed_best_performance * self.threshold

            end_time = datetime.datetime.now()
            if (end_time - start_time).seconds > 60:
                logger.warning(
                    'Curve Fitting Assessor Runtime Exceeds 60s, Trial Id = %s Trial History = %s',
                    trial_job_id, self.trial_history
                )

            if predict_y > standard_performance:
                return AssessResult.Good
            return AssessResult.Bad

        except Exception as exception:
            logger.exception('unrecognize exception in curvefitting_assessor %s', exception)
