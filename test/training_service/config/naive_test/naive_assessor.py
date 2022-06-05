# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os

from nni.assessor import Assessor, AssessResult

_logger = logging.getLogger('NaiveAssessor')
_logger.info('start')

_pwd = os.path.dirname(__file__)
_result = open(os.path.join(_pwd, 'assessor_result.txt'), 'w')

class NaiveAssessor(Assessor):
    def __init__(self, optimize_mode):
        self._killed = set()
        _logger.info('init')

    def assess_trial(self, trial_job_id, trial_history):
        _logger.info('assess trial %s %s', trial_job_id, trial_history)

        id_ = trial_history[0]
        if id_ in self._killed:
            return AssessResult.Bad

        s = 0
        for i, val in enumerate(trial_history):
            s += val
            if s % 11 == 1:
                self._killed.add(id_)
                _result.write('%d %d\n' % (id_, i + 1))
                _result.flush()
                return AssessResult.Bad

        return AssessResult.Good

    def _on_exit(self):
        _result.close()

    def _on_error(self):
        _result.write('ERROR\n')
        _result.close()
