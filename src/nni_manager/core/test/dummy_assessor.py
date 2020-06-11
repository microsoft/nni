# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from nni.assessor import Assessor, AssessResult

class DummyAssessor(Assessor):
    def assess_trial(self, trial_job_id, trial_history):
        return AssessResult.Good
