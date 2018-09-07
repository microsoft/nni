# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================


import logging
from enum import Enum

from .recoverable import Recoverable

_logger = logging.getLogger(__name__)

class AssessResult(Enum):
    Good = True
    Bad = False

class Assessor(Recoverable):
    # pylint: disable=no-self-use,unused-argument

    def assess_trial(self, trial_job_id, trial_history):
        """Determines whether a trial should be killed. Must override.
        trial_job_id: identifier of the trial (str).
        trial_history: a list of intermediate result objects.
        Returns AssessResult.Good or AssessResult.Bad.
        """
        raise NotImplementedError('Assessor: assess_trial not implemented')

    def trial_end(self, trial_job_id, success):
        """Invoked when a trial is completed or terminated. Do nothing by default.
        trial_job_id: identifier of the trial (str).
        success: True if the trial successfully completed; False if failed or terminated.
        """
        pass

    def load_checkpoint(self):
        """Load the checkpoint of assessr.
        path: checkpoint directory for assessor
        """
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Load checkpoint ignored by assessor, checkpoint path: %s' % checkpoin_path)

    def save_checkpoint(self):
        """Save the checkpoint of assessor.
        path: checkpoint directory for assessor
        """
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Save checkpoint ignored by assessor, checkpoint path: %s' % checkpoin_path)

    def _on_exit(self):
        pass

    def _on_error(self):
        pass
