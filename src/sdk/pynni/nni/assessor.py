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


from collections import defaultdict
from enum import Enum
import logging
import os

import json_tricks

from .common import init_logger
from .protocol import CommandType, send, receive


init_logger('assessor.log')
_logger = logging.getLogger(__name__)


class AssessResult(Enum):
    Good = True
    Bad = False


class Assessor:
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

    def load_checkpoint(self, path):
        """Load the checkpoint of assessor.
        path: checkpoint directory of assessor
        """
        _logger.info('Load checkpoint ignored by assessor')

    def save_checkpoint(self, path):
        """Save the checkpoint of assessor.
        path: checkpoint directory of assessor
        """
        _logger.info('Save checkpoint ignored by assessor')

    def request_save_checkpoint(self):
        """Request to save the checkpoint of assessor
        """
        self.save_checkpoint(os.getenv('NNI_CHECKPOINT_DIRECTORY'))

    def run(self):
        """Run the assessor.
        This function will never return unless raise.
        """
        mode = os.getenv('NNI_MODE')
        if mode == 'resume':
            self.load_checkpoint(os.getenv('NNI_CHECKPOINT_DIRECTORY'))
        while _handle_request(self):
            pass
        _logger.info('Terminated by NNI manager')


_trial_history = defaultdict(dict)
'''key: trial job ID; value: intermediate results, mapping from sequence number to data'''

_ended_trials = set()
'''trial_job_id of all ended trials.
We need this because NNI manager may send metrics after reporting a trial ended.
TODO: move this logic to NNI manager
'''

def _sort_history(history):
    ret = [ ]
    for i, _ in enumerate(history):
        if i in history:
            ret.append(history[i])
        else:
            break
    return ret

def _handle_request(assessor):
    _logger.debug('waiting receive_message')

    command, data = receive()

    _logger.debug(command)
    _logger.debug(data)

    if command is CommandType.Terminate:
        return False

    data = json_tricks.loads(data)

    if command is CommandType.ReportMetricData:
        if data['type'] != 'PERIODICAL':
            return True

        trial_job_id = data['trial_job_id']
        if trial_job_id in _ended_trials:
            return True

        history = _trial_history[trial_job_id]
        history[data['sequence']] = data['value']
        ordered_history = _sort_history(history)
        if len(ordered_history) < data['sequence']:  # no user-visible update since last time
            return True

        result = assessor.assess_trial(trial_job_id, ordered_history)
        if isinstance(result, bool):
            result = AssessResult.Good if result else AssessResult.Bad
        elif not isinstance(result, AssessResult):
            msg = 'Result of Assessor.assess_trial must be an object of AssessResult, not %s'
            raise RuntimeError(msg % type(result))

        if result is AssessResult.Bad:
            _logger.debug('BAD, kill %s', trial_job_id)
            send(CommandType.KillTrialJob, json_tricks.dumps(trial_job_id))
        else:
            _logger.debug('GOOD')

    elif command is CommandType.TrialEnd:
        trial_job_id = data['trial_job_id']
        _ended_trials.add(trial_job_id)
        if trial_job_id in _trial_history:
            _trial_history.pop(trial_job_id)
            assessor.trial_end(trial_job_id, data['event'] == 'SUCCEEDED')

    else:
        raise AssertionError('Unsupported command: %s' % command)

    return True
