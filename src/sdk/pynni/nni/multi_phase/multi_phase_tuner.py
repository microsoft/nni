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

from nni.recoverable import Recoverable

_logger = logging.getLogger(__name__)


class MultiPhaseTuner(Recoverable):
    # pylint: disable=no-self-use,unused-argument

    def generate_parameters(self, parameter_id, trial_job_id=None):
        """Returns a set of trial (hyper-)parameters, as a serializable object.
        User code must override either this function or 'generate_multiple_parameters()'.
        parameter_id: identifier of the parameter (int)
        """
        raise NotImplementedError('Tuner: generate_parameters not implemented')

    def generate_multiple_parameters(self, parameter_id_list):
        """Returns multiple sets of trial (hyper-)parameters, as iterable of serializable objects.
        Call 'generate_parameters()' by 'count' times by default.
        User code must override either this function or 'generate_parameters()'.
        parameter_id_list: list of int
        """
        return [self.generate_parameters(parameter_id) for parameter_id in parameter_id_list]

    def receive_trial_result(self, parameter_id, parameters, value, trial_job_id):
        """Invoked when a trial reports its final result. Must override.
        parameter_id: identifier of the parameter (int)
        parameters: object created by 'generate_parameters()'
        value: object reported by trial
        trial_job_id: identifier of the trial (str)
        """
        raise NotImplementedError('Tuner: receive_trial_result not implemented')

    def receive_customized_trial_result(self, parameter_id, parameters, value, trial_job_id):
        """Invoked when a trial added by WebUI reports its final result. Do nothing by default.
        parameter_id: identifier of the parameter (int)
        parameters: object created by user
        value: object reported by trial
        trial_job_id: identifier of the trial (str)
        """
        _logger.info('Customized trial job %s ignored by tuner', parameter_id)

    def trial_end(self, parameter_id, success, trial_job_id):
        """Invoked when a trial is completed or terminated. Do nothing by default.
        parameter_id: identifier of the parameter (int)
        success: True if the trial successfully completed; False if failed or terminated
        trial_job_id: identifier of the trial (str)
        """
        pass

    def update_search_space(self, search_space):
        """Update the search space of tuner. Must override.
        search_space: JSON object
        """
        raise NotImplementedError('Tuner: update_search_space not implemented')

    def import_data(self, data):
        """Import additional data for tuning
        data: a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'
        """
        pass

    def load_checkpoint(self):
        """Load the checkpoint of tuner.
        path: checkpoint directory for tuner
        """
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Load checkpoint ignored by tuner, checkpoint path: %s' % checkpoin_path)

    def save_checkpoint(self):
        """Save the checkpoint of tuner.
        path: checkpoint directory for tuner
        """
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Save checkpoint ignored by tuner, checkpoint path: %s' % checkpoin_path)

    def _on_exit(self):
        pass

    def _on_error(self):
        pass

    def import_data(self, data):
        pass
