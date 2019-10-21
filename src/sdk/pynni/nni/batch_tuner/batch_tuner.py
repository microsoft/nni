# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
batch_tuner.py including:
    class BatchTuner
"""

import logging

import nni
from nni.tuner import Tuner

TYPE = '_type'
CHOICE = 'choice'
VALUE = '_value'

logger = logging.getLogger('batch_tuner_AutoML')

class BatchTuner(Tuner):
    """
    BatchTuner is tuner will running all the configure that user want to run batchly.
    The search space only be accepted like:
    {
        'combine_params': { '_type': 'choice',
                             '_value': '[{...}, {...}, {...}]',
                          }
    }
    """

    def __init__(self):
        self.count = -1
        self.values = []

    def is_valid(self, search_space):
        """
        Check the search space is valid: only contains 'choice' type

        Parameters
        ----------
        search_space : dict
        """
        if not len(search_space) == 1:
            raise RuntimeError('BatchTuner only supprt one combined-paramreters key.')

        for param in search_space:
            param_type = search_space[param][TYPE]
            if not param_type == CHOICE:
                raise RuntimeError('BatchTuner only supprt one combined-paramreters type is choice.')
            else:
                if isinstance(search_space[param][VALUE], list):
                    return search_space[param][VALUE]
                raise RuntimeError('The combined-paramreters value in BatchTuner is not a list.')
        return None

    def update_search_space(self, search_space):
        """Update the search space

        Parameters
        ----------
        search_space : dict
        """
        self.values = self.is_valid(search_space)

    def generate_parameters(self, parameter_id, **kwargs):
        """Returns a dict of trial (hyper-)parameters, as a serializable object.

        Parameters
        ----------
        parameter_id : int
        """
        self.count += 1
        if self.count > len(self.values) - 1:
            raise nni.NoMoreTrialError('no more parameters now.')
        return self.values[self.count]

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        pass

    def import_data(self, data):
        """Import additional data for tuning
        Parameters
        ----------
        data:
            a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'
        """
        if len(self.values) == 0:
            logger.info("Search space has not been initialized, skip this data import")
            return

        self.values = self.values[(self.count+1):]
        self.count = -1

        _completed_num = 0
        for trial_info in data:
            logger.info("Importing data, current processing progress %s / %s", _completed_num, len(data))
            # simply validate data format
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                logger.info("Useless trial data, value is %s, skip this trial data.", _value)
                continue
            _completed_num += 1
            if _params in self.values:
                self.values.remove(_params)
        logger.info("Successfully import data to batch tuner, total data: %d, imported data: %d.", len(data), _completed_num)
