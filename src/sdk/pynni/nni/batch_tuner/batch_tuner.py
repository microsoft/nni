# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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

LOGGER = logging.getLogger('batch_tuner_AutoML')

class BatchTuner(Tuner):
    """
    BatchTuner is tuner will running all the configure that user want to run batchly.

    Examples
    --------
    The search space only be accepted like:

        ::

            {'combine_params':
                { '_type': 'choice',
                            '_value': '[{...}, {...}, {...}]',
                }
            }

    """

    def __init__(self):
        self._count = -1
        self._values = []

    def is_valid(self, search_space):
        """
        Check the search space is valid: only contains 'choice' type

        Parameters
        ----------
        search_space : dict

        Returns
        -------
        None or list
            If valid, return candidate values; else return None.
        """
        if not len(search_space) == 1:
            raise RuntimeError('BatchTuner only supprt one combined-paramreters key.')

        for param in search_space:
            param_type = search_space[param][TYPE]
            if not param_type == CHOICE:
                raise RuntimeError('BatchTuner only supprt \
                                    one combined-paramreters type is choice.')

            if isinstance(search_space[param][VALUE], list):
                return search_space[param][VALUE]

            raise RuntimeError('The combined-paramreters \
                                value in BatchTuner is not a list.')
        return None

    def update_search_space(self, search_space):
        """Update the search space

        Parameters
        ----------
        search_space : dict
        """
        self._values = self.is_valid(search_space)

    def generate_parameters(self, parameter_id, **kwargs):
        """Returns a dict of trial (hyper-)parameters, as a serializable object.

        Parameters
        ----------
        parameter_id : int

        Returns
        -------
        dict
            A candidate parameter group.
        """
        self._count += 1
        if self._count > len(self._values) - 1:
            raise nni.NoMoreTrialError('no more parameters now.')
        return self._values[self._count]

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        pass

    def import_data(self, data):
        """Import additional data for tuning

        Parameters
        ----------
        data:
            a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'
        """
        if not self._values:
            LOGGER.info("Search space has not been initialized, skip this data import")
            return

        self._values = self._values[(self._count+1):]
        self._count = -1

        _completed_num = 0
        for trial_info in data:
            LOGGER .info("Importing data, current processing \
                            progress %s / %s", _completed_num, len(data))
            # simply validate data format
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                LOGGER.info("Useless trial data, value is %s, skip this trial data.", _value)
                continue
            _completed_num += 1
            if _params in self._values:
                self._values.remove(_params)
        LOGGER .info("Successfully import data to batch tuner, \
                        total data: %d, imported data: %d.", len(data), _completed_num)
