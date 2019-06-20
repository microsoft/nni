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
'''
gp_tuner.py
'''

import warnings
import logging
import numpy as np

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward

from .target_space import TargetSpace
from .util import UtilityFunction, acq_max, ensure_rng

logger = logging.getLogger("GP_Tuner_AutoML")


class GPTuner(Tuner):
    '''
    GPTuner
    '''

    def __init__(self, optimize_mode="maximize", cold_start_num=3, random_state=None):
        self.optimize_mode = optimize_mode
        self._random_state = ensure_rng(random_state)

        self._space = None
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25,
            random_state=self._random_state
        )

        self.cold_start_num = cold_start_num
        self.supplement_data_num = 0

    def update_search_space(self, search_space):
        """Update the self.x_bounds and self.x_types by the search_space.json

        Parameters
        ----------
        search_space : dict
        """
        self._space = TargetSpace(search_space, self._random_state)

    def generate_parameters(self, parameter_id):
        """Generate next parameter for trial
        If the number of trial result is lower than cold start number,
        metis will first random generate some parameters.
        Otherwise, choose the parameters by the Gussian Process Model

        Parameters
        ----------
        parameter_id : int

        Returns
        -------
        result : dict
        """
        """Most promissing point to probe next"""
        if len(self._space) == 0 or len(self._space._target) < self.cold_start_num:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        util = UtilityFunction(kind='ei', kappa=0, xi=0)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=util.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state,
            space=self._space
        )
        
        logger.info("Generate paramageters(array):\n" + str(suggestion))
        print("Generate paramageters(array):\n" + str(suggestion))

        results = self._space.array_to_params(suggestion)
        logger.info("Generate paramageters(json):\n" + str(results))
        print("Generate paramageters(json):\n" + str(results))

        return results

    def receive_trial_result(self, parameter_id, parameters, value):
        """Tuner receive result from trial.

        Parameters
        ----------
        parameter_id : int
        parameters : dict
        value : dict/float
            if value is dict, it should have "default" key.
        """
        logger.info("Received trial result.")
        logger.info("value is :" + str(value))
        logger.info("parameter is : " + str(parameters))
        self._space.register(parameters, value)

    def import_data(self, data):
        """Import additional data for tuning
        Parameters
        ----------
        data:
            a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'
        """
        _completed_num = 0
        for trial_info in data:
            logger.info("Importing data, current processing progress %s / %s" %
                        (_completed_num, len(data)))
            _completed_num += 1
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                logger.info(
                    "Useless trial data, value is %s, skip this trial data." % _value)
                continue
            self.supplement_data_num += 1
            _parameter_id = '_'.join(
                ["ImportData", str(self.supplement_data_num)])
            self.receive_trial_result(
                parameter_id=_parameter_id, parameters=_params, value=_value)
        logger.info("Successfully import data to GP tuner.")
