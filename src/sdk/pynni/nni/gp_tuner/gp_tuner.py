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
from .util import UtilityFunction, acq_max

logger = logging.getLogger("GP_Tuner_AutoML")


class GPTuner(Tuner):
    '''
    GPTuner
    '''

    def __init__(self, optimize_mode="maximize", utility='ei', kappa=5, xi=0, nu=2.5, alpha=1e-6, cold_start_num=10,
                 selection_num_warm_up=100000, selection_num_starting_points=250):
        self.optimize_mode = OptimizeMode(optimize_mode)

        # utility function related
        self.utility = utility
        self.kappa = kappa
        self.xi = xi

        # target space
        self._space = None

        self._random_state = np.random.RandomState()

        # nu, alpha are GPR related params
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=nu),
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=25,
            random_state=self._random_state
        )
        # num of random evaluations before GPR
        self._cold_start_num = cold_start_num

        # params for acq_max
        self._selection_num_warm_up = selection_num_warm_up
        self._selection_num_starting_points = selection_num_starting_points

        # num of imported data
        self.supplement_data_num = 0

    def update_search_space(self, search_space):
        """Update the self.bounds and self.types by the search_space.json

        Parameters
        ----------
        search_space : dict
        """
        self._space = TargetSpace(search_space, self._random_state)

    def generate_parameters(self, parameter_id):
        """Generate next parameter for trial
        If the number of trial result is lower than cold start number,
        gp will first randomly generate some parameters.
        Otherwise, choose the parameters by the Gussian Process Model

        Parameters
        ----------
        parameter_id : int

        Returns
        -------
        result : dict
        """
        if self._space.len() < self._cold_start_num:
            results = self._space.random_sample()
        else:
            # Sklearn's GP throws a large number of warnings at times, but
            # we don't really need to see them here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._gp.fit(self._space.params, self._space.target)

            util = UtilityFunction(
                kind=self.utility, kappa=self.kappa, xi=self.xi)

            results = acq_max(
                f_acq=util.utility,
                gp=self._gp,
                y_max=self._space.target.max(),
                bounds=self._space.bounds,
                space=self._space,
                num_warmup=self._selection_num_warm_up,
                num_starting_points=self._selection_num_starting_points
            )

        results = self._space.array_to_params(results)
        logger.info("Generate paramageters:\n %s", results)
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
        value = extract_scalar_reward(value)
        if self.optimize_mode == OptimizeMode.Minimize:
            value = -value

        logger.info("Received trial result.")
        logger.info("value :%s", value)
        logger.info("parameter : %s", parameters)
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
