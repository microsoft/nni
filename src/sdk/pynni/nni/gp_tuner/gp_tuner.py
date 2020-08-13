# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
GPTuner is a Bayesian Optimization method where Gaussian Process is used for modeling loss functions.

See :class:`GPTuner` for details.
"""

import warnings
import logging
import numpy as np
from schema import Schema, Optional

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from nni import ClassArgsValidator
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward

from .target_space import TargetSpace
from .util import UtilityFunction, acq_max

logger = logging.getLogger("GP_Tuner_AutoML")

class GPClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            Optional('optimize_mode'): self.choices('optimize_mode', 'maximize', 'minimize'),
            Optional('utility'): self.choices('utility', 'ei', 'ucb', 'poi'),
            Optional('kappa'): float,
            Optional('xi'): float,
            Optional('nu'): float,
            Optional('alpha'): float,
            Optional('cold_start_num'): int,
            Optional('selection_num_warm_up'):  int,
            Optional('selection_num_starting_points'):  int,
        }).validate(kwargs)

class GPTuner(Tuner):
    """
    GPTuner is a Bayesian Optimization method where Gaussian Process is used for modeling loss functions.

    Parameters
    ----------
    optimize_mode : str
        optimize mode, 'maximize' or 'minimize', by default 'maximize'
    utility : str
        utility function (also called 'acquisition funcition') to use, which can be 'ei', 'ucb' or 'poi'. By default 'ei'.
    kappa : float
        value used by utility function 'ucb'. The bigger kappa is, the more the tuner will be exploratory. By default 5.
    xi : float
        used by utility function 'ei' and 'poi'. The bigger xi is, the more the tuner will be exploratory. By default 0.
    nu : float
        used to specify Matern kernel. The smaller nu, the less smooth the approximated function is. By default 2.5.
    alpha : float
        Used to specify Gaussian Process Regressor. Larger values correspond to increased noise level in the observations.
        By default 1e-6.
    cold_start_num : int
        Number of random exploration to perform before Gaussian Process. By default 10.
    selection_num_warm_up : int
        Number of random points to evaluate for getting the point which maximizes the acquisition function. By default 100000
    selection_num_starting_points : int
        Number of times to run L-BFGS-B from a random starting point after the warmup. By default 250.
    """

    def __init__(self, optimize_mode="maximize", utility='ei', kappa=5, xi=0, nu=2.5, alpha=1e-6, cold_start_num=10,
                 selection_num_warm_up=100000, selection_num_starting_points=250):
        self._optimize_mode = OptimizeMode(optimize_mode)

        # utility function related
        self._utility = utility
        self._kappa = kappa
        self._xi = xi

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
        self._supplement_data_num = 0

    def update_search_space(self, search_space):
        """
        Update the self.bounds and self.types by the search_space.json file.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
        """
        self._space = TargetSpace(search_space, self._random_state)

    def generate_parameters(self, parameter_id, **kwargs):
        """
        Method which provides one set of hyper-parameters.
        If the number of trial result is lower than cold_start_number, GPTuner will first randomly generate some parameters.
        Otherwise, choose the parameters by the Gussian Process Model.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
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
                kind=self._utility, kappa=self._kappa, xi=self._xi)

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

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        Method invoked when a trial reports its final result.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
        """
        value = extract_scalar_reward(value)
        if self._optimize_mode == OptimizeMode.Minimize:
            value = -value

        logger.info("Received trial result.")
        logger.info("value :%s", value)
        logger.info("parameter : %s", parameters)
        self._space.register(parameters, value)

    def import_data(self, data):
        """
        Import additional data for tuning.

        Override of the abstract method in :class:`~nni.tuner.Tuner`.
        """
        _completed_num = 0
        for trial_info in data:
            logger.info(
                "Importing data, current processing progress %s / %s", _completed_num, len(data))
            _completed_num += 1
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                logger.info(
                    "Useless trial data, value is %s, skip this trial data.", _value)
                continue
            self._supplement_data_num += 1
            _parameter_id = '_'.join(
                ["ImportData", str(self._supplement_data_num)])
            self.receive_trial_result(
                parameter_id=_parameter_id, parameters=_params, value=_value)
        logger.info("Successfully import data to GP tuner.")
