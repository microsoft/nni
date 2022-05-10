# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import warnings

import numpy as np
import torch
from pybnn import DNGO
from torch.distributions import Normal

import nni.parameter_expressions as parameter_expressions
from nni import ClassArgsValidator
from nni.common.hpo_utils import validate_search_space
from nni.tuner import Tuner

_logger = logging.getLogger(__name__)


def _random_config(search_space, random_state):
    chosen_config = {}
    for key, val in search_space.items():
        if val['_type'] == 'choice':
            choices = val['_value']
            index = random_state.randint(len(choices))
            if all([isinstance(c, (int, float)) for c in choices]):
                chosen_config[key] = choices[index]
            else:
                raise ValueError('Choices with type other than int and float is not supported.')
        elif val['_type'] == 'uniform':
            chosen_config[key] = random_state.uniform(val['_value'][0], val['_value'][1])
        elif val['_type'] == 'randint':
            chosen_config[key] = random_state.randint(
                val['_value'][0], val['_value'][1])
        elif val['_type'] == 'quniform':
            chosen_config[key] = parameter_expressions.quniform(
                val['_value'][0], val['_value'][1], val['_value'][2], random_state)
        elif val['_type'] == 'loguniform':
            chosen_config[key] = parameter_expressions.loguniform(
                val['_value'][0], val['_value'][1], random_state)
        elif val['_type'] == 'qloguniform':
            chosen_config[key] = parameter_expressions.qloguniform(
                val['_value'][0], val['_value'][1], val['_value'][2], random_state)
        else:
            raise ValueError('Unknown key %s and value %s' % (key, val))
    return chosen_config


class DNGOTuner(Tuner):
    """
    Use neural networks as an alternative to GPs to model distributions over functions in bayesian optimization.

    Parameters
    ----------
    optimize : maximize | minimize, default = maximize
        If 'maximize', the tuner will target to maximize metrics. If 'minimize', the tuner will target to minimize metrics.
    sample_size : int, default = 1000
        Number of samples to select in each iteration. The best one will be picked from the samples as the next trial.
    trials_per_update : int, default = 20
        Number of trials to collect before updating the model.
    num_epochs_per_training : int, default = 500
        Number of epochs to train DNGO model.
    """
    def __init__(self, optimize_mode='maximize', sample_size=1000, trials_per_update=20, num_epochs_per_training=500):
        self.searchspace_json = None
        self.random_state = None
        self.model = DNGO(do_mcmc=False, num_epochs=num_epochs_per_training)
        self._model_initialized = False
        self.sample_size = sample_size
        self.trials_per_update = trials_per_update
        self.optimize_mode = optimize_mode

        self.x = []
        self.y = []

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        self.x.append(parameters)
        self.y.append(self._get_default_value(value))
        if len(self.y) % self.trials_per_update == 0:
            self._update_model()

    def generate_parameters(self, parameter_id, **kwargs):
        if not self._model_initialized:
            return _random_config(self.searchspace_json, self.random_state)
        else:
            # random samples and pick best with model
            candidate_x = [_random_config(self.searchspace_json, self.random_state) for _ in range(self.sample_size)]

            # The model has NaN issue when all the candidates are same
            # Also we can save the predict time when this happens
            if all(x == candidate_x[0] for x in candidate_x):
                return candidate_x[0]

            x_test = np.array([np.array(list(xi.values())) for xi in candidate_x])
            m, v = self.model.predict(x_test)

            # The model has NaN issue when all the candidates are very close
            if np.isnan(m).any() or np.isnan(v).any():
                return candidate_x[0]

            mean = torch.Tensor(m)
            sigma = torch.Tensor(v)
            u = (mean - torch.Tensor([0.95]).expand_as(mean)) / sigma
            normal = Normal(torch.zeros_like(u), torch.ones_like(u))
            ucdf = normal.cdf(u)
            updf = torch.exp(normal.log_prob(u))
            ei = sigma * (updf + u * ucdf)

            if self.optimize_mode == 'maximize':
                ind = torch.argmax(ei)
            else:
                ind = torch.argmin(ei)
            new_x = candidate_x[ind]
            return new_x

    def update_search_space(self, search_space):
        validate_search_space(search_space, ['choice', 'randint', 'uniform', 'quniform', 'loguniform', 'qloguniform'])
        self.searchspace_json = search_space
        self.random_state = np.random.RandomState()

    def import_data(self, data):
        for d in data:
            self.x.append(d['parameter'])
            self.y.append(self._get_default_value(d['value']))
        self._update_model()

    def _update_model(self):
        _logger.info('Updating model on %d samples', len(self.x))
        x_arr = []
        for x in self.x:
            x_arr.append([x[k] for k in sorted(x.keys())])
        try:
            self.model.train(np.array(x_arr), np.array(self.y), do_optimize=True)
        except np.linalg.LinAlgError as e:
            warnings.warn(f'numpy linalg error encountered in DNGO model training: {e}')
        self._model_initialized = True

    def _get_default_value(self, value):
        if isinstance(value, dict) and 'default' in value:
            return value['default']
        elif isinstance(value, float):
            return value
        else:
            raise ValueError(f'Unsupported value: {value}')


class DNGOClassArgsValidator(ClassArgsValidator):
    # DNGO tuner do not have much input arg, so the validation is actually hardly used
    def validate_class_args(self, **kwargs):
        pass
