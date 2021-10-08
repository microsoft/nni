# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Naive random tuner for hyper-parameter optimization.

You can specify an integer seed to determine random result.
"""

__all__ = ['RandomTuner', 'suggest', 'suggest_parameter']

import numpy as np
import schema

from nni import ClassArgsValidator
from nni.common.hpo_utils import format_search_space, deformat_parameters
from nni.tuner import Tuner

class RandomTuner(Tuner):
    def __init__(self, seed=None):
        self.space = None
        self.rng = np.random.default_rng(seed)

    def update_search_space(self, space):
        self.space = format_search_space(space)

    def generate_parameters(self, *args, **kwargs):
        params = suggest(self.rng, self.space)
        return deformat_parameters(params, self.space)

    def receive_trial_result(self, *args, **kwargs):
        pass

class RandomClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        schema.Schema({schema.Optional('seed'): int}).validate(kwargs)

def suggest(rng, space):
    params = {}
    for key, spec in space.items():
        if spec.is_activated_in(params):
            params[key] = suggest_parameter(rng, spec)
    return params

def suggest_parameter(rng, spec):
    if spec.categorical:
        return rng.integers(spec.size)
    if spec.normal_distributed:
        return rng.normal(spec.mu, spec.sigma)
    else:
        return rng.uniform(spec.low, spec.high)
