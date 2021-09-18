# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
    for spec in space.values():
        if not spec.is_activated(params):
            continue

        if spec.categorical:
            params[spec.key] = rng.integers(spec.size)
            continue

        if spec.normal_distributed:
            if spec.log_distributed:
                x = rng.lognormal(spec.mu, spec.sigma)
            else:
                x = rng.normal(spec.mu, spec.sigma)
        else:
            if spec.log_distributed:
                x = np.exp(rng.uniform(np.log(spec.low), np.log(spec.high)))
            else:
                x = rng.uniform(spec.low, spec.high)
        if spec.q is not None:
            x = np.round(x / spec.q) * spec.q
        params[spec.key] = x
    return params
