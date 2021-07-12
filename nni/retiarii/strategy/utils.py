# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import collections
import logging
from typing import Dict, Any, List
from ..graph import Model
from ..mutator import Mutator, Sampler

from nn_meter import get_default_config, load_latency_predictors  # pylint: disable=import-error

_logger = logging.getLogger(__name__)


class _FixedSampler(Sampler):
    def __init__(self, sample):
        self.sample = sample

    def choice(self, candidates, mutator, model, index):
        return self.sample[(mutator, index)]


def dry_run_for_search_space(model: Model, mutators: List[Mutator]) -> Dict[Any, List[Any]]:
    search_space = collections.OrderedDict()
    for mutator in mutators:
        recorded_candidates, model = mutator.dry_run(model)
        for i, candidates in enumerate(recorded_candidates):
            search_space[(mutator, i)] = candidates
    return search_space


def get_targeted_model(base_model: Model, mutators: List[Mutator], sample: dict) -> Model:
    sampler = _FixedSampler(sample)
    model = base_model
    for mutator in mutators:
        model = mutator.bind_sampler(sampler).apply(model)
    return model


class LatencyFilter:
    def __init__(self, threshold, config=None, hardware='', reverse=False):
        """
        Filter the models according to predcted latency.

        Parameters
        ----------
        threshold: `float`
            the threshold of latency
        config, hardware:
            determine the targeted device
        reverse: `bool`
            if reverse is `False`, then the model returns `True` when `latency < threshold`,
            else otherwisse
        """
        default_config, default_hardware = get_default_config()
        if config is None:
            config = default_config
        if not hardware:
            hardware = default_hardware

        self.predictors = load_latency_predictors(config, hardware)
        self.threshold = threshold

    def __call__(self, ir_model):
        latency = self.predictors.predict(ir_model, 'nni')
        return latency < self.threshold


def filter_model(filter, ir_model):
    if filter is not None:
        _logger.debug(f'Check if model satisfies constraints.')
        if filter(ir_model):
            _logger.debug(f'Model satisfied. Submit the model.')
            return True
        else:
            _logger.debug(f'Model unsatisfied. Discard the model.')
            return False
    else:
        return True
