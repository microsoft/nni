# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import collections
import logging
from typing import Dict, Any, List
from ..graph import Model
from ..mutator import Mutator, Sampler

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

def dry_run_for_formatted_search_space(model: Model, mutators: List[Mutator]) -> Dict[Any, Dict[Any, Any]]:
    search_space = collections.OrderedDict()
    for mutator in mutators:
        recorded_candidates, model = mutator.dry_run(model)
        if len(recorded_candidates) == 1:
            search_space[mutator.label] = {'_type': 'choice', '_value': recorded_candidates[0]}
        else:
            for i, candidate in enumerate(recorded_candidates):
                search_space[f'{mutator.label}_{i}'] = {'_type': 'choice', '_value': candidate}
    return search_space

def get_targeted_model(base_model: Model, mutators: List[Mutator], sample: dict) -> Model:
    sampler = _FixedSampler(sample)
    model = base_model
    for mutator in mutators:
        model = mutator.bind_sampler(sampler).apply(model)
    return model


def filter_model(model_filter, ir_model):
    if model_filter is not None:
        _logger.debug(f'Check if model satisfies constraints.')
        if model_filter(ir_model):
            _logger.debug(f'Model satisfied. Submit the model.')
            return True
        else:
            _logger.debug(f'Model unsatisfied. Discard the model.')
            return False
    else:
        return True
