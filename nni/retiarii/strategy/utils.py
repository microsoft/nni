# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import collections
from typing import Dict, Any, List
from ..graph import Model
from ..mutator import Mutator, Sampler


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
