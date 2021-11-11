# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import List

from .. import report_search_space
from ..graph import Model
from ..mutator import Mutator


class BaseStrategy(abc.ABC):

    @classmethod
    def report_model_space(self, base_model: Model, applied_mutators: List[Mutator]) -> None:
        sample_space = []
        new_model = base_model
        for mutator in applied_mutators:
            recorded_candidates, new_model = mutator.dry_run(new_model)
            sample_space.extend(recorded_candidates)
        search_space = {}
        for i, each in enumerate(sample_space):
            search_space[str(i)] = {'_type': 'choice', '_value': each}
        report_search_space(search_space)

    @abc.abstractmethod
    def run(self, base_model: Model, applied_mutators: List[Mutator]) -> None:
        pass
