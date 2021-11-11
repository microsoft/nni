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
        """
        NOTE: nested search space is not supported (currently also not supported on webui),
        currently, only support type `choice`.
        """
        search_space = {}
        auto_label_seq = 0
        new_model = base_model
        for mutator in applied_mutators:
            recorded_candidates, new_model = mutator.dry_run(new_model)
            if mutator.label is not None:
                label = mutator.label
            else:
                label = f'auto_label_{auto_label_seq}'
                auto_label_seq += 1
            if len(recorded_candidates) == 1:
                search_space[label] = {'_type': 'choice', '_value': recorded_candidates[0]}
            else:
                for i, candidate in enumerate(recorded_candidates):
                    search_space[f'{label}_{i}'] = {'_type': 'choice', '_value': candidate}

        report_search_space(search_space)

    @abc.abstractmethod
    def run(self, base_model: Model, applied_mutators: List[Mutator]) -> None:
        pass
