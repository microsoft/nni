# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import List

from ..graph import Model
from ..mutator import Mutator
from .utils import dry_run_for_formated_search_space


class BaseStrategy(abc.ABC):

    @classmethod
    def get_model_space(self, base_model: Model, applied_mutators: List[Mutator]) -> None:
        """
        NOTE: nested search space is not supported (currently also not supported on webui),
        currently, only support type `choice`.
        """
        return dry_run_for_formated_search_space(base_model, applied_mutators)

    @abc.abstractmethod
    def run(self, base_model: Model, applied_mutators: List[Mutator]) -> None:
        pass
