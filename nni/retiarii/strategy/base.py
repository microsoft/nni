# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import List

from .. import report_search_space
from ..graph import Model
from ..mutator import Mutator

from .utils import dry_run_for_search_space


class BaseStrategy(abc.ABC):

    @abc.classmethod
    def report_model_space(self, base_model: Model, applied_mutators: List[Mutator]) -> None:
        search_space = dry_run_for_search_space(base_model, applied_mutators)
        report_search_space(search_space)

    @abc.abstractmethod
    def run(self, base_model: Model, applied_mutators: List[Mutator]) -> None:
        pass
