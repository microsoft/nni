# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import List

from ..graph import Model
from ..mutator import Mutator


class BaseStrategy(abc.ABC):

    @abc.abstractmethod
    def run(self, base_model: Model, applied_mutators: List[Mutator]) -> None:
        pass
