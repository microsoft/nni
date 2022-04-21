# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import List, Any

from ..graph import Model
from ..mutator import Mutator


class BaseStrategy(abc.ABC):

    @abc.abstractmethod
    def run(self, base_model: Model, applied_mutators: List[Mutator]) -> None:
        pass

    def export_top_models(self, top_k: int) -> List[Any]:
        raise NotImplementedError('"export_top_models" is not implemented.')
