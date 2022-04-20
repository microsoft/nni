# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Dict, List

from torch import Tensor

from .base import DataCollector, TrainerBasedDataCollector

_logger = logging.getLogger(__name__)

__all__ = ['WeightDataCollector', 'WeightTrainerBasedDataCollector', 'SingleHookTrainerBasedDataCollector']


class WeightDataCollector(DataCollector):
    """
    Collect all wrapper weights.
    """

    def reset(self):
        pass

    def collect(self) -> Dict[str, Tensor]:
        data = {}
        for _, wrapper in self.compressor.get_modules_wrapper().items():
            data[wrapper.name] = wrapper.module.weight.data
        return data


class WeightTrainerBasedDataCollector(TrainerBasedDataCollector):
    """
    Collect all wrapper weights after training or inference.
    """

    def collect(self) -> Dict[str, Tensor]:
        for _ in range(self.training_epochs):
            self.trainer(self.compressor.bound_model, self.optimizer, self.criterion)

        data = {}
        for _, wrapper in self.compressor.get_modules_wrapper().items():
            data[wrapper.name] = wrapper.module.weight.data
        return data


class SingleHookTrainerBasedDataCollector(TrainerBasedDataCollector):
    """
    Add hooks and collect data during training or inference.
    Single means each wrapper only has one hook to collect data.
    """

    def collect(self) -> Dict[str, List[Tensor]]:
        for _ in range(self.training_epochs):
            self.trainer(self.compressor.bound_model, self.optimizer, self.criterion)

        data = {}
        [data.update(buffer_dict) for _, buffer_dict in self._hook_buffer.items()]
        return data
