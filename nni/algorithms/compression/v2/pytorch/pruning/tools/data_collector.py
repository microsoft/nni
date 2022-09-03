# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Dict, List

from torch import Tensor

from .base import DataCollector, EvaluatorBasedDataCollector
from .base import TrainerBasedDataCollector

_logger = logging.getLogger(__name__)

__all__ = ['TargetDataCollector', 'EvaluatorBasedTargetDataCollector', 'EvaluatorBasedHookDataCollector',
           'WeightDataCollector', 'WeightTrainerBasedDataCollector', 'SingleHookTrainerBasedDataCollector']  # TODO: remove in nni v3.0.


# TODO: remove in nni v3.0.
class WeightDataCollector(DataCollector):
    """
    Collect all wrapper weights.
    """

    def reset(self):
        pass

    def collect(self) -> Dict[str, Dict[str, Tensor]]:
        data = {}
        target_name = 'weight'
        for module_name, wrapper in self.compressor.get_modules_wrapper().items():
            target: Tensor = getattr(wrapper, target_name)
            data[module_name] = {target_name: target.data.clone()}
        return data


# TODO: remove in nni v3.0.
class WeightTrainerBasedDataCollector(TrainerBasedDataCollector):
    """
    Collect all wrapper weights after training or inference.
    """

    def collect(self) -> Dict[str, Dict[str, Tensor]]:
        assert self.compressor.bound_model is not None
        for _ in range(self.training_epochs):
            self.trainer(self.compressor.bound_model, self.optimizer, self.criterion)

        data = {}
        target_name = 'weight'
        for module_name, wrapper in self.compressor.get_modules_wrapper().items():
            target: Tensor = getattr(wrapper, target_name)
            data[module_name] = {target_name: target.data.clone()}
        return data


# TODO: remove in nni v3.0.
class SingleHookTrainerBasedDataCollector(TrainerBasedDataCollector):
    """
    Add hooks and collect data during training or inference.
    Single means each wrapper only has one hook to collect data.
    """

    def collect(self) -> Dict[str, Dict[str, List[Tensor]]]:
        assert self.compressor.bound_model is not None
        for _ in range(self.training_epochs):
            self.trainer(self.compressor.bound_model, self.optimizer, self.criterion)

        data = {}
        target_name = 'weight'
        for _, buffer_dict in self._hook_buffer.items():
            for module_name, target_data in buffer_dict.items():
                data[module_name] = {target_name: target_data}
        return data


class TargetDataCollector(DataCollector):
    """
    Collect all wrapper targets.
    """

    def reset(self):
        # No need to reset anything in this data collector.
        pass

    def collect(self) -> Dict[str, Dict[str, Tensor]]:
        data = {}
        target_name = 'weight'
        for module_name, wrapper in self.compressor.get_modules_wrapper().items():
            target: Tensor = getattr(wrapper, target_name)
            data[module_name] = {target_name: target.data.clone()}
        return data


class EvaluatorBasedTargetDataCollector(EvaluatorBasedDataCollector):
    """
    Collect all wrapper pruning target after training or inference.
    """

    def collect(self) -> Dict[str, Dict[str, Tensor]]:
        assert self.compressor.bound_model is not None
        self.evaluator.train(max_steps=self.max_steps, max_epochs=self.max_epochs)

        data = {}
        target_name = 'weight'
        for module_name, wrapper in self.compressor.get_modules_wrapper().items():
            target: Tensor = getattr(wrapper, target_name)
            data[module_name] = {target_name: target.data.clone()}
        return data


class EvaluatorBasedHookDataCollector(EvaluatorBasedDataCollector):
    """
    Add hooks and collect data during training or inference.
    NOTE: Only support one target has one hook right now.
    """

    def collect(self) -> Dict[str, Dict[str, List]]:
        assert self.compressor.bound_model is not None
        self.evaluator.train(max_steps=self.max_steps, max_epochs=self.max_epochs)

        data = {}
        for module_name, hooks in self._hooks.items():
            data[module_name] = {}
            for target_name, hook in hooks.items():
                data[module_name][target_name] = hook.buffer
        return data
