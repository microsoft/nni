# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Dict, List

from torch import Tensor

from .base import DataCollector, EvaluatorBasedDataCollector

_logger = logging.getLogger(__name__)

__all__ = ['TargetDataCollector', 'EvaluatorBasedTargetDataCollector', 'EvaluatorBasedHookDataCollector']


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
    Single means each wrapper only has one hook to collect data.
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
