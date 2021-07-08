from typing import List, Dict, Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from nni.algorithms.compression_v2.pytorch.base.pruner import Pruner
from nni.algorithms.compression_v2.pytorch.common.data_collector import WeightDataCollector, WeightTrainerBasedDataCollector
from nni.algorithms.compression_v2.pytorch.common.metrics_calculator import NaiveMetricsCalculator, NormMetricsCalculator, DistMetricsCalculator
from nni.algorithms.compression_v2.pytorch.common.sparsity_allocator import NormalSparsityAllocator, GlobalSparsityAllocator


class LevelPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict], back_up: bool = True):
        super().__init__(model, config_list, back_up)

    def _reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = NaiveMetricsCalculator()
        if self.sparsity_allocator is None:
            self.sparsity_allocator = NormalSparsityAllocator(self)


class L1FilterPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict], back_up: bool = True):
        super().__init__(model, config_list, back_up)

    def _reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = NormMetricsCalculator(p=1, dim=0)
        if self.sparsity_allocator is None:
            self.sparsity_allocator = NormalSparsityAllocator(self, dim=0)


class L2FilterPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict], back_up: bool = True):
        super().__init__(model, config_list, back_up)

    def _reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = NormMetricsCalculator(p=2, dim=0)
        if self.sparsity_allocator is None:
            self.sparsity_allocator = NormalSparsityAllocator(self, dim=0)


class FPGMPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict], back_up: bool = True):
        super().__init__(model, config_list, back_up)

    def _reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = DistMetricsCalculator(p=2, dim=0)
        if self.sparsity_allocator is None:
            self.sparsity_allocator = NormalSparsityAllocator(self, dim=0)


class SlimPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor],
                 training_epochs: int, scale: float = 0.0001, back_up: bool = True, max_sparsity_per_layer: float = 1):
        assert 0 < max_sparsity_per_layer <= 1, 'max_sparsity_per_layer must in range (0, 1].'
        self.max_sparsity_per_layer = max_sparsity_per_layer
        self.trainer = trainer
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_epochs = training_epochs
        self._scale = scale
        super().__init__(model, config_list, back_up)

    def criterion_patch(self, criterion: Callable[[Tensor, Tensor], Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
        def patched_criterion(input_tensor: Tensor, target: Tensor):
            sum_l1 = 0
            for _, wrapper in self._get_modules_wrapper().items():
                sum_l1 += torch.norm(wrapper.module.weight.data, p=1)
            return criterion(input_tensor, target) + self._scale * sum_l1
        return patched_criterion

    def _reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightTrainerBasedDataCollector(self, self.trainer, self.optimizer, self.criterion,
                                                                  self.training_epochs, criterion_patch=self.criterion_patch)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = NaiveMetricsCalculator()
        if self.sparsity_allocator is None:
            self.sparsity_allocator = GlobalSparsityAllocator(self, dim=0, max_sparsity_per_layer=self.max_sparsity_per_layer)
