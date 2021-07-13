from typing import List, Dict, Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer

from nni.algorithms.compression_v2.pytorch.base.pruner import Pruner
from nni.algorithms.compression_v2.pytorch.base.common import HookCollectorInfo, MetricsCalculator
from nni.algorithms.compression_v2.pytorch.common.data_collector import WeightDataCollector, WeightTrainerBasedDataCollector, ActivationTrainerBasedDataCollector
from nni.algorithms.compression_v2.pytorch.common.metrics_calculator import AbsMetricsCalculator, NormMetricsCalculator, DistMetricsCalculator, APoZRankMetricsCalculator, MeanRankMetricsCalculator
from nni.algorithms.compression_v2.pytorch.common.sparsity_allocator import get_sparsity_allocator, GRAPH_NEEDED_MODE


class LevelPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict], back_up: bool = True):
        self.mode = 'normal'
        super().__init__(model, config_list, back_up)

    def _reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = AbsMetricsCalculator()
        if self.sparsity_allocator is None:
            self.sparsity_allocator = get_sparsity_allocator(pruner=self, mode=self.mode)


class L1FilterPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict], back_up: bool = True,
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        self.mode = mode
        graph_needed = True if self.mode in GRAPH_NEEDED_MODE else False
        super().__init__(model, config_list, back_up, graph_needed=graph_needed, dummy_input=dummy_input)

    def _reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = NormMetricsCalculator(p=1, dim=0)
        if self.sparsity_allocator is None:
            self.sparsity_allocator = get_sparsity_allocator(pruner=self, mode=self.mode, dim=0)


class L2FilterPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict], back_up: bool = True,
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        self.mode = mode
        graph_needed = True if self.mode in GRAPH_NEEDED_MODE else False
        super().__init__(model, config_list, back_up, graph_needed=graph_needed, dummy_input=dummy_input)

    def _reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = NormMetricsCalculator(p=2, dim=0)
        if self.sparsity_allocator is None:
            self.sparsity_allocator = get_sparsity_allocator(pruner=self, mode=self.mode, dim=0)


class FPGMPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict], back_up: bool = True,
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        self.mode = mode
        graph_needed = True if self.mode in GRAPH_NEEDED_MODE else False
        super().__init__(model, config_list, back_up, graph_needed=graph_needed, dummy_input=dummy_input)

    def _reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = DistMetricsCalculator(p=2, dim=0)
        if self.sparsity_allocator is None:
            self.sparsity_allocator = get_sparsity_allocator(pruner=self, mode=self.mode, dim=0)


class SlimPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor],
                 training_epochs: int, scale: float = 0.0001, back_up: bool = True, max_sparsity_per_layer: float = 1):
        self.mode = 'global'
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
            self.metrics_calculator = AbsMetricsCalculator()
        if self.sparsity_allocator is None:
            self.sparsity_allocator = get_sparsity_allocator(pruner=self, mode=self.mode, dim=0, max_sparsity_per_layer=self.max_sparsity_per_layer)


class ActivationFilterPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor], training_batches: int, activation: str = 'relu',
                 back_up: bool = True, mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        self.mode = mode
        graph_needed = True if self.mode in GRAPH_NEEDED_MODE else False
        self.trainer = trainer
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_batches = training_batches
        self._activation = self._choose_activation(activation)
        super().__init__(model, config_list, back_up, graph_needed=graph_needed, dummy_input=dummy_input)

    def _choose_activation(self, activation: str = 'relu') -> Callable:
        if activation == 'relu':
            return nn.functional.relu
        elif activation == 'relu6':
            return nn.functional.relu6
        else:
            raise 'Unsupported activatoin {}'.format(activation)

    def _collector(self, buffer: List) -> Callable[[Module, Tensor, Tensor], None]:
        def collect_activation(_module: Module, _input: Tensor, output: Tensor):
            if len(buffer) < self.training_batches:
                buffer.append(self._activation(output.detach()))
        return collect_activation

    def _reset_tools(self):
        collector_info = HookCollectorInfo([layer_info for layer_info, _ in self._detect_modules_to_compress()], 'forward', self._collector)
        if self.data_collector is None:
            self.data_collector = ActivationTrainerBasedDataCollector(self, self.trainer, self.optimizer, self.criterion,
                                                                      1, collector_infos=[collector_info])
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = self._get_metrics_calculator()
        if self.sparsity_allocator is None:
            self.sparsity_allocator = get_sparsity_allocator(pruner=self, mode=self.mode, dim=0)

    def _get_metrics_calculator(self) -> MetricsCalculator:
        raise NotImplementedError()


class ActivationAPoZRankFilterPruner(ActivationFilterPruner):
    def _get_metrics_calculator(self) -> MetricsCalculator:
        return APoZRankMetricsCalculator(dim=0)


class ActivationMeanRankFilterPruner(ActivationFilterPruner):
    def _get_metrics_calculator(self) -> MetricsCalculator:
        return MeanRankMetricsCalculator(dim=0)
