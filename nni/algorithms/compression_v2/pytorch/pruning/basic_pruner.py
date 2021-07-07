from typing import List, Dict

from torch.nn import Module

from nni.algorithms.compression_v2.pytorch.base.pruner import Pruner
from nni.algorithms.compression_v2.pytorch.common.data_collector import WeightDataCollector
from nni.algorithms.compression_v2.pytorch.common.metrics_calculator import NaiveMetricsCalculator, NormMetricsCalculator, DistMetricsCalculator
from nni.algorithms.compression_v2.pytorch.common.sparsity_allocator import NormalSparsityAllocator


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
