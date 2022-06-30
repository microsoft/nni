# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch import Tensor

from nni.common.graph_utils import TorchModuleGraph
from nni.compression.pytorch.utils.shape_dependency import ChannelDependency, GroupDependency

from .base import SparsityAllocator
from ...base import Pruner
from ...utils import Scaling


class NormalSparsityAllocator(SparsityAllocator):
    """
    This allocator directly masks the locations of each pruning target with lower metric values.
    """

    def common_target_masks_generation(self, metrics: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        # TODO: Support more target type in wrapper & config list refactor
        target_name = 'weight'
        for module_name, target_metric in metrics.items():
            masks[module_name] = {}
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            sparsity_rate = wrapper.config['total_sparsity']
            prune_num = int(sparsity_rate * target_metric.numel())
            if prune_num != 0:
                threshold = torch.topk(target_metric.reshape(-1), prune_num, largest=False)[0].max()
                shrinked_mask = torch.gt(target_metric, threshold).type_as(target_metric)
            else:
                # target_metric should have the same size as shrinked_mask
                shrinked_mask = torch.ones_like(target_metric)
            masks[module_name][target_name] = self._expand_mask(module_name, target_name, shrinked_mask)
        return masks


class BankSparsityAllocator(SparsityAllocator):
    """
    In bank pruner, all values in weight are divided into different sub blocks each shape
    aligned with balance_gran. Each sub block has the same sparsity which equal to the overall sparsity.
    This allocator pruned the weight in the granularity of block. 
    """

    def __init__(self, pruner: Pruner, balance_gran: list):
        super().__init__(pruner)
        self.balance_gran = balance_gran
        for gran in self.balance_gran:
            assert isinstance(gran, int) and gran > 0, 'All values in list balance_gran \
                should be type int and bigger than zero'

    def common_target_masks_generation(self, metrics: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        # TODO: Support more target type in wrapper & config list refactor
        target_name = 'weight'
        for module_name, target_metric in metrics.items():
            masks[module_name] = {}
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            sparsity_rate = wrapper.config['total_sparsity']

            n_dim = len(target_metric.shape)
            assert n_dim >= len(self.balance_gran), 'Dimension of balance_gran should be smaller than metric'
            # make up for balance_gran
            balance_gran = [1] * (n_dim - len(self.balance_gran)) + self.balance_gran
            for i, j in zip(target_metric.shape, balance_gran):
                assert i % j == 0, 'Length of {} {} is not aligned with balance granularity'.format(module_name, target_name)

            # FIXME: The following code need refactor, do it after scaling refactor is done.
            shrinked_mask = torch.ones(target_metric.shape).type_as(target_metric)
            loop_iters = [range(int(i / j)) for i, j in zip(target_metric.shape, balance_gran)]
            for iter_params in itertools.product(*loop_iters):
                index_str_list = [f"{iter_param * gran}:{(iter_param+1) * gran}"\
                     for iter_param, gran in zip(iter_params, balance_gran)]
                index_str = ",".join(index_str_list)
                sub_metric_str = "target_metric[{}]".format(index_str)
                sub_mask_str =  "shrinked_mask[{}] = mask_bank".format(index_str)
                metric_bank: Tensor = eval(sub_metric_str)
                prune_num = int(sparsity_rate * metric_bank.numel())
                # mask_bank will be used in exec(sub_mask_str)
                if prune_num != 0:
                    threshold = torch.topk(metric_bank.reshape(-1), prune_num, largest=False)[0].max()
                    mask_bank = torch.gt(metric_bank, threshold).type_as(metric_bank)
                else:
                    mask_bank = torch.ones_like(metric_bank)
                exec(sub_mask_str)
            masks[module_name][target_name] = self._expand_mask(module_name, target_name, shrinked_mask)
        return masks


class GlobalSparsityAllocator(SparsityAllocator):
    """
    This allocator sorts all metrics as a whole, mask the locations of pruning target with lower metric value.
    """

    def common_target_masks_generation(self, metrics: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        if not metrics:
            return masks
        # TODO: support more target type in wrapper & config list refactor
        target_name = 'weight'

        # validate all wrapper setting the same sparsity
        # TODO: move validation logic to pruner
        global_sparsity_rate = self.pruner.get_modules_wrapper()[list(metrics.keys())[0]].config['total_sparsity']
        for module_name, target_metric in metrics.items():
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            assert global_sparsity_rate == wrapper.config['total_sparsity']

        # find the largest metric value among all metrics
        max_metric_value = list(metrics.values())[0].max()
        for module_name, target_metric in metrics.items():
            max_metric_value = max_metric_value if max_metric_value >= target_metric.max() else target_metric.max()

        # prevent each module from being over-pruned, prevent ratio is 'max_sparsity_per_layer'
        for module_name, target_metric in metrics.items():
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            max_sparsity = wrapper.config.get('max_sparsity_per_layer', {}).get(module_name, 0.99)
            assert 0 <= max_sparsity <= 1
            old_target_mask: Tensor = getattr(wrapper, f'{target_name}_mask')
            expand_times = old_target_mask.numel() // target_metric.numel()
            max_pruning_numel = int(max_sparsity * target_metric.numel()) * expand_times
            threshold = torch.topk(target_metric.reshape(-1), max_pruning_numel, largest=False)[0].max()
            metrics[module_name] = torch.where(target_metric <= threshold, target_metric, max_metric_value)

        # build the global_matric & calculate global threshold
        metric_list = []
        for module_name, target_metric in metrics.items():
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            old_target_mask: Tensor = getattr(wrapper, f'{target_name}_mask')
            expand_times = old_target_mask.numel() // target_metric.numel()
            metric_list.append(target_metric.reshape(-1).unsqueeze(0).expand(expand_times, -1).reshape(-1))
        global_metric = torch.cat(metric_list)
        max_pruning_num = int((global_metric != max_metric_value).sum().item())
        total_pruning_num = min(int(global_sparsity_rate * global_metric.numel()), max_pruning_num)
        global_threshold = torch.topk(global_metric.reshape(-1), total_pruning_num, largest=False)[0].max()

        # generate masks for each target
        for module_name, target_metric in metrics.items():
            masks[module_name] = {}
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            shrinked_mask = torch.gt(target_metric, global_threshold).type_as(target_metric)
            masks[module_name][target_name] = self._expand_mask(module_name, target_name, shrinked_mask)
        return masks


class DependencyAwareAllocator(NormalSparsityAllocator):
    """
    An specific allocator for Conv2d & Linear module with dependency-aware.
    It will generate a public mask for the modules that have dependencies,
    then generate the part of the non-public mask for each module.
    For other module types, the way to generate the mask is the same as `NormalSparsityAllocator`.
    """

    def __init__(self, pruner: Pruner, dummy_input: Any, scalers: Dict[str, Dict[str, Scaling]] | Scaling | None = None):
        # Scaling(kernel_size=[1], kernel_padding_mode='back') means output channel pruning.
        scalers = scalers if scalers else Scaling(kernel_size=[1], kernel_padding_mode='back')
        super().__init__(pruner, scalers=scalers)
        self.channel_dependency, self.group_dependency = self._get_dependency(dummy_input)

    def _get_dependency(self, dummy_input: Any):
        # get the channel dependency and group dependency
        # channel dependency format: [[module_name1, module_name2], [module_name3], ...]
        # group dependency format: {module_name: group_num}
        self.pruner._unwrap_model()
        graph = TorchModuleGraph(model=self.pruner.bound_model, dummy_input=dummy_input)
        channel_dependency = ChannelDependency(model=self.pruner.bound_model, dummy_input=dummy_input, traced_model=graph.trace).dependency_sets
        group_dependency = GroupDependency(model=self.pruner.bound_model, dummy_input=dummy_input, traced_model=graph.trace).dependency_sets
        self.pruner._wrap_model()
        return channel_dependency, group_dependency

    def _metric_fuse(self, metrics: Union[Dict[str, Tensor], List[Tensor]]) -> Tensor:
        # Sum all metric value in the same position.
        metrics = list(metrics.values()) if isinstance(metrics, dict) else metrics
        assert all(metrics[0].size() == metric.size() for metric in metrics), 'Metrics size do not match.'
        fused_metric = torch.zeros_like(metrics[0])
        for metric in metrics:
            fused_metric += metric
        return fused_metric

    def common_target_masks_generation(self, metrics: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        # generate public part for modules that have dependencies
        for module_names in self.channel_dependency:
            sub_metrics = {module_name: metrics[module_name] for module_name in module_names if module_name in metrics}
            if not sub_metrics:
                continue
            fused_metric = self._metric_fuse(sub_metrics)

            sparsity_rates = {module_name: self.pruner.get_modules_wrapper()[module_name].config['total_sparsity'] for module_name in sub_metrics.keys()}
            min_sparsity_rate = min(sparsity_rates.values())

            group_nums = [self.group_dependency.get(module_name, 1) for module_name in sub_metrics.keys()]
            max_group_nums = int(np.lcm.reduce(group_nums))
            pruned_numel_per_group = int(fused_metric.numel() // max_group_nums * min_sparsity_rate)
            group_step = fused_metric.shape[0] // max_group_nums

            # get the public part of the mask of the module with dependencies
            sub_masks = []
            for gid in range(max_group_nums):
                _start = gid * group_step
                _end = (gid + 1) * group_step
                if pruned_numel_per_group > 0:
                    threshold = torch.topk(fused_metric[_start: _end].reshape(-1), pruned_numel_per_group, largest=False)[0].max()
                    sub_mask = torch.gt(fused_metric[_start:_end], threshold).type_as(fused_metric)
                else:
                    sub_mask = torch.ones_like(fused_metric[_start:_end])
                sub_masks.append(sub_mask)
            dependency_mask = torch.cat(sub_masks, dim=0)

            # change the metric value corresponding to the public mask part to the minimum value
            for module_name, target_metric in sub_metrics.items():
                min_value = target_metric.min()
                metrics[module_name] = torch.where(dependency_mask!=0, target_metric, min_value)

        return super().common_target_masks_generation(metrics)
