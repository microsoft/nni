# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from functools import reduce
from typing import Any, Dict

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

    def common_target_masks_generation(self, metrics: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        # TODO: Support more target type in wrapper & config list refactor
        for module_name, targets_metric in metrics.items():
            masks[module_name] = {}
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            for target_name, target_metric in targets_metric.items():
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

    def common_target_masks_generation(self, metrics: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        # TODO: Support more target type in wrapper & config list refactor
        for module_name, targets_metric in metrics.items():
            masks[module_name] = {}
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            for target_name, target_metric in targets_metric.items():
                sparsity_rate = wrapper.config['total_sparsity']

                n_dim = len(target_metric.shape)
                assert n_dim >= len(self.balance_gran), 'Dimension of balance_gran should be smaller than metric'
                # make up for balance_gran
                balance_gran = [1] * (n_dim - len(self.balance_gran)) + self.balance_gran
                balance_numel = reduce(lambda x, y: x * y, balance_gran)

                reshape_size_split = []
                reshape_size_balance = []
                for i, j in zip(target_metric.shape, balance_gran):
                    assert i % j == 0, 'Length of {} {} is not aligned with balance granularity'.format(module_name, target_name)
                    reshape_size_split.extend([i // j, j])
                    reshape_size_balance.append(i // j)
                reshape_size_balance.append(balance_numel)

                permute_dims_balance = [_ * 2 for _ in range(n_dim)] + [_ * 2 + 1 for _ in range(n_dim)]
                _target_metric = target_metric.reshape(reshape_size_split).permute(permute_dims_balance)
                reshape_size_split_p = _target_metric.shape
                balance_metric = _target_metric.reshape(reshape_size_balance)

                kept_num = balance_numel - int(sparsity_rate * balance_numel)
                kept_indices = torch.topk(balance_metric, kept_num).indices
                shrinked_mask = torch.zeros_like(balance_metric).scatter(-1, kept_indices, 1.0).reshape(reshape_size_split_p)

                permute_dims_split = []
                for i in range(n_dim):
                    permute_dims_split.extend([i, i + n_dim])
                shrinked_mask = shrinked_mask.permute(permute_dims_split).reshape_as(target_metric)
                masks[module_name][target_name] = self._expand_mask(module_name, target_name, shrinked_mask)
        return masks


class GlobalSparsityAllocator(SparsityAllocator):
    """
    This allocator sorts all metrics as a whole, mask the locations of pruning target with lower metric value.
    By default, this allocator will prevent each module from being over-pruned with upper sparsity 0.99.
    """

    def common_target_masks_generation(self, metrics: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        if not metrics:
            return masks
        # TODO: support more target type in wrapper & config list refactor
        # validate all wrapper setting have the same sparsity
        # TODO: move validation logic to pruner
        global_sparsity_rate = self.pruner.get_modules_wrapper()[list(metrics.keys())[0]].config['total_sparsity']
        for module_name in metrics.keys():
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            assert global_sparsity_rate == wrapper.config['total_sparsity']

        # find the largest metric value among all metrics
        max_metric_value = list(list(metrics.values())[0].values())[0].max()
        for targets_metric in metrics.values():
            for target_metric in targets_metric.values():
                max_metric_value = max_metric_value if max_metric_value >= target_metric.max() else target_metric.max()

        # prevent each module from being over-pruned, prevent ratio is 'max_sparsity_per_layer'
        for module_name, targets_metric in metrics.items():
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            for target_name, target_metric in targets_metric.items():
                max_sparsity = wrapper.config.get('max_sparsity_per_layer', {}).get(module_name, 0.99)
                assert 0 <= max_sparsity <= 1
                old_target_mask: Tensor = getattr(wrapper, f'{target_name}_mask')
                expand_times = old_target_mask.numel() // target_metric.numel()
                max_pruning_numel = int(max_sparsity * target_metric.numel()) * expand_times
                threshold = torch.topk(target_metric.reshape(-1), max_pruning_numel, largest=False)[0].max()
                metrics[module_name][target_name] = torch.where(target_metric <= threshold, target_metric, max_metric_value)

        # build the global_matric & calculate global threshold
        metric_list = []
        for module_name, targets_metric in metrics.items():
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            for target_name, target_metric in targets_metric.items():
                old_target_mask: Tensor = getattr(wrapper, f'{target_name}_mask')
                expand_times = old_target_mask.numel() // target_metric.numel()
                metric_list.append(target_metric.reshape(-1).repeat_interleave(expand_times))
        global_metric = torch.cat(metric_list)
        max_pruning_num = int((global_metric != max_metric_value).sum().item())
        total_pruning_num = min(int(global_sparsity_rate * global_metric.numel()), max_pruning_num)
        global_threshold = torch.topk(global_metric.reshape(-1), total_pruning_num, largest=False)[0].max()

        # generate masks for each target
        for module_name, targets_metric in metrics.items():
            masks[module_name] = {}
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            for target_name, target_metric in targets_metric.items():
                wrapper = self.pruner.get_modules_wrapper()[module_name]
                shrinked_mask = torch.gt(target_metric, global_threshold).type_as(target_metric)
                masks[module_name][target_name] = self._expand_mask(module_name, target_name, shrinked_mask)
        return masks


# TODO: This allocator will trace the model, means the model will be inference during initialization,
# sometime we may not aware of this inference and it may lead to some error.
class DependencyAwareAllocator(SparsityAllocator):
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
        channel_dependency = ChannelDependency(model=self.pruner.bound_model, dummy_input=dummy_input,
                                               traced_model=graph.trace).dependency_sets
        group_dependency = GroupDependency(model=self.pruner.bound_model, dummy_input=dummy_input,
                                           traced_model=graph.trace).dependency_sets
        self.pruner._wrap_model()
        return channel_dependency, group_dependency

    def _metric_fuse(self, metrics: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        # Sum all metric value in the same position.
        fused_metrics = {}
        for targets_metric in metrics.values():
            for target_name, target_metric in targets_metric.items():
                if target_name in fused_metrics:
                    fused_metrics[target_name] += target_metric
                else:
                    fused_metrics[target_name] = target_metric
        return fused_metrics

    def common_target_masks_generation(self, metrics: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        # placeholder, here we need more discussion about dependence sparsity, Plan A or Plan B.
        masks = {}
        # generate public part for modules that have dependencies
        for module_names in self.channel_dependency:
            sub_metrics = {module_name: metrics[module_name] for module_name in module_names if module_name in metrics}
            if not sub_metrics:
                continue
            fused_metrics = self._metric_fuse(sub_metrics)

            for target_name, fused_metric in fused_metrics.items():
                sparsity_rates = {module_name: self.pruner.get_modules_wrapper()[module_name].config['total_sparsity'] \
                                  for module_name in sub_metrics.keys()}
                min_sparsity_rate = min(sparsity_rates.values())

                group_nums = [self.group_dependency.get(module_name, 1) for module_name in sub_metrics.keys()]
                max_group_nums = int(np.lcm.reduce(group_nums))
                pruned_numel_per_group = int(fused_metric.numel() // max_group_nums * min_sparsity_rate)
                group_step = fused_metric.shape[0] // max_group_nums

                # get the public part of the mask of the module with dependencies
                dependency_mask = torch.ones_like(fused_metric)
                for gid in range(max_group_nums):
                    _start = gid * group_step
                    _end = (gid + 1) * group_step
                    if pruned_numel_per_group > 0:
                        threshold = torch.topk(fused_metric[_start: _end].reshape(-1), pruned_numel_per_group, largest=False)[0].max()
                        dependency_mask[_start: _end] = torch.gt(fused_metric[_start:_end], threshold).type_as(fused_metric)

                # change the metric value corresponding to the public mask part to the minimum value
                for module_name, targets_metric in sub_metrics.items():
                    if target_name in targets_metric:
                        # Following is Plan A, generate the dependency mask first, and then fill in the sparsity,
                        # the final mask is group unbalanced. - 1 ensure the denpendency metric is the minimum, and will be masked first.
                        # min_value = targets_metric[target_name].min() - 1
                        # metrics[module_name][target_name] = torch.where(dependency_mask!=0, targets_metric[target_name], min_value)

                        # Following is Plan B, just generate the dependency mask, the final mask is group balanced.
                        masks.setdefault(module_name, {})
                        masks[module_name][target_name] = self._expand_mask(module_name, target_name, dependency_mask)

        # generate masks for layers without dependencies
        for module_name, targets_metric in metrics.items():
            masks.setdefault(module_name, {})
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            for target_name, target_metric in targets_metric.items():
                if target_name in masks[module_name]:
                    continue
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
