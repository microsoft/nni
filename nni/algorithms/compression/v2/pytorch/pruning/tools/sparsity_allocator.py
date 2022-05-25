# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import itertools
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from nni.algorithms.compression.v2.pytorch.base import Pruner
from nni.compression.pytorch.utils.shape_dependency import ChannelDependency, GroupDependency

from .base import SparsityAllocator
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
                sub_metric_str = "metric[{}]".format(index_str)
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
        max_pruning_num = (global_metric != max_metric_value).sum().item()
        total_pruning_num = min(int(global_sparsity_rate * global_metric.numel()), max_pruning_num)
        global_threshold = torch.topk(global_metric.reshape(-1), total_pruning_num, largest=False)[0].max()

        # generate masks for each target
        for module_name, target_metric in metrics.items():
            masks[module_name] = {}
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            shrinked_mask = torch.gt(target_metric, global_threshold).type_as(target_metric)
            masks[module_name][target_name] = self._expand_mask(module_name, target_name, shrinked_mask)
        return masks


class Conv2dDependencyAwareAllocator(SparsityAllocator):
    """
    An allocator specific for Conv2d with dependency-aware.
    """

    def __init__(self, pruner: Pruner, dummy_input: Any):
        super().__init__(pruner, scalors=Scaling([1], padding_kernel=True))
        self.dummy_input = dummy_input

    def _get_dependency(self):
        graph = self.pruner.generate_graph(dummy_input=self.dummy_input)
        self.pruner._unwrap_model()
        self.channel_depen = ChannelDependency(model=self.pruner.bound_model, dummy_input=self.dummy_input, traced_model=graph.trace).dependency_sets
        self.group_depen = GroupDependency(model=self.pruner.bound_model, dummy_input=self.dummy_input, traced_model=graph.trace).dependency_sets
        self.pruner._wrap_model()

    def generate_sparsity(self, metrics: Dict) -> Dict[str, Dict[str, Tensor]]:
        self._get_dependency()
        masks = {}
        grouped_metrics = {}
        grouped_names = set()
        # combine metrics with channel dependence
        for idx, names in enumerate(self.channel_depen):
            grouped_metric = {name: metrics[name] for name in names if name in metrics}
            grouped_names.update(grouped_metric.keys())
            if self.continuous_mask:
                for name, metric in grouped_metric.items():
                    metric *= self._compress_mask(self.pruner.get_modules_wrapper()[name].weight_mask)  # type: ignore
            if len(grouped_metric) > 0:
                grouped_metrics[idx] = grouped_metric
        # ungrouped metrics stand alone as a group
        ungrouped_names = set(metrics.keys()).difference(grouped_names)
        for name in ungrouped_names:
            idx += 1  # type: ignore
            grouped_metrics[idx] = {name: metrics[name]}

        # generate masks
        for _, group_metric_dict in grouped_metrics.items():
            group_metric = self._group_metric_calculate(group_metric_dict)

            sparsities = {name: self.pruner.get_modules_wrapper()[name].config['total_sparsity'] for name in group_metric_dict.keys()}
            min_sparsity = min(sparsities.values())

            # generate group mask
            conv2d_groups, group_mask = [], []
            for name in group_metric_dict.keys():
                if name in self.group_depen:
                    conv2d_groups.append(self.group_depen[name])
                else:
                    # not in group_depen means not a Conv2d layer, in this case, assume the group number is 1
                    conv2d_groups.append(1)

            max_conv2d_group = np.lcm.reduce(conv2d_groups)
            pruned_per_conv2d_group = int(group_metric.numel() / max_conv2d_group * min_sparsity)
            conv2d_group_step = int(group_metric.numel() / max_conv2d_group)

            for gid in range(max_conv2d_group):
                _start = gid * conv2d_group_step
                _end = (gid + 1) * conv2d_group_step
                if pruned_per_conv2d_group > 0:
                    threshold = torch.topk(group_metric[_start: _end], pruned_per_conv2d_group, largest=False)[0].max()
                    conv2d_group_mask = torch.gt(group_metric[_start:_end], threshold).type_as(group_metric)
                else:
                    conv2d_group_mask = torch.ones(conv2d_group_step, device=group_metric.device)
                group_mask.append(conv2d_group_mask)
            group_mask = torch.cat(group_mask, dim=0)

            # generate final mask
            for name, metric in group_metric_dict.items():
                # We assume the metric value are all positive right now.
                metric = metric * group_mask
                pruned_num = int(sparsities[name] * len(metric))
                if pruned_num == 0:
                    threshold = metric.min() - 1
                else:
                    threshold = torch.topk(metric, pruned_num, largest=False)[0].max()
                mask = torch.gt(metric, threshold).type_as(metric)
                masks[name] = self._expand_mask(name, mask)
                if self.continuous_mask:
                    masks[name]['weight'] *= self.pruner.get_modules_wrapper()[name].weight_mask
        return masks

    def _group_metric_calculate(self, group_metrics: Union[Dict[str, Tensor], List[Tensor]]) -> Tensor:
        """
        Add all metric value in the same position in one group.
        """
        group_metrics = list(group_metrics.values()) if isinstance(group_metrics, dict) else group_metrics
        assert all(group_metrics[0].size() == group_metric.size() for group_metric in group_metrics), 'Metrics size do not match.'
        group_sum_metric = torch.zeros(group_metrics[0].size(), device=group_metrics[0].device)
        for group_metric in group_metrics:
            group_sum_metric += group_metric
        return group_sum_metric
