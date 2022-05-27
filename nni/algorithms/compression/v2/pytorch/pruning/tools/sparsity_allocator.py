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


class NormalSparsityAllocator(SparsityAllocator):
    """
    This allocator simply pruned the weight with smaller metrics in layer level.
    """
    def generate_sparsity(self, metrics: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        for name, wrapper in self.pruner.get_modules_wrapper().items():
            sparsity_rate = wrapper.config['total_sparsity']

            assert name in metrics, 'Metric of {} is not calculated.'.format(name)

            # We assume the metric value are all positive right now.
            metric = metrics[name]
            if self.continuous_mask:
                metric *= self._compress_mask(wrapper.weight_mask)  # type: ignore
            prune_num = int(sparsity_rate * metric.numel())
            if prune_num == 0:
                threshold = metric.min() - 1
            else:
                threshold = torch.topk(metric.view(-1), prune_num, largest=False)[0].max()
            mask = torch.gt(metric, threshold).type_as(metric)
            masks[name] = self._expand_mask(name, mask)
            if self.continuous_mask:
                masks[name]['weight'] *= wrapper.weight_mask
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

    def generate_sparsity(self, metrics: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        for name, wrapper in self.pruner.get_modules_wrapper().items():
            sparsity_rate = wrapper.config['total_sparsity']

            assert name in metrics, 'Metric of {} is not calculated.'.format(name)

            # We assume the metric value are all positive right now.
            metric = metrics[name]
            if self.continuous_mask:
                metric *= self._compress_mask(wrapper.weight_mask)  # type: ignore
            n_dim = len(metric.shape)
            assert n_dim >= len(self.balance_gran), 'Dimension of balance_gran should be smaller than metric'
            # make up for balance_gran
            balance_gran = [1] * (n_dim - len(self.balance_gran)) + self.balance_gran
            for i, j in zip(metric.shape, balance_gran):
                assert i % j == 0, 'Length of {} weight is not aligned with balance granularity'.format(name)

            mask = torch.zeros(metric.shape).type_as(metric)
            loop_iters = [range(int(i / j)) for i, j in zip(metric.shape, balance_gran)]
            for iter_params in itertools.product(*loop_iters):
                index_str_list = [f"{iter_param * gran}:{(iter_param+1) * gran}"\
                     for iter_param, gran in zip(iter_params, balance_gran)]
                index_str = ",".join(index_str_list)
                sub_metric_str = "metric[{}]".format(index_str)
                sub_mask_str =  "mask[{}] = mask_bank".format(index_str)
                metric_bank = eval(sub_metric_str)
                prune_num = int(sparsity_rate * metric_bank.numel())
                if prune_num == 0:
                    threshold = metric_bank.min() -1
                else:
                    threshold = torch.topk(metric_bank.reshape(-1), prune_num, largest=False)[0].max()
                # mask_bank will be used in exec(sub_mask_str)
                mask_bank = torch.gt(metric_bank, threshold).type_as(metric_bank)
                exec(sub_mask_str)

            masks[name] = self._expand_mask(name, mask)
            if self.continuous_mask:
                masks[name]['weight'] *= wrapper.weight_mask
        return masks

class GlobalSparsityAllocator(SparsityAllocator):
    """
    This allocator pruned the weight with smaller metrics in group level.
    This means all layers in a group will sort metrics uniformly.
    The layers with the same config in config_list is a group.
    """
    def generate_sparsity(self, metrics: Dict) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        # {group_index: {layer_name: metric}}
        grouped_metrics = {idx: {name: metrics[name] for name in names}
                           for idx, names in self.pruner.generate_module_groups().items()}
        for _, group_metric_dict in grouped_metrics.items():
            threshold, sub_thresholds = self._calculate_threshold(group_metric_dict)
            for name, metric in group_metric_dict.items():
                mask = torch.gt(metric, min(threshold, sub_thresholds[name])).type_as(metric)
                masks[name] = self._expand_mask(name, mask)
                if self.continuous_mask:
                    masks[name]['weight'] *= self.pruner.get_modules_wrapper()[name].weight_mask
        return masks

    def _calculate_threshold(self, group_metric_dict: Dict[str, Tensor]) -> Tuple[float, Dict[str, float]]:
        metric_list = []
        sub_thresholds = {}
        total_weight_num = 0

        temp_wrapper_config = self.pruner.get_modules_wrapper()[list(group_metric_dict.keys())[0]].config
        total_sparsity = temp_wrapper_config['total_sparsity']
        max_sparsity_per_layer = temp_wrapper_config.get('max_sparsity_per_layer', {})

        for name, metric in group_metric_dict.items():
            wrapper = self.pruner.get_modules_wrapper()[name]

            # We assume the metric value are all positive right now.
            if self.continuous_mask:
                metric = metric * self._compress_mask(wrapper.weight_mask)  # type: ignore

            layer_weight_num = wrapper.weight.data.numel()  # type: ignore
            total_weight_num += layer_weight_num
            expend_times = int(layer_weight_num / metric.numel())

            retention_ratio = 1 - max_sparsity_per_layer.get(name, 1)
            retention_numel = math.ceil(retention_ratio * layer_weight_num)
            removed_metric_num = math.ceil(retention_numel / (wrapper.weight_mask.numel() / metric.numel()))  # type: ignore
            stay_metric_num = metric.numel() - removed_metric_num
            if stay_metric_num <= 0:
                sub_thresholds[name] = metric.min().item() - 1
                continue
            # Remove the weight parts that must be left
            stay_metric = torch.topk(metric.view(-1), stay_metric_num, largest=False)[0]
            sub_thresholds[name] = stay_metric.max()
            if expend_times > 1:
                stay_metric = stay_metric.expand(int(layer_weight_num / metric.numel()), stay_metric_num).contiguous().view(-1)
            metric_list.append(stay_metric)

        total_prune_num = int(total_sparsity * total_weight_num)
        if total_prune_num == 0:
            threshold = torch.cat(metric_list).min().item() - 1
        else:
            threshold = torch.topk(torch.cat(metric_list).view(-1), total_prune_num, largest=False)[0].max().item()
        return threshold, sub_thresholds


class Conv2dDependencyAwareAllocator(SparsityAllocator):
    """
    An allocator specific for Conv2d with dependency-aware.
    """

    def __init__(self, pruner: Pruner, dim: int, dummy_input: Any):
        assert isinstance(dim, int), 'Only support single dim in Conv2dDependencyAwareAllocator.'
        super().__init__(pruner, dim=dim)
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
