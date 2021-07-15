from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
from torch import Tensor

from nni.algorithms.compression_v2.pytorch.base.pruner import Pruner
from nni.algorithms.compression_v2.pytorch.base.pruner_tools import SparsityAllocator

from nni.compression.pytorch.utils.shape_dependency import ChannelDependency, GroupDependency


def get_sparsity_allocator(pruner: Pruner, mode: str, dim: Optional[Union[int, list]] = None,
                           max_sparsity_per_layer: Optional[float] = None, dummy_input: Optional[Tensor] = None):
    if mode == 'normal':
        return NormalSparsityAllocator(pruner=pruner, dim=dim)
    elif mode == 'global':
        assert max_sparsity_per_layer is not None, 'max_sparsity_per_layer is required in GlobalSparsityAllocator.'
        return GlobalSparsityAllocator(pruner=pruner, dim=dim, max_sparsity_per_layer=max_sparsity_per_layer)
    elif mode == 'dependency_aware':
        assert dummy_input is not None, 'dummy_input is required in Conv2dDependencyAwareAllocator'
        return Conv2dDependencyAwareAllocator(pruner=pruner, dim=dim, dummy_input=dummy_input)


class NormalSparsityAllocator(SparsityAllocator):
    """
    This allocator simply pruned the weight with smaller metrics in layer level.
    """
    def generate_sparsity(self, metrics: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        for name, wrapper in self.pruner._get_modules_wrapper().items():
            sparsity_rate = wrapper.config['sparsity']

            assert name in metrics, 'Metric of %s is not calculated.'
            metric = metrics[name]
            prune_num = int(sparsity_rate * metric.numel())
            if prune_num == 0:
                continue
            threshold = torch.topk(metric.view(-1), prune_num, largest=False)[0].max()
            mask = torch.gt(metric, threshold).type_as(metric)
            masks[name] = self._expand_mask_with_dim(name, mask)
        return masks


class GlobalSparsityAllocator(SparsityAllocator):
    """
    This allocator pruned the weight with smaller metrics in group level.
    This means all layers in a group will sort metrics uniformly.
    The layers with the same config in config_list is a group.
    """
    def __init__(self, pruner: Pruner, dim: Optional[Union[int, List[int]]] = None, max_sparsity_per_layer: float = 1):
        assert 0 < max_sparsity_per_layer <= 1, 'max_sparsity_per_layer must in range (0, 1].'
        self._max_sparsity_per_layer = max_sparsity_per_layer
        super().__init__(pruner, dim=dim)

    def generate_sparsity(self, metrics: Dict) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        # {group_index: {layer_name: metric}}
        grouped_metrics = {idx: {name: metrics[name] for name in names}
                           for idx, names in self.pruner.generate_module_groups().items()}
        for _, group_metric_dict in grouped_metrics.items():
            threshold, sub_thresholds = self._calculate_threshold(group_metric_dict)
            for name, metric in group_metric_dict.items():
                mask = torch.gt(metric, min(threshold, sub_thresholds[name])).type_as(metric)
                masks[name] = self._expand_mask_with_dim(name, mask)
        return masks

    def _calculate_threshold(self, group_metric_dict: Dict[int, Dict[str, Tensor]]) -> Tuple[float, Dict[str, float]]:
        metric_list = []
        sub_thresholds = {}
        total_weight_num = 0
        for name, metric in group_metric_dict.items():
            layer_weight_num = self.pruner._get_modules_wrapper()[name].module.weight.data.numel()
            stay_num = int(metric.numel() * self._max_sparsity_per_layer)
            # Remove the weight parts that must be left
            stay_metric = torch.topk(metric.abs().view(-1), stay_num, largest=False)[0]
            sub_thresholds[name] = stay_metric.max()
            expend_times = int(layer_weight_num / metric.numel())
            if expend_times > 1:
                stay_metric = stay_metric.expand(stay_num, int(layer_weight_num / metric.numel())).view(-1)
            metric_list.append(stay_metric)
            total_weight_num += layer_weight_num

        sparsity = self.pruner._get_modules_wrapper()[name].config['sparsity']
        assert sparsity <= self._max_sparsity_per_layer, 'Sparsity ratio should less than max_sparsity_per_layer.'
        total_prune_num = int(sparsity * total_weight_num)

        threshold = torch.topk(torch.cat(metric_list).view(-1), total_prune_num, largest=False)[0].max().item()
        return threshold, sub_thresholds


class Conv2dDependencyAwareAllocator(SparsityAllocator):
    """
    A specify allocator for Conv2d with dependency aware.
    """

    def __init__(self, pruner: Pruner, dim: int, dummy_input: Tensor):
        assert isinstance(dim, int), 'Only support single dim in Conv2dDependencyAwareAllocator.'
        super().__init__(pruner, dim=dim)
        self.dummy_input = dummy_input

    def _get_dependency(self):
        graph = self.pruner.generate_graph(dummy_input=self.dummy_input)
        self.channel_depen = ChannelDependency(traced_model=graph).dependency_sets
        self.group_depen = GroupDependency(traced_model=graph).dependency_sets

    def generate_sparsity(self, metrics: Dict) -> Dict[str, Dict[str, Tensor]]:
        self._get_dependency()
        masks = {}
        grouped_metrics = {idx: {name: metrics[name] for name in names}
                           for idx, names in enumerate(self.channel_depen)}
        for _, group_metric_dict in grouped_metrics.items():
            group_metric = self._group_metric_calculate(group_metric_dict)

            sparsities = {name: self.pruner._get_modules_wrapper()[name].config['sparsity'] for name in group_metric_dict.keys()}
            min_sparsity = min(sparsities.values())

            conv2d_groups = [self.group_depen[name] for name in group_metric_dict.keys()]
            max_conv2d_group = np.lcm.reduce(conv2d_groups)

            pruned_per_conv2d_group = int(group_metric.numel() / max_conv2d_group * min_sparsity)
            conv2d_group_step = int(group_metric.numel() / max_conv2d_group)

            group_mask = []
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

            for name, metric in group_metric_dict.items():
                metric = (metric - metric.min()) * group_mask
                pruned_num = int(sparsities[name] * len(metric))
                threshold = torch.topk(metric, pruned_num, largest=False)[0].max()
                mask = torch.gt(metric, threshold).type_as(metric)
                masks[name] = self._expand_mask_with_dim(name, mask)

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
