from typing import Dict, List, Tuple, Union, Optional

import torch
from torch import Tensor

from nni.algorithms.compression_v2.pytorch.base.pruner import Pruner
from nni.algorithms.compression_v2.pytorch.base.common import SparsityAllocator


class NormalSparsityAllocator(SparsityAllocator):
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
    def __init__(self, pruner: Pruner, dim: Optional[Union[int, List[int]]] = None, max_sparsity_per_layer: float = 1):
        assert 0 < max_sparsity_per_layer <= 1, 'max_sparsity_per_layer must in range (0, 1].'
        self._max_sparsity_per_layer = max_sparsity_per_layer
        super().__init__(pruner, dim=dim)

    def generate_sparsity(self, metrics: Dict) -> Dict[str, Dict[str, Tensor]]:
        masks = {}
        # {group_index: {layer_name: metric}}
        grouped_metrics = {idx: {name: metrics[name] for name in names}
                           for idx, names in self.pruner._config_based_group_info.items()}
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
            layer_weight_num = self.pruner._get_modules_wrapper()[name].weight.data.numel()
            stay_num = int(metric.numel() * self._max_sparsity_per_layer)
            # Remove the weight parts that must be left
            stay_metric = torch.topk(metric.abs().view(-1), stay_num, largest=False)[0]
            sub_thresholds[name] = stay_metric.max()
            metric_list.append(stay_metric.expand(stay_num, int(layer_weight_num / metric.numel())).view(-1))
            total_weight_num += layer_weight_num

        sparsity = self.pruner._get_modules_wrapper()[name].config['sparsity']
        assert sparsity <= self._max_sparsity_per_layer, 'Sparsity ratio should less than max_sparsity_per_layer.'
        total_prune_num = int(sparsity * total_weight_num)

        threshold = torch.topk(torch.cat(metric_list).view(-1), total_prune_num, largest=False)[0].max().item()
        return threshold, sub_thresholds
