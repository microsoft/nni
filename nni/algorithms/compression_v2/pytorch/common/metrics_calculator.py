from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from nni.algorithms.compression_v2.pytorch.base.common import MetricsCalculator


class AbsMetricsCalculator(MetricsCalculator):
    def calculate_metrics(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {name: layer_data.abs() for name, layer_data in data.items()}


class NormMetricsCalculator(MetricsCalculator):
    def __init__(self, p: Optional[int] = None, dim: Optional[Union[int, List[int]]] = None):
        self.p = p if p is not None else 'fro'
        self.dim = dim if not isinstance(dim, int) else [dim]
        if self.dim is not None:
            assert all(i >= 0 for i in self.dim)
            self.dim = sorted(self.dim)

    def calculate_metrics(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        metrics = {}
        for name, tensor in data.items():
            across_dim = list(range(len(tensor.size())))
            [across_dim.pop(i) for i in reversed(self.dim)]
            metrics[name] = tensor.norm(p=self.p, dim=across_dim)
        return metrics


class DistMetricsCalculator(MetricsCalculator):
    def __init__(self, p: Optional[int], dim: Optional[Union[int, List[int]]]):
        self.p = p
        self.dim = dim if not isinstance(dim, int) else [dim]
        assert all(i >= 0 for i in self.dim)
        self.dim = sorted(self.dim)

    def calculate_metrics(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        metrics = {}
        for name, tensor in data.items():
            reorder_dim = list(self.dim)
            reorder_dim.extend([i for i in range(len(tensor.size())) if i not in self.dim])
            reorder_tensor = tensor.permute(*reorder_dim).clone()

            metric = torch.ones(*reorder_tensor.size()[:len(self.dim)], device=reorder_tensor.device)
            idxs = metric.nonzero()
            for idx in idxs:
                other = reorder_tensor
                for i in idx:
                    other = other[i]
                other = other.clone()
                dist_sum = torch.dist(reorder_tensor, other, self.p).sum()
                tmp_metric = metric
                for i in idx[:-1]:
                    tmp_metric = tmp_metric[i]
                tmp_metric[idx[-1]] = dist_sum

            metrics[name] = metric
        return metrics


class APoZRankMetricsCalculator(MetricsCalculator):
    def __init__(self, dim: Optional[Union[int, List[int]]]):
        self.dim = dim if not isinstance(dim, int) else [dim]
        assert all(i >= 0 for i in self.dim)
        self.dim = sorted(self.dim)

    def calculate_metrics(self, data: Dict[str, Dict[str, List[Tensor]]]) -> Dict[str, Tensor]:
        metrics = {}
        for _, data_dict in data.items():
            for name, tensor_list in data_dict.items():
                assert name not in metrics, 'Already calculate apoz for {}, something goes wrong.'.format(name)
                # NOTE: dim=0 means the batch dim is 0
                activations = torch.cat(tensor_list, dim=0)
                _eq_zero = torch.eq(activations, torch.zeros_like(activations))
                across_dim = list(range(len(_eq_zero.size())))
                [across_dim.pop(i + 1) for i in reversed(self.dim)]
                # The element number on each [self.dim + 1] in _eq_zero
                total_size = 1
                for dim, dim_size in enumerate(_eq_zero.size()):
                    if dim - 1 not in self.dim:
                        total_size *= dim_size
                _apoz = torch.sum(_eq_zero, dim=across_dim, dtype=torch.float64) / total_size
                metrics[name] = _apoz
        return metrics


class MeanRankMetricsCalculator(MetricsCalculator):
    def __init__(self, dim: Optional[Union[int, List[int]]]):
        self.dim = dim if not isinstance(dim, int) else [dim]
        assert all(i >= 0 for i in self.dim)
        self.dim = sorted(self.dim)

    def calculate_metrics(self, data: Dict[str, Dict[str, List[Tensor]]]) -> Dict[str, Tensor]:
        metrics = {}
        for _, data_dict in data.items():
            for name, tensor_list in data_dict.items():
                assert name not in metrics, 'Already calculate mean for {}, something goes wrong.'.format(name)
                # NOTE: dim=0 means the batch dim is 0
                activations = torch.cat(tensor_list, dim=0)
                across_dim = list(range(len(activations.size())))
                [across_dim.pop(i + 1) for i in reversed(self.dim)]
                metrics[name] = torch.mean(activations, across_dim)
        return metrics
