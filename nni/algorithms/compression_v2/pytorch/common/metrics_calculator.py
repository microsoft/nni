from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from nni.algorithms.compression_v2.pytorch.base.common import MetricsCalculator


class NaiveMetricsCalculator(MetricsCalculator):
    def calculate_metrics(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return data


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
