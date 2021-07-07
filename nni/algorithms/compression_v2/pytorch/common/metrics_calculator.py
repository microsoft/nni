from typing import Dict, List, Optional, Union

from torch import Tensor

from nni.algorithms.compression_v2.pytorch.base.common import MetricsCalculator


class NaiveMetricsCalculator(MetricsCalculator):
    def calculate_metrics(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return data


class NormMetricsCalculator(MetricsCalculator):
    def __init__(self, p: Optional[int], dim: Optional[Union[int, List[int]]] = None):
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
