# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from .base import MetricsCalculator

__all__ = ['NormMetricsCalculator', 'MultiDataNormMetricsCalculator', 'DistMetricsCalculator',
           'APoZRankMetricsCalculator', 'MeanRankMetricsCalculator']


class NormMetricsCalculator(MetricsCalculator):
    """
    Calculate the specify norm for each tensor in data.
    L1, L2, Level, Slim pruner use this to calculate metric.
    """

    def __init__(self, dim: Optional[Union[int, List[int]]] = None, p: Optional[Union[int, float]] = None,
                 block_sparse_size: Optional[Union[int, List[int]]] = None):
        """
        Parameters
        ----------
        dim
            The dimensions that corresponding to the under pruning weight dimensions in collected data.
            None means one-to-one correspondence between pruned dimensions and data, which equal to set `dim` as all data dimensions.
            Only these `dim` will be kept and other dimensions of the data will be reduced.

            Example:

            If you want to prune the Conv2d weight in filter level, and the weight size is (32, 16, 3, 3) [out-channel, in-channel, kernal-size-1, kernal-size-2].
            Then the under pruning dimensions is [0], which means you want to prune the filter or out-channel.

                Case 1: Directly collect the conv module weight as data to calculate the metric.
                Then the data has size (32, 16, 3, 3).
                Mention that the dimension 0 of the data is corresponding to the under pruning weight dimension 0.
                So in this case, `dim=0` will set in `__init__`.

                Case 2: Use the output of the conv module as data to calculate the metric.
                Then the data has size (batch_num, 32, feature_map_size_1, feature_map_size_2).
                Mention that the dimension 1 of the data is corresponding to the under pruning weight dimension 0.
                So in this case, `dim=1` will set in `__init__`.

            In both of these two case, the metric of this module has size (32,).
        p
            The order of norm. None means Frobenius norm.
        block_sparse_size
            This used to describe the block size a metric value represented. By default, None means the block size is ones(len(dim)).
            Make sure len(dim) == len(block_sparse_size), and the block_sparse_size dimension position is corresponding to dim.

            Example:

            The under pruning weight size is (768, 768), and you want to apply a block sparse on dim=[0] with block size [64, 768],
            then you can set block_sparse_size=[64]. The final metric size is (12,).
        """
        super().__init__(dim=dim, block_sparse_size=block_sparse_size)
        self.p = p if p is not None else 'fro'

    def calculate_metrics(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        metrics = {}
        for name, tensor in data.items():
            keeped_dim = list(range(len(tensor.size()))) if self.dim is None else self.dim
            across_dim = list(range(len(tensor.size())))
            [across_dim.pop(i) for i in reversed(keeped_dim)]
            if len(across_dim) == 0:
                metrics[name] = tensor.abs()
            else:
                metrics[name] = tensor.norm(p=self.p, dim=across_dim)
            if self.block_sparse_size is not None:
                assert len(self.block_sparse_size) == len(metrics[name].size())
                for i, step in enumerate(self.block_sparse_size):
                    metrics[name] = metrics[name].unfold(i, step, step)
                across_dim = list(range(len(self.block_sparse_size), 2 * len(self.block_sparse_size)))
                metrics[name] = metrics[name].norm(p=self.p, dim=across_dim)
        return metrics


class MultiDataNormMetricsCalculator(NormMetricsCalculator):
    """
    Sum each list of tensor in data at first, then calculate the specify norm for each sumed tensor.
    TaylorFO pruner use this to calculate metric.
    """

    def calculate_metrics(self, data: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
        new_data = {name: sum(list_tensor) for name, list_tensor in data.items()}
        return super().calculate_metrics(new_data)


class DistMetricsCalculator(MetricsCalculator):
    """
    Calculate the sum of specify distance for each element with all other elements in specify `dim` in each tensor in data.
    FPGM pruner use this to calculate metric.
    """

    def __init__(self, p: float, dim: Union[int, List[int]], block_sparse_size: Optional[Union[int, List[int]]] = None):
        """
        Parameters
        ----------
        dim
            The dimensions that corresponding to the under pruning weight dimensions in collected data.
            None means one-to-one correspondence between pruned dimensions and data, which equal to set `dim` as all data dimensions.
            Only these `dim` will be kept and other dimensions of the data will be reduced.

            Example:

            If you want to prune the Conv2d weight in filter level, and the weight size is (32, 16, 3, 3) [out-channel, in-channel, kernal-size-1, kernal-size-2].
            Then the under pruning dimensions is [0], which means you want to prune the filter or out-channel.

                Case 1: Directly collect the conv module weight as data to calculate the metric.
                Then the data has size (32, 16, 3, 3).
                Mention that the dimension 0 of the data is corresponding to the under pruning weight dimension 0.
                So in this case, `dim=0` will set in `__init__`.

                Case 2: Use the output of the conv module as data to calculate the metric.
                Then the data has size (batch_num, 32, feature_map_size_1, feature_map_size_2).
                Mention that the dimension 1 of the data is corresponding to the under pruning weight dimension 0.
                So in this case, `dim=1` will set in `__init__`.

            In both of these two case, the metric of this module has size (32,).
        p
            The order of norm.
        block_sparse_size
            This used to describe the block size a metric value represented. By default, None means the block size is ones(len(dim)).
            Make sure len(dim) == len(block_sparse_size), and the block_sparse_size dimension position is corresponding to dim.

            Example:

            The under pruning weight size is (768, 768), and you want to apply a block sparse on dim=[0] with block size [64, 768],
            then you can set block_sparse_size=[64]. The final metric size is (12,).
        """
        super().__init__(dim=dim, block_sparse_size=block_sparse_size)
        self.p = p

    def calculate_metrics(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        metrics = {}
        for name, tensor in data.items():
            keeped_dim = list(range(len(tensor.size()))) if self.dim is None else self.dim
            reorder_dim = list(keeped_dim)
            reorder_dim.extend([i for i in range(len(tensor.size())) if i not in keeped_dim])
            reorder_tensor = tensor.permute(*reorder_dim).clone()

            for i, step in enumerate(self.block_sparse_size):
                reorder_tensor = reorder_tensor.unfold(i, step, step)

            metric = torch.ones(*reorder_tensor.size()[:len(keeped_dim)], device=reorder_tensor.device)
            across_dim = list(range(len(keeped_dim), len(reorder_tensor.size())))
            idxs = metric.nonzero(as_tuple=False)
            for idx in idxs:
                other = reorder_tensor
                for i in idx:
                    other = other[i]
                other = other.clone()
                if len(across_dim) == 0:
                    dist_sum = torch.abs(reorder_tensor - other).sum()
                else:
                    dist_sum = torch.norm((reorder_tensor - other), p=self.p, dim=across_dim).sum()
                # NOTE: this place need refactor when support layer level pruning.
                tmp_metric = metric
                for i in idx[:-1]:
                    tmp_metric = tmp_metric[i]
                tmp_metric[idx[-1]] = dist_sum

            metrics[name] = metric
        return metrics


class APoZRankMetricsCalculator(MetricsCalculator):
    """
    This metric counts the zero number at the same position in the tensor list in data,
    then sum the zero number on `dim` and calculate the non-zero rate.
    Note that the metric we return is (1 - apoz), because we assume a higher metric value has higher importance.
    APoZRank pruner use this to calculate metric.
    """
    def calculate_metrics(self, data: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
        metrics = {}
        for name, tensor_list in data.items():
            # NOTE: dim=0 means the batch dim is 0
            activations = torch.cat(tensor_list, dim=0)
            _eq_zero = torch.eq(activations, torch.zeros_like(activations))
            keeped_dim = list(range(len(_eq_zero.size()))) if self.dim is None else self.dim
            across_dim = list(range(len(_eq_zero.size())))
            [across_dim.pop(i) for i in reversed(keeped_dim)]
            # The element number on each [keeped_dim + 1] in _eq_zero
            total_size = 1
            for dim, dim_size in enumerate(_eq_zero.size()):
                if dim not in keeped_dim:
                    total_size *= dim_size
            _apoz = torch.sum(_eq_zero, dim=across_dim).type_as(activations) / total_size
            # NOTE: the metric is (1 - apoz) because we assume the smaller metric value is more needed to be pruned.
            metrics[name] = torch.ones_like(_apoz) - _apoz
        return metrics


class MeanRankMetricsCalculator(MetricsCalculator):
    """
    This metric simply concat the list of tensor on dim 0, and average on `dim`.
    MeanRank pruner use this to calculate metric.
    """
    def calculate_metrics(self, data: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
        metrics = {}
        for name, tensor_list in data.items():
            # NOTE: dim=0 means the batch dim is 0
            activations = torch.cat(tensor_list, dim=0)
            keeped_dim = list(range(len(activations.size()))) if self.dim is None else self.dim
            across_dim = list(range(len(activations.size())))
            [across_dim.pop(i) for i in reversed(keeped_dim)]
            metrics[name] = torch.mean(activations, across_dim)
        return metrics
