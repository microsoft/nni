# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from .base import MetricsCalculator
from ...utils import Scaling

__all__ = ['NormMetricsCalculator', 'HookDataNormMetricsCalculator', 'DistMetricsCalculator',
           'APoZRankMetricsCalculator', 'MeanRankMetricsCalculator', 'StraightMetricsCalculator']


class StraightMetricsCalculator(MetricsCalculator):
    """
    This metrics calculator directly returns a copy of data as metrics.
    """
    def calculate_metrics(self, data: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        metrics = {}
        for module_name, targets_data in data.items():
            metrics[module_name] = {}
            for target_name, target_data in targets_data.items():
                metrics[module_name][target_name] = target_data.clone().detach()
        return metrics


class NormMetricsCalculator(MetricsCalculator):
    """
    Calculate the specify norm for each tensor in data.
    L1, L2, Level, Slim pruner use this to calculate metric.

    Parameters
    ----------
    p
        The order of norm. None means Frobenius norm.
    scalers
        Please view the base class `MetricsCalculator` docstring.
    """

    def __init__(self, p: int | float | None = None, scalers: Dict[str, Dict[str, Scaling]] | Scaling | None = None):
        super().__init__(scalers=scalers)
        self.p = p if p is not None else 'fro'

    def calculate_metrics(self, data: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        def reduce_func(t: Tensor) -> Tensor:
            return t.norm(p=self.p, dim=-1)  # type: ignore

        metrics = {}
        for module_name, targets_data in data.items():
            metrics[module_name] = {}
            for target_name, target_data in targets_data.items():
                scaler = self._get_scaler(module_name, target_name)
                metrics[module_name][target_name] = scaler.shrink(target_data, reduce_func)
        return metrics


class HookDataNormMetricsCalculator(NormMetricsCalculator):
    """
    The hook data value format is a two-element list [batch_number, cumulative_data].
    Directly use the cumulative_data as new_data to calculate norm metric.
    TaylorFO pruner uses this to calculate metric.
    """

    def calculate_metrics(self, data: Dict[str, Dict[str, List[Tensor]]]) -> Dict[str, Dict[str, Tensor]]:
        new_data = {}
        for module_name, targets_data in data.items():
            new_data[module_name] = {}
            for target_name, (_, target_data) in targets_data.items():
                new_data[module_name][target_name] = target_data
        return super().calculate_metrics(new_data)


class DistMetricsCalculator(MetricsCalculator):
    """
    Calculate the sum of specify distance for each element with all other elements in specify `dim` in each tensor in data.
    FPGM pruner uses this to calculate metric.

    Parameters
    ----------
    p
        The order of norm. None means Frobenius norm.
    scalers
        Please view the base class `MetricsCalculator` docstring.
    """

    def __init__(self, p: int | float | None = None, scalers: Dict[str, Dict[str, Scaling]] | Scaling | None = None):
        super().__init__(scalers=scalers)
        self.p = p if p is not None else 'fro'

    def calculate_metrics(self, data: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        def reduce_func(t: Tensor) -> Tensor:
            reshape_data = t.reshape(-1, t.shape[-1])
            metric = torch.zeros(reshape_data.shape[0], device=reshape_data.device)
            for i in range(reshape_data.shape[0]):
                metric[i] = (reshape_data - reshape_data[i]).norm(p=self.p, dim=-1).sum()  # type: ignore
            return metric.reshape(t.shape[:-1])

        metrics = {}
        for module_name, targets_data in data.items():
            metrics[module_name] = {}
            for target_name, target_data in targets_data.items():
                scaler = self._get_scaler(module_name, target_name)
                metrics[module_name][target_name] = scaler.shrink(target_data, reduce_func)
        return metrics


class APoZRankMetricsCalculator(MetricsCalculator):
    """
    The data value format is a two-element list [batch_number, batch_wise_zeros_count_sum].
    This metric sum the zero number on `dim` then devide the (batch_number * across_dim_size) to calculate the non-zero rate.
    Note that the metric we return is (1 - apoz), because we assume a higher metric value has higher importance.
    APoZRank pruner uses this to calculate metric.
    """

    def calculate_metrics(self, data: Dict[str, Dict[str, List[Tensor]]]) -> Dict[str, Dict[str, Tensor]]:
        def reduce_func(t: Tensor) -> Tensor:
            return 1 - t.mean(dim=-1)

        metrics = {}
        for module_name, targets_data in data.items():
            metrics[module_name] = {}
            for target_name, target_data in targets_data.items():
                target_data = target_data[1] / target_data[0]
                scaler = self._get_scaler(module_name, target_name)
                metrics[module_name][target_name] = scaler.shrink(target_data, reduce_func)
        return metrics


class MeanRankMetricsCalculator(MetricsCalculator):
    """
    The data value format is a two-element list [batch_number, batch_wise_activation_sum].
    This metric simply calculate the average on `self.dim`, then divide by the batch_number.
    MeanRank pruner uses this to calculate metric.
    """
    def calculate_metrics(self, data: Dict[str, Dict[str, List[Tensor]]]) -> Dict[str, Dict[str, Tensor]]:
        def reduce_func(t: Tensor) -> Tensor:
            return t.mean(dim=-1)

        metrics = {}
        for module_name, targets_data in data.items():
            metrics[module_name] = {}
            for target_name, target_data in targets_data.items():
                target_data = target_data[1] / target_data[0]
                scaler = self._get_scaler(module_name, target_name)
                metrics[module_name][target_name] = scaler.shrink(target_data, reduce_func)
        return metrics
