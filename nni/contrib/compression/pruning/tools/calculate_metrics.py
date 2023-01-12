# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch

from .collect_data import _DATA
from ...base.compressor import _PRUNING_TARGET_SPACES


_METRICS = Dict[str, Dict[str, torch.Tensor]]


def norm_metrics(p: str | int, data: _DATA, target_spaces: _PRUNING_TARGET_SPACES) -> _METRICS:
    """
    Calculate the norm of each block of the value in the given data.

    Parameters
    ----------
    p
        The order of norm. Please refer `torch.norm <https://pytorch.org/docs/stable/generated/torch.norm.html>`__.
    data
        {module_name: {target_name: val}}.
    target_spaces
        {module_name: {target_name: pruning_target_space}}. Used to get the related scaler for each value in data.
    """
    def reduce_func(t: torch.Tensor) -> torch.Tensor:
        return t.norm(p=p, dim=-1)

    metrics = defaultdict(dict)
    for module_name, module_data in data.items():
        for target_name, target_data in module_data.items():
            target_space = target_spaces[module_name][target_name]
            if target_space._scaler is None:
                metrics[module_name][target_name] = target_data.abs()
            else:
                metrics[module_name][target_name] = target_space._scaler.shrink(target_data, reduce_func)
    return metrics


def sum_sigmoid_metric(data: _DATA, target_spaces: _PRUNING_TARGET_SPACES) -> _METRICS:
    """
    Calculate the sum of each block of the value in the given data.

    Parameters
    ----------
    data
        {module_name: {target_name: val}}.
    target_spaces
        {module_name: {target_name: pruning_target_space}}. Used to get the related scaler for each value in data.
    """
    def reduce_func(t: torch.Tensor) -> torch.Tensor:
        return t.sum(dim=-1).sigmoid()

    metrics = defaultdict(dict)
    for module_name, module_data in data.items():
        for target_name, target_data in module_data.items():
            target_space = target_spaces[module_name][target_name]
            if target_space._scaler is None:
                metrics[module_name][target_name] = target_data
            else:
                metrics[module_name][target_name] = target_space._scaler.shrink(target_data, reduce_func)
    return metrics


def mean_metric(data: _DATA, target_spaces: _PRUNING_TARGET_SPACES) -> _METRICS:
    """
    Calculate the mean of each block of the value in the given data.

    Parameters
    ----------
    data
        {module_name: {target_name: val}}.
    target_spaces
        {module_name: {target_name: pruning_target_space}}. Used to get the related scaler for each value in data.
    """
    def reduce_func(t: torch.Tensor) -> torch.Tensor:
        return t.mean(dim=-1)

    metrics = defaultdict(dict)
    for module_name, module_data in data.items():
        for target_name, target_data in module_data.items():
            target_space = target_spaces[module_name][target_name]
            if target_space._scaler is None:
                metrics[module_name][target_name] = target_data
            else:
                metrics[module_name][target_name] = target_space._scaler.shrink(target_data, reduce_func)
    return metrics
