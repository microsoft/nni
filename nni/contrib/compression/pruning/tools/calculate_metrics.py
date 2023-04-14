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
        return t.norm(p=p, dim=-1)  # type: ignore

    metrics = defaultdict(dict)
    for module_name, module_data in data.items():
        for target_name, target_data in module_data.items():
            target_space = target_spaces[module_name][target_name]
            if target_space._scaler is None:
                metrics[module_name][target_name] = target_data.abs()
            else:
                metrics[module_name][target_name] = target_space._scaler.shrink(target_data, reduce_func, keepdim=True)
    return metrics


def fpgm_metrics(p: str | int, data: _DATA, target_spaces: _PRUNING_TARGET_SPACES) -> _METRICS:
    def reduce_func(t: torch.Tensor) -> torch.Tensor:
        reshape_data = t.reshape(-1, t.shape[-1])
        metric = torch.zeros(reshape_data.shape[0]).type_as(reshape_data)
        for i in range(reshape_data.shape[0]):
            metric[i] = (reshape_data - reshape_data[i]).norm(p=p, dim=-1).sum()  # type: ignore
        return metric.reshape(t.shape[:-1])

    metrics = defaultdict(dict)
    for module_name, module_data in data.items():
        for target_name, target_data in module_data.items():
            target_space = target_spaces[module_name][target_name]
            assert target_space._scaler is not None, 'FPGM metric do not support finegrained sparse pattern.'
            metrics[module_name][target_name] = target_space._scaler.shrink(target_data, reduce_func, keepdim=True)
    return metrics
