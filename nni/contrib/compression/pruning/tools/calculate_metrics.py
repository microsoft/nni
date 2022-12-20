from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch

from .common import _METRICS, _TARGET_SPACES


def norm_metrics(p: str | int, data: Dict[str, Dict[str, torch.Tensor]], target_spaces: _TARGET_SPACES) -> _METRICS:
    metrics = defaultdict(dict)
    for module_name, module_data in data.items():
        for target_name, target_data in module_data.items():
            target_space = target_spaces[module_name][target_name]
            if target_space._scaler is None:
                metrics[module_name][target_name] = target_data.abs()
            else:
                def reduce_func(t: torch.Tensor) -> torch.Tensor:
                    return t.norm(p=p, dim=-1)

                metrics[module_name][target_name] = target_space._scaler.shrink(target_data, reduce_func)
    return metrics
