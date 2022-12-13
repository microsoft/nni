# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Dict, List

import torch

from .tools.common import _MASKS, _METRICS, _TARGET_SPACES
from .tools.collect_data import active_sparse_targets_filter
from .tools.sparse_gen import generate_sparsity
from ..base.compressor import Pruner
from ..base.config import trans_legacy_config_list


class L1NormPruner(Pruner):
    def __init__(self, model: torch.nn.Module, config_list: List[Dict]):
        config_list = trans_legacy_config_list(config_list, default_sparse_granularity='out_channel')
        super().__init__(model, config_list)
        self._target_spaces: _TARGET_SPACES

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return active_sparse_targets_filter(self._target_spaces)

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> _METRICS:
        metrics = defaultdict(dict)
        for module_name, module_data in data.items():
            for target_name, target_data in module_data.items():
                target_space = self._target_spaces[module_name][target_name]
                if target_space._scaler is None:
                    metrics[module_name][target_name] = target_data.abs()
                else:
                    def reduce_func(t: torch.Tensor) -> torch.Tensor:
                        return t.norm(p=1, dim=-1)

                    metrics[module_name][target_name] = target_space._scaler.shrink(target_data, reduce_func)
        return metrics

    def _generate_sparsity(self, metrics: _METRICS) -> _MASKS:
        return generate_sparsity(metrics, self._target_spaces)
