# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Dict, List

import torch

from nni.compression.pytorch.utils.scaling import Scaling

from .utils import active_sparse_targets_filter, generate_sparsity, _MASKS, _METRICS
from ..base.compressor import Pruner
from ..base.config import trans_legacy_config_list
from ..base.target_space import TargetType


class BasicPruner(Pruner):
    def __init__(self, model: torch.nn.Module, config_list: List[Dict]):
        config_list = trans_legacy_config_list(config_list, sparse_granularity='out_channel')
        super().__init__(model, config_list)
        self._register_scalers()

    def _register_scalers(self):
        for _, ts in self.target_spaces.items():
            for _, target_space in ts.items():
                if target_space.sparse_granularity is None:
                    continue
                if target_space.sparse_granularity == 'out_channel':
                    assert target_space._target_type is TargetType.PARAMETER
                    target_space._scaler = Scaling([1], kernel_padding_mode='back', kernel_padding_val=-1)
                elif target_space.sparse_granularity == 'in_channel':
                    assert target_space._target_type is TargetType.PARAMETER
                    target_space._scaler = Scaling([1], kernel_padding_mode='front', kernel_padding_val=-1)
                else:
                    assert all(isinstance(_, int) for _ in target_space.sparse_granularity)
                    target_space._scaler = Scaling(target_space.sparse_granularity, kernel_padding_mode='front', kernel_padding_val=1)

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return active_sparse_targets_filter(self.target_spaces)

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> _METRICS:
        metrics = defaultdict(dict)
        for module_name, module_data in data.items():
            for target_name, target_data in module_data.items():
                target_space = self.target_spaces[module_name][target_name]
                if target_space._scaler is None:
                    metrics[module_name][target_name] = target_data.abs()
                else:
                    def reduce_func(t: torch.Tensor) -> torch.Tensor:
                        return t.norm(p=1, dim=-1)

                    metrics[module_name][target_name] = target_space._scaler.shrink(target_data, reduce_func)
        return metrics

    def _generate_sparsity(self, metrics: _METRICS) -> _MASKS:
        return generate_sparsity(metrics, self.target_spaces)
