# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Dict

import torch

from .utils import active_sparse_targets_filter, generate_sparsity, _MASKS, _METRICS
from ..base.compressor import Pruner


class BasicPruner(Pruner):

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return active_sparse_targets_filter(self.target_spaces)

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> _METRICS:
        metrics = defaultdict(dict)
        for module_name, module_data in data.items():
            for target_name, target_data in module_data.items():
                metrics[module_name][target_name] = target_data.clone()
        return metrics

    def _generate_sparsity(self, metrics: _METRICS) -> _MASKS:
        return generate_sparsity(metrics, self.target_spaces)
