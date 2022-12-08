# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Dict

import torch

from .utils import active_sparse_targets_filter, generate_sparsity, _MASKS, _METRICS
from ..base.compressor import Pruner


class BasicPruner(Pruner):

    def _collect_data(self) -> _METRICS:
        return active_sparse_targets_filter(self.target_spaces)

    def _generate_sparsity(self, metrics: _METRICS) -> _MASKS:
        return generate_sparsity(metrics, self.target_spaces)
