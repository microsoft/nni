# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Dict

import torch

from .utils import active_sparse_targets_filter
from ..base.compressor import Pruner


class BasicPruner(Pruner):

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return active_sparse_targets_filter(self.target_spaces)

    def _generate_sparsity(self, metrics: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        return super()._generate_sparsity(metrics)
