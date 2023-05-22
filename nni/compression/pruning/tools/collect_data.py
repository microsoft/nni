# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch

from .utils import is_active_target
from ...base.compressor import _PRUNING_TARGET_SPACES


_DATA = Dict[str, Dict[str, torch.Tensor]]


def active_sparse_targets_filter(target_spaces: _PRUNING_TARGET_SPACES) -> _DATA:
    # filter all targets need to active generate sparsity
    active_targets = defaultdict(dict)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            if is_active_target(target_space):
                assert target_space.target is not None
                active_targets[module_name][target_name] = target_space.target.clone().detach()
    return active_targets
