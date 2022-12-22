# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

from collections import defaultdict

from .common import _METRICS
from ...base.compressor import _PRUNING_TARGET_SPACES


def active_sparse_targets_filter(target_spaces: _PRUNING_TARGET_SPACES) -> _METRICS:
    # filter all targets need to active generate sparsity
    active_targets = defaultdict(dict)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            if target_space.sparse_ratio or target_space.sparse_threshold:
                active_targets[module_name][target_name] = target_space.target
    return active_targets
