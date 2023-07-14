# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from ...base.target_space import PruningTargetSpace


def is_active_target(target_space: PruningTargetSpace):
    return target_space.sparse_ratio is not None or target_space.sparse_threshold is not None
