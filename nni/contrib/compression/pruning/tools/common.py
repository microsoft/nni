# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

import torch

from ...base.target_space import PruningTargetSpace


_TARGET_SPACES = Dict[str, Dict[str, PruningTargetSpace]]
_MASKS = Dict[str, Dict[str, torch.Tensor]]
_METRICS = Dict[str, Dict[str, torch.Tensor]]
