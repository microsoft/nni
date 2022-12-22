# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

import torch


_MASKS = Dict[str, Dict[str, torch.Tensor]]
_METRICS = Dict[str, Dict[str, torch.Tensor]]
