# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import List, Optional, Union

from .common import TrainingServiceConfig
from . import util

__all__ = ['LocalConfig']

@dataclass(init=False)
class LocalConfig(TrainingServiceConfig):
    platform: str = 'local'
    use_active_gpu: Optional[bool] = None
    max_trial_number_per_gpu: int = 1
    gpu_indices: Union[List[int], str, int, None] = None

    _canonical_rules = {
        'gpu_indices': util.canonical_gpu_indices
    }

    _validation_rules = {
        'platform': lambda value: (value == 'local', 'cannot be modified'),
        'max_trial_number_per_gpu': lambda value: value > 0,
        'gpu_indices': lambda value: all(idx >= 0 for idx in value) and len(value) == len(set(value))
    }
