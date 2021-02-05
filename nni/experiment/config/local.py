# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import List, Optional, Union

from .common import TrainingServiceConfig

__all__ = ['LocalConfig']

@dataclass(init=False)
class LocalConfig(TrainingServiceConfig):
    platform: str = 'local'
    use_active_gpu: Optional[bool] = None
    max_trial_number_per_gpu: int = 1
    gpu_indices: Optional[Union[List[int], str]] = None

    _canonical_rules = {
        'gpu_indices': lambda value: [int(idx) for idx in value.split(',')] if isinstance(value, str) else value
    }

    _validation_rules = {
        'platform': lambda value: (value == 'local', 'cannot be modified'),
        'max_trial_number_per_gpu': lambda value: value > 0,
        'gpu_indices': lambda value: all(idx >= 0 for idx in value) and len(value) == len(set(value))
    }
