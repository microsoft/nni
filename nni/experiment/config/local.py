# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import List, Optional, Union

from .common import ExperimentConfig, TrainingServiceConfig

__all__ = ['LocalConfig']


@dataclass(init=False)
class LocalConfig(TrainingServiceConfig):
    use_active_gpu: bool
    max_trial_number_per_gpu: int = 1
    gpu_indices: Optional[Union[List[int], str]] = None

    _training_service: str = 'local'

    _validation_rules = {
        'max_trial_number_per_gpu': lambda: val, _: val > 0,
        'gpu_indices': lambda val, _: isinstance(val, str) or (all(i >= 0 for i in val) and len(val) == len(set(val)))
    }

    def _cluster_metadata(self, exp: ExperimentConfig) -> Any:
        ...  # for Experiment.start() only
