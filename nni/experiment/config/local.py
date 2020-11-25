# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Experiment configuration structures.
"""

from dataclasses import dataclass
from typing import Any, Dict

from .base import ExperimentConfig


@dataclass(init=False)
class LocalExperimentConfig(ExperimentConfig):
    use_active_gpu: bool = False

    _training_service: str = 'local'

    def to_json(self) -> Dict[str, Any]:
        ret = super().to_json()
        ret['clusterMetaData'] = {
            'codeDir': str(self.trial_code_directory),
            'command': self.trial_command
        }
        return ret
