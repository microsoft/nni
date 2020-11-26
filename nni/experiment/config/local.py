# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Experiment configuration structures.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .base import ExperimentConfig


@dataclass(init=False)
class LocalExperimentConfig(ExperimentConfig):
    use_active_gpu: bool = False

    _training_service: str = 'local'

    _json_schema = {
        **ExperimentConfig._json_schema,
        'use_active_gpu': lambda value: (None, None)
    }

    def to_json(self) -> Dict[str, Any]:
        ret = super().to_json()
        ret['clusterMetaData'] = [
            {
                'key': 'codeDir',
                'value': str(Path(self.trial_code_directory).resolve())
            },
            {
                'key': 'command',
                'value': self.trial_command
            }
        ]
        return ret

    def to_cluster_metadata(self) -> Any:
        return {
            'trial_config': {
                'command': self.trial_command,
                'codeDir': str(Path(self.trial_code_directory).resolve())
            }
        }
