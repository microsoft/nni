# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .base import ExperimentConfig


@dataclass(init=False)
class LocalExperimentConfig(ExperimentConfig):
    use_active_gpu: bool = False

    _training_service: str = 'local'

    def experiment_config_json(self) -> Dict[str, Any]:
        ret = super().experiment_config_json()
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
        #ret['local_config'] = {
        #    'useActiveGpu': self.use_active_gpu
        #}
        return ret

    def cluster_metadata_json(self) -> Any:
        return {
            'trial_config': {
                'command': self.trial_command,
                'codeDir': str(Path(self.trial_code_directory).resolve())
            }
        }
