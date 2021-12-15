# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Configuration for OpenPAI training service.

Check the reference_ for explaination of each field.

You may also want to check `OpenPAI training service doc`_.

.. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html

.. _OpenPAI training service doc: https://nni.readthedocs.io/en/stable/TrainingService/PaiMode.html

"""

__all__ = ['OpenpaiConfig']

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from ..training_service import TrainingServiceConfig
from ..utils import PathLike

@dataclass(init=False)
class OpenpaiConfig(TrainingServiceConfig):
    platform: str = 'openpai'
    host: str
    username: str
    token: str
    trial_cpu_number: int
    trial_memory_size: Union[str, int]
    storage_config_name: str
    docker_image: str = 'msranni/nni:latest'
    virtual_cluster: Optional[str]
    local_storage_mount_point: PathLike
    container_storage_mount_point: str
    reuse_mode: bool = True

    openpai_config: Optional[Dict] = None
    openpai_config_file: Optional[PathLike] = None

    def _canonicalize(self, parents):
        super()._canonicalize(parents)
        if '://' not in self.host:
            self.host = 'https://' + self.host

    def _validate_canonical(self) -> None:
        super()._validate_canonical()
        if self.trial_gpu_number is None:
            raise ValueError('OpenpaiConfig: trial_gpu_number is not set')
        if not Path(self.local_storage_mount_point).is_dir():
            raise ValueError(
                f'OpenpaiConfig: local_storage_mount_point "(self.local_storage_mount_point)" is not a directory'
            )
        if self.openpai_config is not None and self.openpai_config_file is not None:
            raise ValueError('openpai_config and openpai_config_file can only be set one')
        if self.openpai_config_file is not None and not Path(self.openpai_config_file).is_file():
            raise ValueError(f'OpenpaiConfig: openpai_config_file "(self.openpai_config_file)" is not a file')
