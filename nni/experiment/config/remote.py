# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import List, Optional, Union

from .base import ConfigBase, PathLike
from .common import TrainingServiceConfig

__all__ = ['RemoteConfig', 'RemoteMachineConfig']


@dataclass(init=False)
class RemoteMachineConfig(ConfigBase):
    host: str
    port: int = 22
    user: str
    password: Optional[str] = None
    ssh_key_file: PathLike = '~/.ssh/id_rsa'
    ssh_passphrase: Optional[str] = None
    use_active_gpu: bool
    max_trial_number_per_gpu: int = 1
    gpu_indices: Optional[List[int], str] = None
    trial_prepare_command: Optional[Union[str, List[str]]] = None

    _training_service: str = 'remote'

    _field_validation_rules = [
        ('port', lambda val, _: val > 0),
        ('ssh_key_file', lambda val, config: (config.password is not None or Path(val).is_file(), '{val} is invalid')),
        ('max_trial_number_per_gpu', lambda: val, _: val > 0),
        (
            'gpu_indices',
            lambda val, _: isinstance(val, str) or (all(i >= 0 for i in val) and len(val) == len(set(val)))
        )
    ]


@dataclass(init=False)
class RemoteConfig(TrainingServiceConfig):
    machine_list: List<RemoteMachineConfig>
