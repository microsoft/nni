# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .base import ConfigBase, PathLike
from .common import TrainingServiceConfig
from . import util

__all__ = ['RemoteConfig', 'RemoteMachineConfig']

@dataclass(init=False)
class RemoteMachineConfig(ConfigBase):
    host: str
    port: int = 22
    user: str
    password: Optional[str] = None
    ssh_key_file: Optional[PathLike] = None
    ssh_passphrase: Optional[str] = None
    use_active_gpu: bool = False
    max_trial_number_per_gpu: int = 1
    gpu_indices: Optional[Union[List[int], str]] = None
    trial_prepare_command: Optional[str] = None

    _canonical_rules = {
        'ssh_key_file': util.canonical_path,
        'gpu_indices': lambda value: [int(idx) for idx in value.split(',')] if isinstance(value, str) else value,
    }

    _validation_rules = {
        'port': lambda value: 0 < value < 65536,
        'max_trial_number_per_gpu': lambda value: value > 0,
        'gpu_indices': lambda value: all(idx >= 0 for idx in value) and len(value) == len(set(value))
    }

    def validate(self):
        super().validate()
        if self.password is None and not Path(self.ssh_key_file).is_file():
            raise ValueError(f'Password is not provided and cannot find SSH key file "{self.ssh_key_file}"')

@dataclass(init=False)
class RemoteConfig(TrainingServiceConfig):
    platform: str = 'remote'
    reuse_mode: bool = False
    machine_list: List[RemoteMachineConfig]

    def __init__(self, **kwargs):
        kwargs = util.case_insensitive(kwargs)
        kwargs['machinelist'] = util.load_config(RemoteMachineConfig, kwargs.get('machinelist'))
        super().__init__(**kwargs)

    _validation_rules = {
        'platform': lambda value: (value == 'remote', 'cannot be modified')
    }
