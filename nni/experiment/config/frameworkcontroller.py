# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import List, Optional

from .base import ConfigBase
from .common import TrainingServiceConfig
from . import util

__all__ = [
    'FrameworkControllerConfig',
    'FrameworkControllerRoleConfig',
    '_FrameworkControllerStorageConfig'
]


@dataclass(init=False)
class _FrameworkControllerStorageConfig(ConfigBase):
    storage_type: str
    server: Optional[str] = None
    path: Optional[str] = None
    azure_account: Optional[str] = None
    azure_share: Optional[str] = None
    key_vault_name: Optional[str] = None
    key_vault_key: Optional[str] = None

@dataclass(init=False)
class FrameworkAttemptCompletionPolicy(ConfigBase):
    min_failed_task_count: int
    min_succeed_task_count: int

@dataclass(init=False)
class FrameworkControllerRoleConfig(ConfigBase):
    name: str
    docker_image: str = 'msranni/nni:latest'
    task_number: int
    command: str
    gpu_number: int
    cpu_number: int
    memory_size: str
    framework_attempt_completion_policy: FrameworkAttemptCompletionPolicy


@dataclass(init=False)
class FrameworkControllerConfig(TrainingServiceConfig):
    platform: str = 'frameworkcontroller'
    service_account_name: str
    storage: _FrameworkControllerStorageConfig
    task_roles: List[FrameworkControllerRoleConfig]
    reuse_mode: Optional[bool] = True #set reuse mode as true for v2 config
    service_account_name: Optional[str]

    def __init__(self, **kwargs):
        kwargs = util.case_insensitive(kwargs)
        kwargs['storage'] = util.load_config(_FrameworkControllerStorageConfig, kwargs.get('storage'))
        kwargs['taskroles'] = util.load_config(FrameworkControllerRoleConfig, kwargs.get('taskroles'))
        super().__init__(**kwargs)

    _validation_rules = {
        'platform': lambda value: (value == 'frameworkcontroller', 'cannot be modified')
    }
