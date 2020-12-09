# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import List, Union

from .base import ConfigBase
from .common import TrainingServiceConfig

__all__ = [
    'FrameworkControllerConfig',
    'FrameworkControllerRoleConfig',
    'FrameworkControllerNfsConfig',
    'FrameworkControllerAzureStorageConfig'
]


@dataclass(init=False)
class FrameworkControllerNfsConfig(ConfigBase):
    storage: str = 'nfs'
    server: str
    path: str

@dataclass(init=False)
class FrameworkControllerAzureStorageConfig(ConfigBase):
    storage: str = 'azureStorage'
    azure_account: str
    azure_share: str
    key_vault: str
    key_vault_secret: str


@dataclass(init=False)
class FrameworkControllerRoleConfig(ConfigBase):
    name: str
    docker_image: str = 'msranni/nni:latest'
    task_number: int
    command: str
    gpu_number: int
    cpu_number: int
    memory_size: str
    attempt_completion_min_failed_tasks: int
    attempt_completion_min_succeeded_tasks: int


@dataclass(init=False)
class FrameworkControllerConfig(TrainingServiceConfig):
    platform: str = 'frameworkcontroller'
    service_account_name: str
    storage: Union[FrameworkControllerNfsConfig, FrameworkControllerAzureStorageConfig]
    task_roles: List[FrameworkControllerRoleConfig]

    _validation_rules = {
        'platform': lambda value: (value == 'frameworkcontroller', 'cannot be modified')
    }
