# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Optional, Union

from .base import ConfigBase
from .common import TrainingServiceConfig

__all__ = ['KubeflowConfig', 'KubeflowRoleConfig', 'KubeflowNfsConfig', 'KubeflowAzureStorageConfig']


@dataclass(init=False)
class KubeflowNfsConfig(ConfigBase):
    storage: str = 'nfs'
    server: str
    path: str

@dataclass(init=False)
class KubeflowAzureStorageConfig(ConfigBase):
    storage: str = 'azureStorage'
    azure_account: str
    azure_share: str
    key_vault: str
    key_vault_secret: str


@dataclass(init=False)
class KubeflowRoleConfig(ConfigBase):
    replicas: int
    command: str
    gpu_number: int
    cpu_number: int
    memory_size: str
    docker_image: str = 'msranni/nni:latest'


@dataclass(init=False)
class KubeflowConfig(TrainingServiceConfig):
    platform: str = 'kubeflow'
    operator: str
    api_version: str
    storage: Union[KubeflowNfsConfig, KubeflowAzureStorageConfig]
    worker: KubeflowRoleConfig
    parameter_server: Optional[KubeflowRoleConfig] = None

    _validation_rules = {
        'platform': lambda value: (value == 'kubeflow', 'cannot be modified'),
        'operator': lambda value: value in ['tf-operator', 'pytorch-operator']
    }
