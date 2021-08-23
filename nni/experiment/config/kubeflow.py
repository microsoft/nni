# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Optional

from .base import ConfigBase
from .common import TrainingServiceConfig
from . import util

__all__ = ['KubeflowConfig', 'KubeflowRoleConfig', 'KubeflowNfsConfig', 'KubeflowAzureStorageConfig']


@dataclass(init=False)
class _KubeflowStorageConfig(ConfigBase):
    storage: str
    server: Optional[str] = None
    path: Optional[str] = None
    azure_account: Optional[str] = None
    azure_share: Optional[str] = None
    key_vault: Optional[str] = None
    key_vault_secret: Optional[str] = None

@dataclass(init=False)
class KubeflowNfsConfig(_KubeflowStorageConfig):
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
    storage: _KubeflowStorageConfig
    worker: KubeflowRoleConfig
    parameter_server: Optional[KubeflowRoleConfig] = None

    def __init__(self, **kwargs):
        kwargs = util.case_insensitve(kwargs)
        kwargs['storage'] = util.load_config(_KubeflowStorageConfig, kwargs.get('storage'))
        kwargs['worker'] = util.load_config(KubeflowRoleConfig, kwargs.get('worker'))
        kwargs['parameterserver'] = util.load_config(KubeflowRoleConfig, kwargs.get('parameterserver'))
        super().__init__(**kwargs)

    _validation_rules = {
        'platform': lambda value: (value == 'kubeflow', 'cannot be modified'),
        'operator': lambda value: value in ['tf-operator', 'pytorch-operator']
    }
