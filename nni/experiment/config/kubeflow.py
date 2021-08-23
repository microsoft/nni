# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Optional

from .base import ConfigBase
from .common import TrainingServiceConfig
from . import util

__all__ = ['KubeflowConfig', 'KubeflowRoleConfig', 'KubeflowStorageConfig', 'KubeflowNfsConfig', 'KubeflowAzureStorageConfig']


@dataclass(init=False)
class KubeflowStorageConfig(ConfigBase):
    storage_type: str
    server: Optional[str] = None
    path: Optional[str] = None
    azure_account: Optional[str] = None
    azure_share: Optional[str] = None
    key_vault_name: Optional[str] = None
    key_vault_key: Optional[str] = None

@dataclass(init=False)
class KubeflowNfsConfig(KubeflowStorageConfig):
    storage: str = 'nfs'
    server: str
    path: str

@dataclass(init=False)
class KubeflowAzureStorageConfig(ConfigBase):
    storage: str = 'azureStorage'
    azure_account: str
    azure_share: str
    key_vault_name: str
    key_vault_key: str


@dataclass(init=False)
class KubeflowRoleConfig(ConfigBase):
    replicas: int
    command: str
    gpu_number: Optional[int] = 0
    cpu_number: int
    memory_size: str
    docker_image: str = 'msranni/nni:latest'
    code_directory: str


@dataclass(init=False)
class KubeflowConfig(TrainingServiceConfig):
    platform: str = 'kubeflow'
    operator: str
    api_version: str
    storage: KubeflowStorageConfig
    worker: Optional[KubeflowRoleConfig] = None
    ps: Optional[KubeflowRoleConfig] = None
    master: Optional[KubeflowRoleConfig] = None
    reuse_mode: Optional[bool] = True #set reuse mode as true for v2 config

    def __init__(self, **kwargs):
        kwargs = util.case_insensitive(kwargs)
        kwargs['storage'] = util.load_config(KubeflowStorageConfig, kwargs.get('storage'))
        kwargs['worker'] = util.load_config(KubeflowRoleConfig, kwargs.get('worker'))
        kwargs['ps'] = util.load_config(KubeflowRoleConfig, kwargs.get('ps'))
        kwargs['master'] = util.load_config(KubeflowRoleConfig, kwargs.get('master'))
        super().__init__(**kwargs)

    _validation_rules = {
        'platform': lambda value: (value == 'kubeflow', 'cannot be modified'),
        'operator': lambda value: value in ['tf-operator', 'pytorch-operator']
    }