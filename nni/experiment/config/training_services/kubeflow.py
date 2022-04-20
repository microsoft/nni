# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Configuration for Kubeflow training service.

Check the reference_ for explaination of each field.

You may also want to check `Kubeflow training service doc`_.

.. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html

.. _Kubeflow training service doc: https://nni.readthedocs.io/en/stable/TrainingService/KubeflowMode.html

"""

__all__ = ['KubeflowConfig', 'KubeflowRoleConfig']

from dataclasses import dataclass
from typing import Optional, Union

from ..base import ConfigBase
from ..training_service import TrainingServiceConfig
from .k8s_storage import K8sStorageConfig

@dataclass(init=False)
class KubeflowRoleConfig(ConfigBase):
    replicas: int
    command: str
    gpu_number: Optional[int] = 0
    cpu_number: int
    memory_size: Union[str, int]
    docker_image: str = 'msranni/nni:latest'
    code_directory: str

@dataclass(init=False)
class KubeflowConfig(TrainingServiceConfig):
    platform: str = 'kubeflow'
    operator: str
    api_version: str
    storage: K8sStorageConfig
    worker: Optional[KubeflowRoleConfig] = None
    ps: Optional[KubeflowRoleConfig] = None
    master: Optional[KubeflowRoleConfig] = None
    reuse_mode: Optional[bool] = True #set reuse mode as true for v2 config

    def _canonicalize(self, parents):
        super()._canonicalize(parents)
        # kubeflow does not need these fields, set empty string for type check
        if self.trial_command is None:
            self.trial_command = ''
        if self.trial_code_directory is None:
            self.trial_code_directory = ''

    def _validate_canonical(self):
        super()._validate_canonical()
        assert self.operator in ['tf-operator', 'pytorch-operator']
