# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Configuration for FrameworkController training service.

Check the reference_ for explaination of each field.

You may also want to check `FrameworkController training service doc`_.

.. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html

.. _FrameworkController training service doc: https://nni.readthedocs.io/en/stable/TrainingService/FrameworkControllerMode.html

"""

__all__ = ['FrameworkControllerConfig', 'FrameworkControllerRoleConfig', 'FrameworkAttemptCompletionPolicy']

from dataclasses import dataclass
from typing import List, Optional, Union

from ..base import ConfigBase
from ..training_service import TrainingServiceConfig
from .k8s_storage import K8sStorageConfig

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
    memory_size: Union[str, int]
    framework_attempt_completion_policy: FrameworkAttemptCompletionPolicy

@dataclass(init=False)
class FrameworkControllerConfig(TrainingServiceConfig):
    platform: str = 'frameworkcontroller'
    storage: K8sStorageConfig
    service_account_name: Optional[str]
    task_roles: List[FrameworkControllerRoleConfig]
    reuse_mode: Optional[bool] = True

    def _canonicalize(self, parents):
        super()._canonicalize(parents)
        # framework controller does not need these fields, set empty string for type check
        if self.trial_command is None:
            self.trial_command = ''
