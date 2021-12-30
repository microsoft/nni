# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Configuration for AML training service.

Check the reference_ for explaination of each field.

You may also want to check `AML training service doc`_.

.. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html

.. _AML training service doc: https://nni.readthedocs.io/en/stable/TrainingService/AMLMode.html

"""

__all__ = ['AmlConfig']

from dataclasses import dataclass

from ..training_service import TrainingServiceConfig

@dataclass(init=False)
class AmlConfig(TrainingServiceConfig):
    platform: str = 'aml'
    subscription_id: str
    resource_group: str
    workspace_name: str
    compute_target: str
    docker_image: str = 'msranni/nni:latest'
    max_trial_number_per_gpu: int = 1
