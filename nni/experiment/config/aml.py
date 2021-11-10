# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass

from .common import TrainingServiceConfig

__all__ = ['AmlConfig']

@dataclass(init=False)
class AmlConfig(TrainingServiceConfig):
    platform: str = 'aml'
    subscription_id: str
    resource_group: str
    workspace_name: str
    compute_target: str
    docker_image: str = 'msranni/nni:latest'
    max_trial_number_per_gpu: int = 1

    _validation_rules = {
        'platform': lambda value: (value == 'aml', 'cannot be modified')
    }
