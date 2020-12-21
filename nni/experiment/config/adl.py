# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass

from .common import TrainingServiceConfig

__all__ = ['AdlConfig']

@dataclass(init=False)
class AdlConfig(TrainingServiceConfig):
    platform: str = 'adl'
    docker_image: str = 'msranni/nni:latest'

    _validation_rules = {
        'platform': lambda value: (value == 'adl', 'cannot be modified')
    }
