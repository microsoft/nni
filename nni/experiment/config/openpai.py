# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Literal

from .common import TrainingServiceConfig
from . import util

__all__ = ['OpenPaiTrainingServiceConfig']

@dataclass(init=False)
class OpenPaiTrainingServiceConfig(TrainingServiceConfig):
    platform: Literal['openpai'] = 'openpai'
    host: str
    username: str
    token: str
    trial_cpu_number: int = 1
    trial_memory_size: str
    docker_image: str = 'msranni/nni:latest'
    reuse_mode: bool = False

    _training_service: str = 'openpai'

    _canonical_rules = {
        'host': lambda value: value.split('://', 1)[1] if '://' in value else value,
        'trial_memory_size': lambda value: str(util.parse_size(value)) + 'mb'
    }

    _validation_rules = {
        'trial_cpu_number': lambda value: value > 0,
        'trial_memory_size': lambda value: util.parse_size(value) > 0
    }
