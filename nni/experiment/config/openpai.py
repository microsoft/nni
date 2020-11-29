# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import List, Optional, Union

from .base import ConfigBase, PathLike
from .common import TrainingServiceConfig

__all__ = ['OpenPaiConfig']


@dataclass(init=False)
class OpenPaiConfig(TrainingServiceConfig):
    host: str
    user_name: str
    token: str
    trial_cpu_number: int = 1
    trial_memory: str
    docker_image: str = 'msranni/nni'
    docker_auth_file: Optional[PathLike] = None  # FIXME

    _training_service: str = 'openpai'

    _field_validation_rules = [
        ('trial_cpu_number', lambda val, _: val > 0),
        ('trial_memory', lambda val, _: unit.parse_size(val) > 0)
    ]
