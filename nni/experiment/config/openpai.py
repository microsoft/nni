# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .base import PathLike
from .common import TrainingServiceConfig
from . import util

__all__ = ['OpenpaiConfig']

@dataclass(init=False)
class OpenpaiConfig(TrainingServiceConfig):
    platform: str = 'openpai'
    host: str
    username: str
    token: str
    docker_image: str = 'msranni/nni:latest'
    local_storage_mount_point: PathLike
    container_storage_mount_point: str
    reuse_mode: bool = False

    openpai_config: Optional[Dict[str, Any]] = None
    openpai_config_file: Optional[PathLike] = None

    _canonical_rules = {
        'host': lambda value: 'https://' + value if '://' not in value else value,  # type: ignore
        'local_storage_mount_point': util.canonical_path,
        'openpai_config_file': util.canonical_path
    }

    _validation_rules = {
        'platform': lambda value: (value == 'openpai', 'cannot be modified'),
        'local_storage_mount_point': lambda value: Path(value).is_dir(),
        'container_storage_mount_point': lambda value: (Path(value).is_absolute(), 'is not absolute'),
        'openpai_config_file': lambda value: Path(value).is_file()
    }

    def validate(self) -> None:
        super().validate()
        if self.openpai_config is not None and self.openpai_config_file is not None:
            raise ValueError('openpai_config and openpai_config_file can only be set one')
