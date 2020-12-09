# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .base import PathLike
from .common import TrainingServiceConfig
from . import util

__all__ = ['OpenPaiConfig']

@dataclass(init=False)
class OpenPaiConfig(TrainingServiceConfig):
    platform: str = 'openpai'
    host: str
    username: str
    token: str
    docker_image: str = 'msranni/nni:latest'
    local_storage_mount_point: PathLike
    container_storage_mount_point: str
    reuse_mode: bool = False

    open_pai_config: Optional[Dict[str, Any]]
    open_pai_config_file: Optional[PathLike]

    _canonical_rules = {
        'host': lambda value: value.split('://', 1)[1] if '://' in value else value,  # type: ignore
        'local_storage_mount_point': util.canonical_path,
        'open_pai_config_file': util.canonical_path
    }

    _validation_rules = {
        'platform': lambda value: (value == 'openpai', 'cannot be modified'),
        'local_storage_mount_point': lambda value: Path(value).is_dir(),
        'container_storage_mount_point': lambda value: Path(value).is_absolute(),
        'open_pai_config_file': lambda value: Path(value).is_file()
    }

    def validate(self) -> None:
        super().validate()
        if self.open_pai_config is not None and self.open_pai_config_file is not None:
            raise ValueError('open_pai_config and open_pai_config_file can only be set one')
