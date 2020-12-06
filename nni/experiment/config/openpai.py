# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass

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
    nni_manager_storage_mount_point: PathLike
    countainer_storage_mount_point: Annotated[PathLike, 'Absolute']
    reuse_mode: bool = False

    open_pai_config: Optional[Dict[str, Any]]
    open_pai_config_file: Optional[PathLike]

    _canonical_rules = {
        'host': lambda value: value.split('://', 1)[1] if '://' in value else value,
        'nni_manager_nfs_mount_point': util.canonical_path,
        'countainer_nfs_mount_point': lambda value: str(value),
        'open_pai_config_file': util.canonical_path
    }

    _validation_rules = {
        'platform': lambda value: (value == 'openpai', 'cannot be modified'),
        'nni_manager_nfs_mount_point': lambda value: Path(value).is_dir(),
        'open_pai_config_file': lambda value: Path(value).is_file()
    }

    def validate(self) -> None:
        super().validate()
        if open_pai_config is not None and open_pai_config_file is not None:
            raise ValueError('open_pai_config and open_pai_config_file can only be set one')
