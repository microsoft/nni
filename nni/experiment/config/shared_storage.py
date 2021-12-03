# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Optional

from .base import ConfigBase
from .utils import PathLike

__all__ = ['NfsConfig', 'AzureBlobConfig']

@dataclass(init=False)
class SharedStorageConfig(ConfigBase):
    storage_type: str
    local_mount_point: PathLike
    remote_mount_point: str
    local_mounted: str
    storage_account_name: Optional[str] = None
    storage_account_key: Optional[str] = None
    container_name: Optional[str] = None
    nfs_server: Optional[str] = None
    exported_directory: Optional[str] = None

@dataclass(init=False)
class NfsConfig(SharedStorageConfig):
    storage_type: str = 'NFS'
    nfs_server: str
    exported_directory: str

@dataclass(init=False)
class AzureBlobConfig(SharedStorageConfig):
    storage_type: str = 'AzureBlob'
    storage_account_name: str
    storage_account_key: Optional[str] = None
    container_name: str
