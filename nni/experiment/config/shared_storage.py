# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Optional

from .common import SharedStorageConfig

__all__ = ['NfsConfig', 'AzureBlobConfig']

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
    resource_group_name: Optional[str] = None
    container_name: str
