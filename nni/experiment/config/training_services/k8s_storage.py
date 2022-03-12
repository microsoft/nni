# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Storage config classes for ``KubeflowConfig`` and ``FrameworkControllerConfig``
"""

__all__ = ['K8sStorageConfig', 'K8sAzureStorageConfig', 'K8sNfsConfig']

from dataclasses import dataclass
from typing import Optional

from ..base import ConfigBase

@dataclass(init=False)
class K8sStorageConfig(ConfigBase):
    storage_type: str
    azure_account: Optional[str] = None
    azure_share: Optional[str] = None
    key_vault_name: Optional[str] = None
    key_vault_key: Optional[str] = None
    server: Optional[str] = None
    path: Optional[str] = None

    def _validate_canonical(self):
        super()._validate_canonical()
        if self.storage_type == 'azureStorage':
            assert self.server is None and self.path is None
        elif self.storage_type == 'nfs':
            assert self.azure_account is None and self.azure_share is None
            assert self.key_vault_name is None and self.key_vault_key is None
        else:
            raise ValueError(f'Kubernetes storage_type ("{self.storage_type}") must either be "azureStorage" or "nfs"')

@dataclass(init=False)
class K8sNfsConfig(K8sStorageConfig):
    storage: str = 'nfs'
    server: str
    path: str

@dataclass(init=False)
class K8sAzureStorageConfig(K8sStorageConfig):
    storage: str = 'azureStorage'
    azure_account: str
    azure_share: str
    key_vault_name: str
    key_vault_key: str
