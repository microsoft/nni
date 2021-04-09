// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { SharedStorageConfig } from '../../common/experimentConfig';
import { StorageService } from './storageService'

export type SharedStorageType = 'NFS' | 'AzureBlob'
export type LocalMountedType = 'usermount' | 'nnimount' | 'nomount'

export abstract class SharedStorageService {
    public abstract config(config: SharedStorageConfig): Promise<void>;
    public abstract get canLocalMounted(): boolean;
    public abstract get storageService(): StorageService;
    public abstract get localMountCommand(): string;
    public abstract get remoteMountCommand(): string;
    public abstract get localWorkingRoot(): string;
    public abstract get remoteWorkingRoot(): string;
}
