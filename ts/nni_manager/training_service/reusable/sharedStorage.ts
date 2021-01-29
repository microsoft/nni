// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { StorageService } from './storageService'

export type SharedStorageType = 'NFS'

export interface SharedStorageConfig {
    readonly storageType: SharedStorageType;
    readonly localMountPoint: string;
    readonly remoteMountPoint: string;
}

export abstract class SharedStorageService {
    public abstract config(key: string, value: string): Promise<void>;
    public abstract get storageService(): StorageService;
    public abstract get localMountCommand(): string;
    public abstract get remoteMountCommand(): string;
    public abstract get localWorkingRoot(): string;
    public abstract get remoteWorkingRoot(): string;
}
