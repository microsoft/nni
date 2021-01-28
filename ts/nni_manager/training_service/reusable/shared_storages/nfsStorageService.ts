// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as cpp from 'child-process-promise';
import * as path from 'path';

import { SharedStorageService, SharedStorageConfig, SharedStorageType } from '../sharedStorage'
import { MountedStorageService } from '../storages/mountedStorageService';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { getLogger, Logger } from '../../../common/log';
import { getExperimentId } from '../../../common/experimentStartupInfo';

class NFSSharedStorageConfig implements SharedStorageConfig {
    public storageType: SharedStorageType;
    public localMountPoint: string;
    public remoteMountPoint: string;

    public nfsServer: string;
    public exportedDirectory: string;
    public userMounted: boolean;

    constructor(storageType: SharedStorageType, localMountPoint: string, remoteMountPoint: string,
                nfsServer: string, exportedDirectory: string, userMounted: boolean) {
        this.storageType = storageType;
        this.localMountPoint = localMountPoint;
        this.remoteMountPoint = remoteMountPoint;
        this.nfsServer = nfsServer;
        this.exportedDirectory = exportedDirectory;
        this.userMounted = userMounted;
    }
}

export class NFSSharedStorageService extends SharedStorageService {
    private log: Logger;
    private internalStorageService: MountedStorageService;
    private experimentId: string;

    private storageType?: SharedStorageType;
    private nfsServer?: string;
    private exportedDirectory?: string;

    private localMountPoint?: string;
    private remoteMountPoint?: string;

    constructor() {
        super();
        this.log = getLogger();
        this.internalStorageService = new MountedStorageService();
        this.experimentId = getExperimentId();
    }

    public async config(key: string, value: string): Promise<void> {
        if (key === TrialConfigMetadataKey.SHARED_STORAGE_CONFIG) {
            const nfsConfig = <NFSSharedStorageConfig>JSON.parse(value);
            this.localMountPoint = nfsConfig.localMountPoint;
            this.remoteMountPoint = nfsConfig.remoteMountPoint;

            this.storageType = nfsConfig.storageType;
            this.nfsServer = nfsConfig.nfsServer;
            this.exportedDirectory = nfsConfig.exportedDirectory;
            if ( nfsConfig.userMounted === false ) {
                await this.helpLocalMount();
            }

            this.internalStorageService.initialize(this.localMountPoint, path.join(this.localMountPoint, 'nni', this.experimentId));
        }
    }

    public get storageService(): MountedStorageService {
        return this.internalStorageService;
    }

    public get localMountCommand(): string {
        return `mkdir -p ${this.localMountPoint} && sudo mount ${this.nfsServer}:${this.exportedDirectory} ${this.localMountPoint}`;
    }

    public get remoteMountCommand(): string {
        return `mkdir -p ${this.remoteMountPoint} && sudo mount ${this.nfsServer}:${this.exportedDirectory} ${this.remoteMountPoint}`;
    }

    public get remoteWorkingRoot(): string {
        return `${this.remoteMountPoint}/nni/${this.experimentId}`;
    }

    private async helpLocalMount(): Promise<void> {
        if (process.platform === 'win32') {
            const errorMessage = `${this.storageType} Shared Storage: NNI not support auto mount ${this.storageType} under Windows yet.`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }

        try {
            await cpp.exec(this.localMountCommand);
        } catch (error) {
            const errorMessage: string = `${this.storageType} Shared Storage: Mount ${this.nfsServer}:${this.exportedDirectory} to ${this.localMountPoint} failed, error is ${error}`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }

        return Promise.resolve();
    }
}
