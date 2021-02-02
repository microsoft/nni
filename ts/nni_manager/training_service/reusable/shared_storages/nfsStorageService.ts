// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as cpp from 'child-process-promise';
import * as path from 'path';

import { SharedStorageService, SharedStorageConfig, SharedStorageType, LocalMountedType } from '../sharedStorage'
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
    public localMounted: LocalMountedType;

    constructor(storageType: SharedStorageType, localMountPoint: string, remoteMountPoint: string,
                nfsServer: string, exportedDirectory: string, localMounted: LocalMountedType) {
        this.storageType = storageType;
        this.localMountPoint = localMountPoint;
        this.remoteMountPoint = remoteMountPoint;
        this.nfsServer = nfsServer;
        this.exportedDirectory = exportedDirectory;
        this.localMounted = localMounted;
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
            if (nfsConfig.localMounted === 'nnimount') {
                await this.helpLocalMount();
            } else if (nfsConfig.localMounted === 'nomount') {
                const errorMessage = `${this.storageType} Shared Storage:  ${this.storageType} not Support 'nomount'.`;
                this.log.error(errorMessage);
                return Promise.reject(errorMessage);
            }

            this.internalStorageService.initialize(this.localMountPoint, path.join(this.localMountPoint, 'nni', this.experimentId));
        }
        return Promise.resolve();
    }

    public get canLocalMounted(): boolean{
        return true;
    }

    public get storageService(): MountedStorageService {
        return this.internalStorageService;
    }

    public get localMountCommand(): string {
        return `sudo apt-get update && sudo apt-get -y install nfs-common && mkdir -p ${this.localMountPoint} && sudo mount ${this.nfsServer}:${this.exportedDirectory} ${this.localMountPoint}`;
    }

    public get remoteMountCommand(): string {
        return `sudo apt-get update && sudo apt-get -y install nfs-common && mkdir -p ${this.remoteMountPoint} && sudo mount ${this.nfsServer}:${this.exportedDirectory} ${this.remoteMountPoint}`;
    }

    public get localWorkingRoot(): string {
        return `${this.localMountPoint}/nni/${this.experimentId}`;
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
            const result = await cpp.exec(this.localMountCommand);
            if (result.stderr) {
                throw new Error(result.stderr);
            }
        } catch (error) {
            const errorMessage: string = `${this.storageType} Shared Storage: Mount ${this.nfsServer}:${this.exportedDirectory} to ${this.localMountPoint} failed, error is ${error}`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }

        return Promise.resolve();
    }
}
