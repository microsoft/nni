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

const INSTALL_NFS_CLIENT = `
#!/bin/bash
if [ -n "$(command -v nfsstat)" ]
then
    exit 0
fi

if [ -n "$(command -v apt-get)" ]
then
    sudo apt-get update
    sudo apt-get install -y nfs-common
elif [ -n "$(command -v yum)" ]
then
    sudo yum install -y nfs-utils
elif [ -n "$(command -v dnf)" ]
then
    sudo dnf install -y nfs-utils
else
    echo "Unknown package management."
    exit 1
fi
`

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
    private localMounted?: string;

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
            this.localMounted = nfsConfig.localMounted;
            if (this.localMounted === 'nnimount') {
                await this.helpLocalMount();
            } else if (this.localMounted === 'nomount') {
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
        if (this.localMountPoint) {
            return this.getCommand(this.localMountPoint);
        } else {
            this.log.error(`${this.storageType} Shared Storage: localMountPoint is not initialized.`);
            return '';
        }
    }

    public get remoteMountCommand(): string {
        if (this.remoteMountPoint) {
            return this.getCommand(this.remoteMountPoint);
        } else {
            this.log.error(`${this.storageType} Shared Storage: remoteMountPoint is not initialized.`);
            return '';
        }
    }

    public get remoteUmountCommand(): string {
        if (this.remoteMountPoint) {
            return `sudo umount -f -l ${this.remoteMountPoint}`;
        } else {
            this.log.error(`${this.storageType} Shared Storage: remoteMountPoint is not initialized.`);
            return '';
        }
    }

    private getCommand(mountPoint: string): string {
        const install = `rm -f nni_install_nfsclient.sh && touch nni_install_nfsclient.sh && echo "${INSTALL_NFS_CLIENT.replace(/\$/g, `\\$`).replace(/\n/g, `\\n`).replace(/"/g, `\\"`)}" >> nni_install_nfsclient.sh && bash nni_install_nfsclient.sh`;
        const mount = `mkdir -p ${mountPoint} && sudo mount ${this.nfsServer}:${this.exportedDirectory} ${mountPoint}`;
        const clean = `rm -f nni_install_nfsclient.sh`;
        return `${install} && ${mount} && ${clean}`;
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

    public async cleanUp(): Promise<void> {
        if (this.localMounted !== 'nnimount') {
            return Promise.resolve();
        }
        try {
            const result = await cpp.exec(`sudo umount -f -l ${this.localMountPoint}`);
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
