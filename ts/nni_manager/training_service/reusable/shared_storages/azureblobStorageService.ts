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

const INSTALL_BLOBFUSE = `
#!/bin/bash
if [ -n "$(command -v blobfuse)" ]
then
    exit 0
fi

if [ -n "$(command -v apt-get)" ]
then
    sudo apt-get update
    sudo apt-get install -y lsb-release
elif [ -n "$(command -v yum)" ]
then
    sudo yum install -y redhat-lsb
else
    echo "Unknown package management."
    exit 1
fi

id=$(lsb_release -i | cut -c16- | sed s/[[:space:]]//g)
version=$(lsb_release -r | cut -c9- | sed s/[[:space:]]//g)

if [ $id = "Ubuntu" ]
then
    wget https://packages.microsoft.com/config/ubuntu/$version/packages-microsoft-prod.deb
    sudo dpkg -i packages-microsoft-prod.deb
    sudo apt-get update
    sudo apt-get install -y blobfuse fuse
elif [ $id = "CentOS" ] || [ $id = "RHEL" ]
then
    sudo rpm -Uvh https://packages.microsoft.com/config/rhel/$(echo $version | cut -c1)/packages-microsoft-prod.rpm
    sudo yum install -y blobfuse fuse
else
    echo "Not support distributor."
    exit 1
fi
`

class AzureBlobSharedStorageConfig implements SharedStorageConfig {
    public storageType: SharedStorageType;
    public localMountPoint?: string;
    public remoteMountPoint: string;

    public resourceGroupName?: string;
    public storageAccountName: string;
    public storageAccountKey?: string;
    public containerName: string;

    public localMounted: LocalMountedType;

    constructor(storageType: SharedStorageType,  remoteMountPoint: string, storageAccountName: string, containerName: string,
                localMounted: LocalMountedType, localMountPoint?: string, resourceGroupName?: string, storageAccountKey?: string) {
        this.storageType = storageType;
        this.localMountPoint = localMountPoint;
        this.remoteMountPoint = remoteMountPoint;
        this.resourceGroupName = resourceGroupName;
        this.storageAccountName = storageAccountName;
        this.storageAccountKey = storageAccountKey;
        this.containerName = containerName;
        this.localMounted = localMounted;
    }
}

export class AzureBlobSharedStorageService extends SharedStorageService {
    private log: Logger;
    private internalStorageService: MountedStorageService;
    private experimentId: string;

    private storageType?: SharedStorageType;
    private storageAccountName?: string;
    private storageAccountKey?: string;
    private containerName?: string;

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
            const azureblobConfig = <AzureBlobSharedStorageConfig>JSON.parse(value);
            this.localMountPoint = azureblobConfig.localMountPoint;
            this.remoteMountPoint = azureblobConfig.remoteMountPoint;

            this.storageType = azureblobConfig.storageType;
            this.storageAccountName = azureblobConfig.storageAccountName;
            this.containerName = azureblobConfig.containerName;
            if (azureblobConfig.storageAccountKey !== undefined) {
                this.storageAccountKey =azureblobConfig.storageAccountKey;
            } else if (azureblobConfig.resourceGroupName !== undefined) {
                await this.setAccountKey(azureblobConfig.resourceGroupName);
            } else {
                const errorMessage = `${this.storageType} Shared Storage: must set one of 'storageAccountKey' or 'resourceGroupName'.`;
                this.log.error(errorMessage);
                return Promise.reject(errorMessage);
            }

            if (azureblobConfig.localMounted === 'nnimount') {
                await this.helpLocalMount();
            } else if (azureblobConfig.localMounted === 'nomount') {
                const errorMessage = `${this.storageType} Shared Storage: ${this.storageType} not Support 'nomount' yet.`;
                this.log.error(errorMessage);
                return Promise.reject(errorMessage);
            }

            if (this.canLocalMounted && this.localMountPoint) {
                this.internalStorageService.initialize(this.localMountPoint, path.join(this.localMountPoint, 'nni', this.experimentId));
            }
        }
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

    private getCommand(mountPoint: string): string {
        const install = `rm -f nni_install_fuseblob.sh && touch nni_install_fuseblob.sh && echo "${INSTALL_BLOBFUSE.replace(/\$/g, `\\$`).replace(/\n/g, `\\n`).replace(/"/g, `\\"`)}" >> nni_install_fuseblob.sh && bash nni_install_fuseblob.sh`;
        const prepare = `sudo mkdir /mnt/resource/nniblobfusetmp -p && rm -f nni_fuse_connection.cfg && touch nni_fuse_connection.cfg && echo "accountName ${this.storageAccountName}\\naccountKey ${this.storageAccountKey}\\ncontainerName ${this.containerName}" >> nni_fuse_connection.cfg`;
        const mount = `mkdir -p ${mountPoint} && sudo blobfuse ${mountPoint} --tmp-path=/mnt/resource/nniblobfusetmp  --config-file=$(pwd)/nni_fuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other`;
        const clean = `rm -f nni_install_fuseblob.sh nni_fuse_connection.cfg`;
        return `${install} && ${prepare} && ${mount} && ${clean}`;
    }

    public get localWorkingRoot(): string {
        return `${this.localMountPoint}/nni/${this.experimentId}`;
    }

    public get remoteWorkingRoot(): string {
        return `${this.remoteMountPoint}/nni/${this.experimentId}`;
    }

    private async helpLocalMount(): Promise<void> {
        if (process.platform === 'win32') {
            const errorMessage = `${this.storageType} Shared Storage: ${this.storageType} do not support mount under Windows yet.`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }

        try {
            this.log.debug(`Local mount command is: ${this.localMountCommand}`);
            const result = await cpp.exec(this.localMountCommand);
            if (result.stderr) {
                throw new Error(result.stderr);
            }
        } catch (error) {
            const errorMessage: string = `${this.storageType} Shared Storage: Mount ${this.storageAccountName}/${this.containerName} to ${this.localMountPoint} failed, error is ${error}`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }

        return Promise.resolve();
    }

    private async setAccountKey(resourceGroupName: string): Promise<void> {
        try {
            const result = await cpp.exec(`az storage account keys list --resource-group ${resourceGroupName} --account-name ${this.storageAccountName} --query "[0].value" | tr -d '"'`);
            if (result.stderr) {
                throw Error(result.stderr);
            } else {
                this.storageAccountKey = result.stdout.trim();
            }
        } catch (error) {
            const errorMessage: string = `${this.storageType} Shared Storage: get account key failed, error is ${error}`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }
    }
}
