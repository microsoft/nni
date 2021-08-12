// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as cpp from 'child-process-promise';
import * as path from 'path';

import { SharedStorageService, SharedStorageType } from '../sharedStorage'
import { MountedStorageService } from '../storages/mountedStorageService';
import { getLogger, Logger } from '../../../common/log';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import { AzureBlobConfig } from '../../../common/experimentConfig';
import { getExperimentRootDir } from '../../../common/utils';

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

if [ "$id" = "Ubuntu" ]
then
    wget https://packages.microsoft.com/config/ubuntu/$version/packages-microsoft-prod.deb
    sudo DEBIAN_FRONTEND=noninteractive dpkg -i packages-microsoft-prod.deb
    sudo apt-get update
    sudo apt-get install -y blobfuse fuse
elif [ "$id" = "CentOS" ] || [ "$id" = "RHEL" ]
then
    sudo rpm -Uvh https://packages.microsoft.com/config/rhel/$(echo $version | cut -c1)/packages-microsoft-prod.rpm
    sudo yum install -y blobfuse fuse
else
    echo "Not support distributor."
    exit 1
fi
`

export class AzureBlobSharedStorageService extends SharedStorageService {
    private log: Logger;
    private internalStorageService: MountedStorageService;
    private experimentId: string;
    private localMounted?: string;

    private storageType?: SharedStorageType;
    private storageAccountName?: string;
    private storageAccountKey?: string;
    private containerName?: string;

    private localMountPoint?: string;
    private remoteMountPoint?: string;

    constructor() {
        super();
        this.log = getLogger('AzureBlobSharedStorageService');
        this.internalStorageService = new MountedStorageService();
        this.experimentId = getExperimentId();
    }

    public async config(azureblobConfig: AzureBlobConfig): Promise<void> {
        this.localMountPoint = azureblobConfig.localMountPoint;
        this.remoteMountPoint = azureblobConfig.remoteMountPoint;

        this.storageType = azureblobConfig.storageType as SharedStorageType;
        this.storageAccountName = azureblobConfig.storageAccountName;
        this.containerName = azureblobConfig.containerName;
        if (azureblobConfig.storageAccountKey !== undefined) {
            this.storageAccountKey = azureblobConfig.storageAccountKey;
        } else {
            const errorMessage = `${this.storageType} Shared Storage: must set 'storageAccountKey'.`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }
        this.localMounted = azureblobConfig.localMounted;
        if (this.localMounted === 'nnimount') {
            await this.helpLocalMount();
        } else if (this.localMounted === 'nomount') {
            const errorMessage = `${this.storageType} Shared Storage: ${this.storageType} not Support 'nomount' yet.`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }

        if (this.canLocalMounted && this.localMountPoint) {
            this.internalStorageService.initialize(this.localMountPoint, path.join(this.localMountPoint, 'nni', this.experimentId));
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

    public get remoteUmountCommand(): string {
        if (this.remoteMountPoint) {
            return `sudo umount -l ${this.remoteMountPoint}`;
        } else {
            this.log.error(`${this.storageType} Shared Storage: remoteMountPoint is not initialized.`);
            return '';
        }
    }

    private getCommand(mountPoint: string): string {
        let install_fuseblob_script_path = 'nni_install_fuseblob.sh';
        let fuseblob_config_path = 'nni_fuse_connection.cfg';
        if (mountPoint == this.localMountPoint) {
            const experimentRootDir = getExperimentRootDir();
            install_fuseblob_script_path = path.join(experimentRootDir, install_fuseblob_script_path);
            fuseblob_config_path = path.join(experimentRootDir, fuseblob_config_path);
        }

        const install = `rm -f ${install_fuseblob_script_path} && touch ${install_fuseblob_script_path} && echo "${INSTALL_BLOBFUSE.replace(/\$/g, `\\$`).replace(/\n/g, `\\n`).replace(/"/g, `\\"`)}" >> ${install_fuseblob_script_path} && bash ${install_fuseblob_script_path}`;
        const prepare = `sudo mkdir /mnt/resource/nniblobfusetmp -p && rm -f ${fuseblob_config_path} && touch ${fuseblob_config_path} && echo "accountName ${this.storageAccountName}\\naccountKey ${this.storageAccountKey}\\ncontainerName ${this.containerName}" >> ${fuseblob_config_path}`;
        const mount = `mkdir -p ${mountPoint} && sudo blobfuse ${mountPoint} --tmp-path=/mnt/resource/nniblobfusetmp  --config-file=${fuseblob_config_path} -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other`;
        const clean = `rm -f ${install_fuseblob_script_path} ${fuseblob_config_path}`;
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

    public async cleanUp(): Promise<void> {
        if (this.localMounted !== 'nnimount') {
            return Promise.resolve();
        }
        try {
            const result = await cpp.exec(`sudo umount -l ${this.localMountPoint}`);
            if (result.stderr) {
                throw new Error(result.stderr);
            }
        } catch (error) {
            const errorMessage: string = `${this.storageType} Shared Storage: Umount ${this.localMountPoint}  failed, error is ${error}`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }
        return Promise.resolve();
    }
}
