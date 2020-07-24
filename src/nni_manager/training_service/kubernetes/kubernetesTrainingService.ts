// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as cpp from 'child-process-promise';
import * as path from 'path';

import * as azureStorage from 'azure-storage';
import { EventEmitter } from 'events';
import { Base64 } from 'js-base64';
import { String } from 'typescript-string-operations';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import { MethodNotImplementedError } from '../../common/errors';
import {
    NNIManagerIpConfig, TrialJobDetail, TrialJobMetric, LogType
} from '../../common/trainingService';
import { delay, getExperimentRootDir, getIPV4Address, getJobCancelStatus, getVersion, uniqueString } from '../../common/utils';
import { AzureStorageClientUtility } from './azureStorageClientUtils';
import { GeneralK8sClient, KubernetesCRDClient } from './kubernetesApiClient';
import { KubernetesClusterConfig } from './kubernetesConfig';
import { kubernetesScriptFormat, KubernetesTrialJobDetail } from './kubernetesData';
import { KubernetesJobRestServer } from './kubernetesJobRestServer';

const fs = require('fs');

/**
 * Training Service implementation for Kubernetes
 */
abstract class KubernetesTrainingService {
    protected readonly NNI_KUBERNETES_TRIAL_LABEL: string = 'nni-kubernetes-trial';
    protected readonly log!: Logger;
    protected readonly metricsEmitter: EventEmitter;
    protected readonly trialJobsMap: Map<string, KubernetesTrialJobDetail>;
    //  experiment root dir in NFS
    protected readonly trialLocalNFSTempFolder: string;
    protected stopping: boolean = false;
    protected experimentId!: string;
    protected kubernetesRestServerPort?: number;
    protected readonly CONTAINER_MOUNT_PATH: string;
    protected azureStorageClient?: azureStorage.FileService;
    protected azureStorageShare?: string;
    protected azureStorageSecretName?: string;
    protected azureStorageAccountName?: string;
    protected nniManagerIpConfig?: NNIManagerIpConfig;
    protected readonly genericK8sClient: GeneralK8sClient;
    protected kubernetesCRDClient?: KubernetesCRDClient;
    protected kubernetesJobRestServer?: KubernetesJobRestServer;
    protected kubernetesClusterConfig?: KubernetesClusterConfig;
    protected versionCheck: boolean = true;
    protected logCollection: string;
    protected copyExpCodeDirPromise?: Promise<string>;
    protected expContainerCodeFolder: string;

    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, KubernetesTrialJobDetail>();
        this.trialLocalNFSTempFolder = path.join(getExperimentRootDir(), 'trials-nfs-tmp');
        this.experimentId = getExperimentId();
        this.CONTAINER_MOUNT_PATH = '/tmp/mount';
        this.expContainerCodeFolder = path.join(this.CONTAINER_MOUNT_PATH, 'nni', this.experimentId, 'nni-code');
        this.genericK8sClient = new GeneralK8sClient();
        this.logCollection = 'none';
    }

    public generatePodResource(memory: number, cpuNum: number, gpuNum: number): any {
        const resources: any = {
            memory: `${memory}Mi`,
            cpu: `${cpuNum}`
        };

        if (gpuNum !== 0) {
            resources['nvidia.com/gpu'] = `${gpuNum}`;
        }

        return resources;
    }

    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        const jobs: TrialJobDetail[] = [];

        for (const key of this.trialJobsMap.keys()) {
            jobs.push(await this.getTrialJob(key));
        }

        return Promise.resolve(jobs);
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {

        const kubernetesTrialJob: TrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (kubernetesTrialJob === undefined) {
            return Promise.reject(`trial job ${trialJobId} not found`);
        }

        return Promise.resolve(kubernetesTrialJob);
    }

    public async getTrialLog(_trialJobId: string, _logType: LogType): Promise<string> {
        throw new MethodNotImplementedError();
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.off('metric', listener);
    }

    public get isMultiPhaseJobSupported(): boolean {
        return false;
    }

    public getClusterMetadata(_key: string): Promise<string> {
        return Promise.resolve('');
    }

    public get MetricsEmitter(): EventEmitter {
        return this.metricsEmitter;
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trialJobDetail: KubernetesTrialJobDetail | undefined =  this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            const errorMessage: string = `CancelTrialJob: trial job id ${trialJobId} not found`;
            this.log.error(errorMessage);

            return Promise.reject(errorMessage);
        }
        if (this.kubernetesCRDClient === undefined) {
            const errorMessage: string = `CancelTrialJob: trial job id ${trialJobId} failed because operatorClient is undefined`;
            this.log.error(errorMessage);

            return Promise.reject(errorMessage);
        }

        try {
            await this.kubernetesCRDClient.deleteKubernetesJob(new Map(
                [
                    ['app', this.NNI_KUBERNETES_TRIAL_LABEL],
                    ['expId', getExperimentId()],
                    ['trialId', trialJobId]
                ]
            ));
        } catch (err) {
            const errorMessage: string = `Delete trial ${trialJobId} failed: ${err}`;
            this.log.error(errorMessage);

            return Promise.reject(errorMessage);
        }

        trialJobDetail.endTime = Date.now();
        trialJobDetail.status = getJobCancelStatus(isEarlyStopped);

        return Promise.resolve();
    }

    public async cleanUp(): Promise<void> {
        this.stopping = true;

        // First, cancel all running kubernetes jobs
        for (const [trialJobId, kubernetesTrialJob] of this.trialJobsMap) {
            if (['RUNNING', 'WAITING', 'UNKNOWN'].includes(kubernetesTrialJob.status)) {
                try {
                    await this.cancelTrialJob(trialJobId);
                } catch (error) {
                  // DONT throw error during cleanup
                }
                kubernetesTrialJob.status = 'SYS_CANCELED';
            }
        }

        // Delete all kubernetes jobs whose expId label is current experiment id
        try {
            if (this.kubernetesCRDClient !== undefined) {
                await this.kubernetesCRDClient.deleteKubernetesJob(new Map(
                    [
                        ['app', this.NNI_KUBERNETES_TRIAL_LABEL],
                        ['expId', getExperimentId()]
                    ]
                ));
            }
        } catch (error) {
            this.log.error(`Delete kubernetes job with label: app=${this.NNI_KUBERNETES_TRIAL_LABEL},\
            expId=${getExperimentId()} failed, error is ${error}`);
        }

        // Unmount NFS
        try {
            await cpp.exec(`sudo umount ${this.trialLocalNFSTempFolder}`);
        } catch (error) {
            this.log.error(`Unmount ${this.trialLocalNFSTempFolder} failed, error is ${error}`);
        }

        // Stop kubernetes rest server
        if (this.kubernetesJobRestServer === undefined) {
            throw new Error('kubernetesJobRestServer not initialized!');
        }
        try {
            await this.kubernetesJobRestServer.stop();
            this.log.info('Kubernetes Training service rest server stopped successfully.');
        } catch (error) {
            this.log.error(`Kubernetes Training service rest server stopped failed, error: ${error.message}`);

            return Promise.reject(error);
        }

        return Promise.resolve();
    }

    protected async createAzureStorage(vaultName: string, valutKeyName: string): Promise<void> {
        try {
            const result: any = await cpp.exec(`az keyvault secret show --name ${valutKeyName} --vault-name ${vaultName}`);
            if (result.stderr) {
                const errorMessage: string = result.stderr;
                this.log.error(errorMessage);

                return Promise.reject(errorMessage);
            }
            const storageAccountKey: any = JSON.parse(result.stdout).value;
            if (this.azureStorageAccountName === undefined) {
                throw new Error('azureStorageAccountName not initialized!');
            }
            //create storage client
            this.azureStorageClient = azureStorage.createFileService(this.azureStorageAccountName, storageAccountKey);
            await AzureStorageClientUtility.createShare(this.azureStorageClient, this.azureStorageShare);
            //create sotrage secret
            this.azureStorageSecretName = String.Format('nni-secret-{0}', uniqueString(8)
                                                                            .toLowerCase());
            await this.genericK8sClient.createSecret(
                {
                    apiVersion: 'v1',
                    kind: 'Secret',
                    metadata: {
                        name: this.azureStorageSecretName,
                        namespace: 'default',
                        labels: {
                            app: this.NNI_KUBERNETES_TRIAL_LABEL,
                            expId: getExperimentId()
                        }
                    },
                    type: 'Opaque',
                    data: {
                        azurestorageaccountname: Base64.encode(this.azureStorageAccountName),
                        azurestorageaccountkey: Base64.encode(storageAccountKey)
                    }
                }
            );
        } catch (error) {
            this.log.error(error);

            return Promise.reject(error);
        }

        return Promise.resolve();
    }

    /**
     * Genereate run script for different roles(like worker or ps)
     * @param trialJobId trial job id
     * @param trialWorkingFolder working folder
     * @param command command
     * @param trialSequenceId sequence id
     */
    protected async generateRunScript(platform: string, trialJobId: string, trialWorkingFolder: string,
                                      command: string, trialSequenceId: string, roleName: string, gpuNum: number): Promise<string> {
        let nvidiaScript: string = '';
        // Nvidia devcie plugin for K8S has a known issue that requesting zero GPUs allocates all GPUs
        // Refer https://github.com/NVIDIA/k8s-device-plugin/issues/61
        // So we have to explicitly set CUDA_VISIBLE_DEVICES to empty if user sets gpuNum to 0 in NNI config file
        if (gpuNum === 0) {
            nvidiaScript = 'export CUDA_VISIBLE_DEVICES=';
        }
        const nniManagerIp: string = this.nniManagerIpConfig ? this.nniManagerIpConfig.nniManagerIp : getIPV4Address();
        const version: string = this.versionCheck ? await getVersion() : '';
        const runScript: string = String.Format(
            kubernetesScriptFormat,
            platform,
            trialWorkingFolder,
            path.join(trialWorkingFolder, 'output', `${roleName}_output`),
            trialJobId,
            getExperimentId(),
            this.expContainerCodeFolder,
            trialSequenceId,
            nvidiaScript,
            command,
            nniManagerIp,
            this.kubernetesRestServerPort,
            version,
            this.logCollection
        );

        return Promise.resolve(runScript);
    }
    protected async createNFSStorage(nfsServer: string, nfsPath: string): Promise<void> {
        await cpp.exec(`mkdir -p ${this.trialLocalNFSTempFolder}`);
        try {
            await cpp.exec(`sudo mount ${nfsServer}:${nfsPath} ${this.trialLocalNFSTempFolder}`);
        } catch (error) {
            const mountError: string = `Mount NFS ${nfsServer}:${nfsPath} to ${this.trialLocalNFSTempFolder} failed, error is ${error}`;
            this.log.error(mountError);

            return Promise.reject(mountError);
        }

        return Promise.resolve();
    }

    protected async createRegistrySecret(filePath: string | undefined): Promise<string | undefined> {
        if(filePath === undefined || filePath === '') {
            return undefined;
        }
        const body = fs.readFileSync(filePath).toString('base64');
        const registrySecretName = String.Format('nni-secret-{0}', uniqueString(8)
                                                                            .toLowerCase());
        await this.genericK8sClient.createSecret(
            {
                apiVersion: 'v1',
                kind: 'Secret',
                metadata: {
                    name: registrySecretName,
                    namespace: 'default',
                    labels: {
                        app: this.NNI_KUBERNETES_TRIAL_LABEL,
                        expId: getExperimentId()
                    }
                },
                type: 'kubernetes.io/dockerconfigjson',
                data: {
                    '.dockerconfigjson': body
                }
            }
        );
        return registrySecretName;
    }
    
    /**
     * upload local directory to azureStorage
     * @param srcDirectory the source directory of local folder
     * @param destDirectory the target directory in azure
     * @param uploadRetryCount the retry time when upload failed
     */
    protected async uploadFolderToAzureStorage(srcDirectory: string, destDirectory: string, uploadRetryCount: number | undefined): Promise<string> {
        if (this.azureStorageClient === undefined) {
            throw new Error('azureStorageClient is not initialized');
        }
        let retryCount: number = 1;
        if(uploadRetryCount) {
            retryCount = uploadRetryCount;
        }
        let uploadSuccess: boolean = false;
        let folderUriInAzure = '';
        try {
            do {
                uploadSuccess = await AzureStorageClientUtility.uploadDirectory(
                    this.azureStorageClient,
                    `${destDirectory}`, 
                    this.azureStorageShare,
                    `${srcDirectory}`);
                if (!uploadSuccess) {
                    //wait for 5 seconds to re-upload files
                    await delay(5000);
                    this.log.info('Upload failed, Retry: upload files to azure-storage');
                } else {
                    folderUriInAzure = `https://${this.azureStorageAccountName}.file.core.windows.net/${this.azureStorageShare}/${destDirectory}`;
                    break;
                }
            } while (retryCount-- >= 0)
        } catch (error) {
            this.log.error(error);
            //return a empty url when got error
            return Promise.resolve('');
        }
        return Promise.resolve(folderUriInAzure);
    }
     
}
export { KubernetesTrainingService };
