// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import cpp from 'child-process-promise';
import path from 'path';
import azureStorage from 'azure-storage';
import {Base64} from 'js-base64';
import {String} from 'typescript-string-operations';
import { ExperimentStartupInfo } from 'common/experimentStartupInfo';
import { getLogger, Logger } from 'common/log';
import { EnvironmentInformation, EnvironmentService } from 'training_service/reusable/environment';
import {GeneralK8sClient, KubernetesCRDClient} from 'training_service/kubernetes/kubernetesApiClient';
import {AzureStorageClientUtility} from 'training_service/kubernetes/azureStorageClientUtils';
import { KubeflowJobStatus } from 'training_service/kubernetes/kubeflow/kubeflowConfig';
import {delay, uniqueString} from 'common/utils';
const fs = require('fs');

export class KubernetesEnvironmentService extends EnvironmentService {
    protected azureStorageClient?: azureStorage.FileService;
    protected azureStorageShare?: string;
    protected azureStorageSecretName?: string;
    protected azureStorageAccountName?: string;
    protected genericK8sClient: GeneralK8sClient;
    protected kubernetesCRDClient?: KubernetesCRDClient;
    protected experimentRootDir: string;
    protected experimentId: string;
    
    //  experiment root dir in NFS
    protected environmentLocalTempFolder: string;
    protected NNI_KUBERNETES_TRIAL_LABEL: string = 'nni-kubernetes-trial';
    protected CONTAINER_MOUNT_PATH: string;
    protected log: Logger = getLogger('KubernetesEnvironmentService');
    protected environmentWorkingFolder: string;
    protected nfsRootDir: string;

    constructor(_config: any, info: ExperimentStartupInfo) {
        super();
        this.CONTAINER_MOUNT_PATH = '/tmp/mount';
        this.genericK8sClient = new GeneralK8sClient();
        this.experimentRootDir = info.logDir;
        this.environmentLocalTempFolder = path.join(this.experimentRootDir, "environment-temp");
        this.nfsRootDir = path.join(this.experimentRootDir, "nfs-root");
        this.experimentId = info.experimentId;
        this.environmentWorkingFolder = path.join(this.CONTAINER_MOUNT_PATH, 'nni', this.experimentId);
    }

    public get environmentMaintenceLoopInterval(): number {
        return 5000;
    }

    public get hasStorageService(): boolean {
        return false;
    }

    public get getName(): string {
        return 'kubernetes';
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
            if (this.genericK8sClient === undefined) {
                throw new Error("genericK8sClient undefined!");
            }
            const namespace = this.genericK8sClient.getNamespace ?? "default";
            await this.genericK8sClient.createSecret(
                {
                    apiVersion: 'v1',
                    kind: 'Secret',
                    metadata: {
                        name: this.azureStorageSecretName,
                        namespace: namespace,
                        labels: {
                            app: this.NNI_KUBERNETES_TRIAL_LABEL,
                            expId: this.experimentId
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
        if (uploadRetryCount) {
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

    protected async createNFSStorage(nfsServer: string, nfsPath: string): Promise<void> {
        await cpp.exec(`mkdir -p ${this.nfsRootDir}`);
        try {
            await cpp.exec(`sudo mount ${nfsServer}:${nfsPath} ${this.nfsRootDir}`);
        } catch (error) {
            const mountError: string = `Mount NFS ${nfsServer}:${nfsPath} to ${this.nfsRootDir} failed, error is ${error}`;
            this.log.error(mountError);

            return Promise.reject(mountError);
        }

        return Promise.resolve();
    }
    protected async createPVCStorage(pvcPath: string): Promise<void> {
        try {
            await cpp.exec(`mkdir -p ${pvcPath}`);
            await cpp.exec(`sudo ln -s ${pvcPath} ${this.environmentLocalTempFolder}`);
        } catch (error) {
            const linkError: string = `Linking ${pvcPath} to ${this.environmentLocalTempFolder} failed, error is ${error}`;
            this.log.error(linkError);

            return Promise.reject(linkError);
        }

        return Promise.resolve();
    }

    protected async createRegistrySecret(filePath: string | undefined): Promise<string | undefined> {
        if (filePath === undefined || filePath === '') {
            return undefined;
        }
        const body = fs.readFileSync(filePath).toString('base64');
        const registrySecretName = String.Format('nni-secret-{0}', uniqueString(8)
            .toLowerCase());
        const namespace = this.genericK8sClient.getNamespace ?? "default";
        await this.genericK8sClient.createSecret(
            {
                apiVersion: 'v1',
                kind: 'Secret',
                metadata: {
                    name: registrySecretName,
                    namespace: namespace,
                    labels: {
                        app: this.NNI_KUBERNETES_TRIAL_LABEL,
                        expId: this.experimentId
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


    public async refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void> {
        environments.forEach(async (environment) => {
            if (this.kubernetesCRDClient === undefined) {
                throw new Error("kubernetesCRDClient undefined")
            }
            const kubeflowJobName: string = `nniexp${this.experimentId}env${environment.id}`.toLowerCase();
            const kubernetesJobInfo = await this.kubernetesCRDClient.getKubernetesJob(kubeflowJobName);
            if (kubernetesJobInfo.status && kubernetesJobInfo.status.conditions) {
                const latestCondition: any = kubernetesJobInfo.status.conditions[kubernetesJobInfo.status.conditions.length - 1];
                const tfJobType: KubeflowJobStatus = <KubeflowJobStatus>latestCondition.type;
                switch (tfJobType) {
                    case 'Created':
                        environment.setStatus('WAITING');
                        break;
                    case 'Running':
                        environment.setStatus('RUNNING');
                        break;
                    case 'Failed':
                        environment.setStatus('FAILED');
                        break;
                    case  'Succeeded':
                        environment.setStatus('SUCCEEDED');
                        break;
                    default:
                }
            }
        });
    }

    public async startEnvironment(_environment: EnvironmentInformation): Promise<void> {
        throw new Error("Not implemented");
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        if (this.kubernetesCRDClient === undefined) {
            throw new Error('kubernetesCRDClient not initialized!');
        }
        try {
            await this.kubernetesCRDClient.deleteKubernetesJob(new Map(
                [
                    ['app', this.NNI_KUBERNETES_TRIAL_LABEL],
                    ['expId', this.experimentId],
                    ['envId', environment.id]
                ]
            ));
        } catch (err) {
            const errorMessage: string = `Delete env ${environment.id} failed: ${err}`;
            this.log.error(errorMessage);

            return Promise.reject(errorMessage);
        }
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
}
