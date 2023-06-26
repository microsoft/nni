// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import cpp from 'child-process-promise';
import fs from 'fs';
import path from 'path';
import { KubeflowConfig, toMegaBytes } from 'common/experimentConfig';
import { ExperimentStartupInfo } from 'common/experimentStartupInfo';
import { EnvironmentInformation } from 'training_service/reusable/environment';
import { KubernetesEnvironmentService } from './kubernetesEnvironmentService';
import { KubeflowOperatorClientFactory } from 'training_service/kubernetes/kubeflow/kubeflowApiClient';
import { KubeflowClusterConfigAzure } from 'training_service/kubernetes/kubeflow/kubeflowConfig';
import { KeyVaultConfig, AzureStorage } from 'training_service/kubernetes/kubernetesConfig';

export class KubeflowEnvironmentService extends KubernetesEnvironmentService {

    private config: KubeflowConfig;
    private createStoragePromise?: Promise<void>;
    

    constructor(config: KubeflowConfig, info: ExperimentStartupInfo) {
        super(config, info);
        this.experimentId = info.experimentId;
        this.config = config;
        // Create kubernetesCRDClient
        this.kubernetesCRDClient = KubeflowOperatorClientFactory.createClient(
            this.config.operator, this.config.apiVersion);
        this.kubernetesCRDClient.namespace = this.config.namespace ?? "default";
        // Create storage
        if (this.config.storage.storageType === 'azureStorage') {
            if (this.config.storage.azureShare === undefined ||
                this.config.storage.azureAccount === undefined ||
                this.config.storage.keyVaultName === undefined ||
                this.config.storage.keyVaultKey === undefined) {
                throw new Error("Azure storage configuration error!");
            }

            const azureStorage: AzureStorage = new AzureStorage(this.config.storage.azureShare, this.config.storage.azureAccount);
            const keyValutConfig: KeyVaultConfig = new KeyVaultConfig(this.config.storage.keyVaultName, this.config.storage.keyVaultKey);
            const azureKubeflowClusterConfig: KubeflowClusterConfigAzure = new KubeflowClusterConfigAzure(
                this.config.operator, this.config.apiVersion, keyValutConfig, azureStorage);
            this.azureStorageAccountName = azureKubeflowClusterConfig.azureStorage.accountName;
            this.azureStorageShare = azureKubeflowClusterConfig.azureStorage.azureShare;
            this.genericK8sClient.setNamespace = this.config.namespace ?? "default";
            this.createStoragePromise = this.createAzureStorage(
                azureKubeflowClusterConfig.keyVault.vaultName,
                azureKubeflowClusterConfig.keyVault.name
            );
        } else if (this.config.storage.storageType === 'nfs') {
            if (this.config.storage.server === undefined ||
                this.config.storage.path === undefined) {
                    throw new Error("NFS storage configuration error!");
                }
            this.createStoragePromise = this.createNFSStorage(
                this.config.storage.server,
                this.config.storage.path
            );
        }
    }

    public get environmentMaintenceLoopInterval(): number {
        return 5000;
    }

    public get hasStorageService(): boolean {
        return false;
    }

    public get getName(): string {
        return 'kubeflow';
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
        if (this.kubernetesCRDClient === undefined) {
            throw new Error("kubernetesCRDClient not initialized!");
        }
        if (this.createStoragePromise) {
            await this.createStoragePromise;
        }
        const expFolder = `${this.CONTAINER_MOUNT_PATH}/nni/${this.experimentId}`;
        environment.command = `cd ${expFolder} && ${environment.command} \
1>${expFolder}/envs/${environment.id}/trialrunner_stdout 2>${expFolder}/envs/${environment.id}/trialrunner_stderr`;
        environment.maxTrialNumberPerGpu = this.config.maxTrialNumberPerGpu;

        const kubeflowJobName: string = `nniexp${this.experimentId}env${environment.id}`.toLowerCase();
        
        await fs.promises.writeFile(path.join(this.environmentLocalTempFolder, `${environment.id}_run.sh`), environment.command, { encoding: 'utf8' });

        //upload script files to sotrage
        const trialJobOutputUrl: string = await this.uploadFolder(this.environmentLocalTempFolder, `nni/${this.experimentId}`);
        environment.trackingUrl = trialJobOutputUrl;
        // Generate kubeflow job resource config object
        const kubeflowJobConfig: any = await this.prepareKubeflowConfig(environment.id, kubeflowJobName);
        // Create kubeflow job based on generated kubeflow job resource config
        await this.kubernetesCRDClient.createKubernetesJob(kubeflowJobConfig);
    }

    /**
     * upload local folder to nfs or azureStroage
     */
    private async uploadFolder(srcDirectory: string, destDirectory: string): Promise<string> {
        if (this.config.storage.storageType === 'azureStorage') {
            if (this.azureStorageClient === undefined) {
                throw new Error('azureStorageClient is not initialized');
            }
            return await this.uploadFolderToAzureStorage(srcDirectory, destDirectory, 2);
        } else {
            try {
                // copy envs and run.sh from environments-temp to nfs-root(mounted)
                await cpp.exec(`mkdir -p ${this.nfsRootDir}/${destDirectory}`);
                await cpp.exec(`cp -r ${srcDirectory}/* ${this.nfsRootDir}/${destDirectory}`);
            } catch (uploadError) {
                return Promise.reject(uploadError);
            }
            return `nfs://${this.config.storage.server}:${destDirectory}`;
        }
    }

    private async prepareKubeflowConfig(envId: string, kubeflowJobName: string): Promise<any> {
        const workerPodResources: any = {};
        if (this.config.worker !== undefined) {
            workerPodResources.requests = this.generatePodResource(toMegaBytes(this.config.worker.memorySize),
                                                                   this.config.worker.cpuNumber, this.config.worker.gpuNumber);
        }
        workerPodResources.limits = {...workerPodResources.requests};

        const nonWorkerResources: any = {};
        if (this.config.operator === 'tf-operator') {
            if (this.config.ps !== undefined) {
                nonWorkerResources.requests = this.generatePodResource(toMegaBytes(this.config.ps.memorySize),
                                                                       this.config.ps.cpuNumber, this.config.ps.gpuNumber);
                nonWorkerResources.limits = {...nonWorkerResources.requests};
            }
        } else if (this.config.operator === 'pytorch-operator') {
            if (this.config.master !== undefined) {
                nonWorkerResources.requests = this.generatePodResource(toMegaBytes(this.config.master.memorySize),
                                                                       this.config.master.cpuNumber, this.config.master.gpuNumber);
                nonWorkerResources.limits = {...nonWorkerResources.requests};
            }
        }

        // Generate kubeflow job resource config object
        const kubeflowJobConfig: any = await this.generateKubeflowJobConfig(envId, kubeflowJobName, workerPodResources, nonWorkerResources);

        return Promise.resolve(kubeflowJobConfig);
    }

    /**
     * Generate kubeflow resource config file
     * @param kubeflowJobName job name
     * @param workerPodResources worker pod template
     * @param nonWorkerPodResources non-worker pod template, like ps or master
     */
    private async generateKubeflowJobConfig(envId: string, kubeflowJobName: string, workerPodResources: any,
                                            nonWorkerPodResources?: any): Promise<any> {

        if (this.kubernetesCRDClient === undefined) {
            throw new Error('Kubeflow operator client is not initialized');
        }

        const replicaSpecsObj: any = {};
        const replicaSpecsObjMap: Map<string, object> = new Map<string, object>();
        if (this.config.operator === 'tf-operator') {
            if (this.config.worker) {
                const privateRegistrySecretName = await this.createRegistrySecret(this.config.worker.privateRegistryAuthPath);
                replicaSpecsObj.Worker = this.generateReplicaConfig(this.config.worker.replicas,
                                                                    this.config.worker.dockerImage, 
                                                                    `${envId}_run.sh`, workerPodResources, privateRegistrySecretName);
            }
            if (this.config.ps !== undefined) {
                const privateRegistrySecretName: string | undefined = await this.createRegistrySecret(this.config.ps.privateRegistryAuthPath);
                replicaSpecsObj.Ps = this.generateReplicaConfig(this.config.ps.replicas,
                                                                this.config.ps.dockerImage,
                                                                `${envId}_run.sh`, nonWorkerPodResources, privateRegistrySecretName);
            }
            replicaSpecsObjMap.set(this.kubernetesCRDClient.jobKind, {tfReplicaSpecs: replicaSpecsObj});
        } else if (this.config.operator === 'pytorch-operator') {
            if (this.config.worker !== undefined) {
                const privateRegistrySecretName: string | undefined = await this.createRegistrySecret(this.config.worker.privateRegistryAuthPath);
                replicaSpecsObj.Worker = this.generateReplicaConfig(this.config.worker.replicas,
                                                                    this.config.worker.dockerImage, `${envId}_run.sh`, workerPodResources, privateRegistrySecretName);
            }
            if (this.config.master !== undefined) {
                const privateRegistrySecretName: string | undefined = await this.createRegistrySecret(this.config.master.privateRegistryAuthPath);
                replicaSpecsObj.Master = this.generateReplicaConfig(this.config.master.replicas,
                                                                    this.config.master.dockerImage, `${envId}_run.sh`, nonWorkerPodResources, privateRegistrySecretName);
    
            }
            replicaSpecsObjMap.set(this.kubernetesCRDClient.jobKind, {pytorchReplicaSpecs: replicaSpecsObj});
        }

        return Promise.resolve({
            apiVersion: `kubeflow.org/${this.kubernetesCRDClient.apiVersion}`,
            kind: this.kubernetesCRDClient.jobKind,
            metadata: {
                name: kubeflowJobName,
                namespace: this.kubernetesCRDClient.namespace,
                labels: {
                    app: this.NNI_KUBERNETES_TRIAL_LABEL,
                    expId: this.experimentId,
                    envId: envId
                }
            },
            spec: replicaSpecsObjMap.get(this.kubernetesCRDClient.jobKind)
        });
    }

    /**
     * Generate tf-operator's tfjobs replica config section
     * @param replicaNumber replica number
     * @param replicaImage image
     * @param runScriptFile script file name
     * @param podResources pod resource config section
     */
    private generateReplicaConfig(replicaNumber: number, replicaImage: string, runScriptFile: string,
                                  podResources: any, privateRegistrySecretName: string | undefined): any {
        if (this.kubernetesCRDClient === undefined) {
            throw new Error('Kubeflow operator client is not initialized');
        }
        // The config spec for volume field
        const volumeSpecMap: Map<string, object> = new Map<string, object>();
        if (this.config.storage.storageType === 'azureStorage') {
            volumeSpecMap.set('nniVolumes', [
            {
                    name: 'nni-vol',
                    azureFile: {
                        secretName: `${this.azureStorageSecretName}`,
                        shareName: `${this.azureStorageShare}`,
                        readonly: false
                    }
            }]);
        } else {
            volumeSpecMap.set('nniVolumes', [
            {
                name: 'nni-vol',
                nfs: {
                    server: `${this.config.storage.server}`,
                    path: `${this.config.storage.path}`
                }
            }]);
        }
        // The config spec for container field
        const containersSpecMap: Map<string, object> = new Map<string, object>();
        containersSpecMap.set('containers', [
        {
                // Kubeflow tensorflow operator requires that containers' name must be tensorflow
                // TODO: change the name based on operator's type
                name: this.kubernetesCRDClient.containerName,
                image: replicaImage,
                args: ['sh', `${path.join(this.environmentWorkingFolder, runScriptFile)}`],
                volumeMounts: [
                {
                    name: 'nni-vol',
                    mountPath: this.CONTAINER_MOUNT_PATH
                }],
                resources: podResources
            }
        ]);
        const spec: any = {
            containers: containersSpecMap.get('containers'),
            restartPolicy: 'ExitCode',
            volumes: volumeSpecMap.get('nniVolumes')
        }
        if (privateRegistrySecretName) {
            spec.imagePullSecrets = [
                {
                    name: privateRegistrySecretName
                }]
        }
        return {
            replicas: replicaNumber,
            template: {
                metadata: {
                    creationTimestamp: null
                },
                spec: spec
            }
        }
    }
}
