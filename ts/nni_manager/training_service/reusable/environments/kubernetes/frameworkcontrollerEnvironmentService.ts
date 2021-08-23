// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as path from 'path';
import * as component from '../../../../common/component';
import { ExperimentConfig, FrameworkControllerConfig, flattenConfig, FrameworkControllerTaskRoleConfig } from '../../../../common/experimentConfig';
import { ExperimentStartupInfo } from '../../../../common/experimentStartupInfo';
import { EnvironmentInformation } from '../../environment';
import { KubernetesEnvironmentService } from './kubernetesEnvironmentService';
import { FrameworkControllerClientFactory } from '../../../kubernetes/frameworkcontroller/frameworkcontrollerApiClient';
import { FrameworkControllerClusterConfigAzure, FrameworkControllerClusterConfig, FrameworkControllerClusterConfigFactory,
    FrameworkControllerClusterConfigNFS, FrameworkControllerTrialConfig, FrameworkControllerTrialConfigTemplate,
    FrameworkControllerClusterConfigPVC} from '../../../kubernetes/frameworkcontroller/frameworkcontrollerConfig';
import { KeyVaultConfig, AzureStorage } from '../../../kubernetes/kubernetesConfig';

interface FlattenKubeflowConfig extends ExperimentConfig, FrameworkControllerConfig { }

@component.Singleton
export class FrameworkControllerEnvironmentService extends KubernetesEnvironmentService {

    private config: FlattenKubeflowConfig;
    private createStoragePromise?: Promise<void>;
    private readonly fcContainerPortMap: Map<string, number> = new Map<string, number>(); // store frameworkcontroller container port
    

    constructor(config: ExperimentConfig, info: ExperimentStartupInfo) {
        super(config, info);
        this.experimentId = info.experimentId;
        this.config = flattenConfig(config, 'frameworkcontroller');
        // Create kubernetesCRDClient
        this.kubernetesCRDClient = FrameworkControllerClientFactory.createClient(
            this.config.namespace);
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
            const azureKubeflowClusterConfig: FrameworkControllerClusterConfigAzure = new FrameworkControllerClusterConfigAzure(
                this.config.namespace, this.config.apiVersion, keyValutConfig, azureStorage);
            this.azureStorageAccountName = azureKubeflowClusterConfig.azureStorage.accountName;
            this.azureStorageShare = azureKubeflowClusterConfig.azureStorage.azureShare;
            
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
        return 'frameworkcontroller';
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
        if (this.kubernetesCRDClient === undefined) {
            throw new Error("kubernetesCRDClient not initialized!");
        }
        if (this.createStoragePromise) {
            await this.createStoragePromise;
        }
        let configTaskRoles: any = undefined;
        configTaskRoles = this.config.taskRoles;
        //Generate the port used for taskRole
        this.generateContainerPort(configTaskRoles);

        const expFolder = `${this.CONTAINER_MOUNT_PATH}/nni/${this.experimentId}`;
        environment.command = `cd ${expFolder} && ${environment.command} \
        1>${expFolder}/envs/${environment.id}/trialrunner_stdout 2>${expFolder}/envs/${environment.id}/trialrunner_stderr`;
        if (this.config.deprecated && this.config.deprecated.useActiveGpu !== undefined) {
            environment.useActiveGpu = this.config.deprecated.useActiveGpu;
        }
        environment.maxTrialNumberPerGpu = this.config.maxTrialNumberPerGpu;

        const frameworkcontrollerJobName: string = `nni-exp-${this.experimentId}-env-${environment.id}`.toLowerCase();
        let command = this.generateCommandScript(this.config.taskRoles, environment.command);
        await fs.promises.writeFile(path.join(this.environmentLocalTempFolder, "run.sh"), command, { encoding: 'utf8' });

        //upload script files to sotrage
        const trialJobOutputUrl: string = await this.uploadFolder(this.environmentLocalTempFolder, `nni/${this.experimentId}`);
        environment.trackingUrl = trialJobOutputUrl;
        // Generate kubeflow job resource config object
        const frameworkcontrollerJobConfig: any = await this.prepareFrameworkControllerConfig(
            environment.id,
            environment.workingFolder,
            frameworkcontrollerJobName
        );
        // Create kubeflow job based on generated kubeflow job resource config
        await this.kubernetesCRDClient.createKubernetesJob(frameworkcontrollerJobConfig);
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
            // do not need to upload files to nfs server, temp folder already mounted to nfs
            return `nfs://${this.config.storage.server}:${destDirectory}`;
        }
    }

    /**
     * generate trial's command for frameworkcontroller
     * expose port and execute injector.sh before executing user's command
     * @param command
     */
    private generateCommandScript(taskRoles: FrameworkControllerTaskRoleConfig[], command: string): string {
        let portScript: string = '';
        for (const taskRole of taskRoles) {
            portScript += `FB_${taskRole.name.toUpperCase()}_PORT=${this.fcContainerPortMap.get(
                taskRole.name
            )} `;
        }
        return `${portScript} . /mnt/frameworkbarrier/injector.sh && ${command}`;
    }
    
    private async prepareFrameworkControllerConfig(trialJobId: string, trialWorkingFolder: string, frameworkcontrollerJobName: string):
            Promise<any> {
        const podResources: any = [];
        for (const taskRole of this.config.taskRoles) {
            const resource: any = {};
            resource.requests = this.generatePodResource(taskRole.memorySize, taskRole.cpuNumber, taskRole.gpuNumber);
            resource.limits = {...resource.requests};
            podResources.push(resource);
        }
        // Generate frameworkcontroller job resource config object
        const frameworkcontrollerJobConfig: any =
            await this.generateFrameworkControllerJobConfig(trialJobId, trialWorkingFolder, frameworkcontrollerJobName, podResources);

        return Promise.resolve(frameworkcontrollerJobConfig);
    }
    
        private generateContainerPort(taskRoles: FrameworkControllerTrialConfigTemplate[]): void {
            if (taskRoles === undefined) {
                throw new Error('frameworkcontroller trial config is not initialized');
            }
    
            let port: number = 4000; //The default port used in container
            for (const index of taskRoles.keys()) {
                this.fcContainerPortMap.set(taskRoles[index].name, port);
                port += 1;
            }
        }
    
        /**
         * Generate frameworkcontroller resource config file
         * @param trialJobId trial job id
         * @param trialWorkingFolder working folder
         * @param frameworkcontrollerJobName job name
         * @param podResources  pod template
         */
        private async generateFrameworkControllerJobConfig(trialJobId: string, trialWorkingFolder: string,
            frameworkcontrollerJobName: string, podResources: any): Promise<any> {
    
            const taskRoles: any = [];
            for (const index of this.config.taskRoles.keys()) {
                const containerPort: number | undefined = this.fcContainerPortMap.get(this.config.taskRoles[index].name);
                if (containerPort === undefined) {
                    throw new Error('Container port is not initialized');
                }
    
                const taskRole: any = this.generateTaskRoleConfig(
                    trialWorkingFolder,
                    this.config.taskRoles[index].dockerImage,
                    `run_${this.config.taskRoles[index].name}.sh`,
                    podResources[index],
                    containerPort,
                    await this.createRegistrySecret(this.config.taskRoles[index].privateRegistryAuthPath)
                );
                taskRoles.push({
                    name: this.config.taskRoles[index].name,
                    taskNumber: this.config.taskRoles[index].taskNumber,
                    frameworkAttemptCompletionPolicy: {
                        minFailedTaskCount: this.config.taskRoles[index].frameworkAttemptCompletionPolicy.minFailedTaskCount,
                        minSucceededTaskCount: this.config.taskRoles[index].frameworkAttemptCompletionPolicy.minSucceedTaskCount
                    },
                    task: taskRole
                });
            }
    
            return Promise.resolve({
                apiVersion: `frameworkcontroller.microsoft.com/v1`,
                kind: 'Framework',
                metadata: {
                    name: frameworkcontrollerJobName,
                    namespace: this.config.namespace ? this.config.namespace : "default",
                    labels: {
                        app: this.NNI_KUBERNETES_TRIAL_LABEL,
                        expId: this.experimentId,
                        trialId: trialJobId
                    }
                },
                spec: {
                    executionType: 'Start',
                    taskRoles: taskRoles
                }
            });
        }
    
        private generateTaskRoleConfig(trialWorkingFolder: string, replicaImage: string, runScriptFile: string,
            podResources: any, containerPort: number, privateRegistrySecretName: string | undefined): any {
    
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
                    }, {
                        name: 'frameworkbarrier-volume',
                        emptyDir: {}
                    }]);
            } else {
                volumeSpecMap.set('nniVolumes', [
                    {
                        name: 'nni-vol',
                        nfs: {
                            server: `${this.config.storage.server}`,
                            path: `${this.config.storage.path}`
                        }
                    }, {
                        name: 'frameworkbarrier-volume',
                        emptyDir: {}
                    }]);
            }
    
            const containers: any = [
                {
                    name: 'framework',
                    image: replicaImage,
                    command: ['sh', `${path.join(trialWorkingFolder, runScriptFile)}`],
                    volumeMounts: [
                        {
                            name: 'nni-vol',
                            mountPath: this.CONTAINER_MOUNT_PATH
                        }, {
                            name: 'frameworkbarrier-volume',
                            mountPath: '/mnt/frameworkbarrier'
                        }],
                    resources: podResources,
                    ports: [{
                        containerPort: containerPort
                    }]
                }];
    
            const initContainers: any = [
                {
                    name: 'frameworkbarrier',
                    image: 'frameworkcontroller/frameworkbarrier',
                    volumeMounts: [
                        {
                            name: 'frameworkbarrier-volume',
                            mountPath: '/mnt/frameworkbarrier'
                        }]
                }];
    
            const spec: any = {
                containers: containers,
                initContainers: initContainers,
                restartPolicy: 'OnFailure',
                volumes: volumeSpecMap.get('nniVolumes'),
                hostNetwork: false
            };
            if (privateRegistrySecretName) {
                spec.imagePullSecrets = [
                    {
                        name: privateRegistrySecretName
                    }
                ]
            }
    
            if (this.config.serviceAccountName !== undefined) {
                spec.serviceAccountName = this.config.serviceAccountName;
            }
    
            return {
                pod: {
                    spec: spec
                }
            };
        }
}
