// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as path from 'path';
import * as component from '../../../common/component';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import {
    NNIManagerIpConfig, TrialJobApplicationForm, TrialJobDetail, TrialJobStatus
} from '../../../common/trainingService';
import { delay, generateParamFileName, getExperimentRootDir, uniqueString } from '../../../common/utils';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../../common/containerJobData';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { validateCodeDir } from '../../common/util';
import { NFSConfig } from '../kubernetesConfig';
import { KubernetesTrialJobDetail } from '../kubernetesData';
import { KubernetesTrainingService } from '../kubernetesTrainingService';
import { FrameworkControllerClientFactory } from './frameworkcontrollerApiClient';
import { FrameworkControllerClusterConfig, FrameworkControllerClusterConfigAzure, FrameworkControllerClusterConfigFactory,
    FrameworkControllerClusterConfigNFS, FrameworkControllerTrialConfig} from './frameworkcontrollerConfig';
import { FrameworkControllerJobInfoCollector } from './frameworkcontrollerJobInfoCollector';
import { FrameworkControllerJobRestServer } from './frameworkcontrollerJobRestServer';

/**
 * Training Service implementation for frameworkcontroller
 */
@component.Singleton
class FrameworkControllerTrainingService extends KubernetesTrainingService implements KubernetesTrainingService {
    private fcTrialConfig?: FrameworkControllerTrialConfig; // frameworkcontroller trial configuration
    private readonly fcJobInfoCollector: FrameworkControllerJobInfoCollector; // frameworkcontroller job info collector
    private readonly fcContainerPortMap: Map<string, number> = new Map<string, number>(); // store frameworkcontroller container port
    private fcClusterConfig?: FrameworkControllerClusterConfig;

    constructor() {
        super();
        this.fcJobInfoCollector = new FrameworkControllerJobInfoCollector(this.trialJobsMap);
        this.experimentId = getExperimentId();
    }

    public async run(): Promise<void> {
        this.kubernetesJobRestServer = component.get(FrameworkControllerJobRestServer);
        if (this.kubernetesJobRestServer === undefined) {
            throw new Error('kubernetesJobRestServer not initialized!');
        }
        await this.kubernetesJobRestServer.start();
        this.kubernetesJobRestServer.setEnableVersionCheck = this.versionCheck;
        this.log.info(`frameworkcontroller Training service rest server listening on: ${this.kubernetesJobRestServer.endPoint}`);
        while (!this.stopping) {
            // collect metrics for frameworkcontroller jobs by interacting with Kubernetes API server
            await delay(3000);
            await this.fcJobInfoCollector.retrieveTrialStatus(this.kubernetesCRDClient);
            if (this.kubernetesJobRestServer.getErrorMessage !== undefined) {
                throw new Error(this.kubernetesJobRestServer.getErrorMessage);
                this.stopping = true;
            }
        }
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        if (this.fcClusterConfig === undefined) {
            throw new Error('frameworkcontrollerClusterConfig is not initialized');
        }
        if (this.kubernetesCRDClient === undefined) {
            throw new Error('kubernetesCRDClient is undefined');
        }

        if (this.kubernetesRestServerPort === undefined) {
            const restServer: FrameworkControllerJobRestServer = component.get(FrameworkControllerJobRestServer);
            this.kubernetesRestServerPort = restServer.clusterRestServerPort;
        }

        // wait upload of code Dir to finish
        if (this.copyExpCodeDirPromise !== undefined) {
            await this.copyExpCodeDirPromise;
        }

        const trialJobId: string = uniqueString(5);
        // Set trial's NFS working folder
        const trialWorkingFolder: string = path.join(this.CONTAINER_MOUNT_PATH, 'nni', getExperimentId(), trialJobId);
        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        const frameworkcontrollerJobName: string = `nniexp${this.experimentId}trial${trialJobId}`.toLowerCase();
        //Generate the port used for taskRole
        this.generateContainerPort();
        await this.prepareRunScript(trialLocalTempFolder, trialJobId, trialWorkingFolder, form);

        //wait upload of script files to finish
        const trialJobOutputUrl: string = await this.uploadFolder(trialLocalTempFolder, `nni/${getExperimentId()}/${trialJobId}`);
        let initStatus: TrialJobStatus = 'WAITING';
        if (!trialJobOutputUrl) {
            initStatus = 'FAILED';
        }
        const trialJobDetail: KubernetesTrialJobDetail = new KubernetesTrialJobDetail(
            trialJobId,
            initStatus,
            Date.now(),
            trialWorkingFolder,
            form,
            frameworkcontrollerJobName,
            trialJobOutputUrl
        );

        // Set trial job detail until create frameworkcontroller job successfully
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        // Create frameworkcontroller job based on generated frameworkcontroller job resource config
        const frameworkcontrollerJobConfig: any = await this.prepareFrameworkControllerConfig(
            trialJobId, trialWorkingFolder, frameworkcontrollerJobName);
        await this.kubernetesCRDClient.createKubernetesJob(frameworkcontrollerJobConfig);

        // Set trial job detail until create frameworkcontroller job successfully
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;
            case TrialConfigMetadataKey.FRAMEWORKCONTROLLER_CLUSTER_CONFIG: {
                const frameworkcontrollerClusterJsonObject: any = JSON.parse(value);
                this.fcClusterConfig = FrameworkControllerClusterConfigFactory
                  .generateFrameworkControllerClusterConfig(frameworkcontrollerClusterJsonObject);
                if (this.fcClusterConfig.storageType === 'azureStorage') {
                    const azureFrameworkControllerClusterConfig: FrameworkControllerClusterConfigAzure =
                      <FrameworkControllerClusterConfigAzure>this.fcClusterConfig;
                    this.azureStorageAccountName = azureFrameworkControllerClusterConfig.azureStorage.accountName;
                    this.azureStorageShare = azureFrameworkControllerClusterConfig.azureStorage.azureShare;
                    await this.createAzureStorage(
                        azureFrameworkControllerClusterConfig.keyVault.vaultName,
                        azureFrameworkControllerClusterConfig.keyVault.name
                    );
                } else if (this.fcClusterConfig.storageType === 'nfs') {
                    const nfsFrameworkControllerClusterConfig: FrameworkControllerClusterConfigNFS =
                      <FrameworkControllerClusterConfigNFS>this.fcClusterConfig;
                    await this.createNFSStorage(
                        nfsFrameworkControllerClusterConfig.nfs.server,
                        nfsFrameworkControllerClusterConfig.nfs.path
                    );
                }
                this.kubernetesCRDClient = FrameworkControllerClientFactory.createClient();
                break;
            }
            case TrialConfigMetadataKey.TRIAL_CONFIG: {
                const frameworkcontrollerTrialJsonObjsect: any = JSON.parse(value);

                this.fcTrialConfig = new FrameworkControllerTrialConfig(
                    frameworkcontrollerTrialJsonObjsect.codeDir,
                    frameworkcontrollerTrialJsonObjsect.taskRoles
                );

                // Validate to make sure codeDir doesn't have too many files
                try {
                    await validateCodeDir(this.fcTrialConfig.codeDir);
                    //upload codeDir to storage
                    this.copyExpCodeDirPromise = this.uploadFolder(this.fcTrialConfig.codeDir, `nni/${getExperimentId()}/nni-code`);
                } catch (error) {
                    this.log.error(error);

                    return Promise.reject(new Error(error));
                }
                break;
            }
            case TrialConfigMetadataKey.VERSION_CHECK:
                this.versionCheck = (value === 'true' || value === 'True');
                break;
            case TrialConfigMetadataKey.LOG_COLLECTION:
                this.logCollection = value;
                break;
            default:
        }

        return Promise.resolve();
    }

    /**
     * upload local folder to nfs or azureStroage
     */
    private async uploadFolder(srcDirectory: string, destDirectory: string): Promise<string> {
        if (this.fcClusterConfig === undefined) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        assert(this.fcClusterConfig.storage === undefined
            || this.fcClusterConfig.storage === 'azureStorage'
            || this.fcClusterConfig.storage === 'nfs');

        if (this.fcClusterConfig.storage === 'azureStorage') {
            if (this.azureStorageClient === undefined) {
                throw new Error('azureStorageClient is not initialized');
            }
            const fcClusterConfigAzure: FrameworkControllerClusterConfigAzure = <FrameworkControllerClusterConfigAzure>this.fcClusterConfig;
            return await this.uploadFolderToAzureStorage(srcDirectory, destDirectory, fcClusterConfigAzure.uploadRetryCount);
        } else if (this.fcClusterConfig.storage === 'nfs' || this.fcClusterConfig.storage === undefined) {
            await cpp.exec(`mkdir -p ${this.trialLocalNFSTempFolder}/${destDirectory}`);
            await cpp.exec(`cp -r ${srcDirectory}/* ${this.trialLocalNFSTempFolder}/${destDirectory}/.`);
            const fcClusterConfigNFS: FrameworkControllerClusterConfigNFS = <FrameworkControllerClusterConfigNFS>this.fcClusterConfig;
            const nfsConfig: NFSConfig = fcClusterConfigNFS.nfs;
            return `nfs://${nfsConfig.server}:${destDirectory}`;
        }
        return '';
    }

    /**
     * generate trial's command for frameworkcontroller
     * expose port and execute injector.sh before executing user's command
     * @param command
     */
    private generateCommandScript(command: string): string {
        let portScript: string = '';
        if (this.fcTrialConfig === undefined) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }
        for (const taskRole of this.fcTrialConfig.taskRoles) {
            portScript += `FB_${taskRole.name.toUpperCase()}_PORT=${this.fcContainerPortMap.get(taskRole.name)} `;
        }

        return `${portScript} . /mnt/frameworkbarrier/injector.sh && ${command}`;
    }

    private async prepareRunScript(trialLocalTempFolder: string, trialJobId: string,
                                   trialWorkingFolder: string, form: TrialJobApplicationForm): Promise<void> {
        if (this.fcTrialConfig === undefined) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }

        await cpp.exec(`mkdir -p ${trialLocalTempFolder}`);
        
        const installScriptContent: string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), installScriptContent, { encoding: 'utf8' });
        // Create tmp trial working folder locally.

        for (const taskRole of this.fcTrialConfig.taskRoles) {
            const runScriptContent: string =
              await this.generateRunScript('frameworkcontroller', trialJobId, trialWorkingFolder,
                                           this.generateCommandScript(taskRole.command), form.sequenceId.toString(),
                                           taskRole.name, taskRole.gpuNum);
            await fs.promises.writeFile(path.join(trialLocalTempFolder, `run_${taskRole.name}.sh`), runScriptContent, { encoding: 'utf8' });
        }

        // Write file content ( parameter.cfg ) to local tmp folders
        if (form !== undefined) {
            await fs.promises.writeFile(path.join(trialLocalTempFolder, generateParamFileName(form.hyperParameters)),
                                        form.hyperParameters.value, { encoding: 'utf8' });
        }
    }

    private async prepareFrameworkControllerConfig(trialJobId: string, trialWorkingFolder: string, frameworkcontrollerJobName: string):
     Promise<any> {

        if (this.fcTrialConfig === undefined) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }

        const podResources: any = [];
        for (const taskRole of this.fcTrialConfig.taskRoles) {
            const resource: any = {};
            resource.requests = this.generatePodResource(taskRole.memoryMB, taskRole.cpuNum, taskRole.gpuNum);
            resource.limits = {...resource.requests};
            podResources.push(resource);
        }
        // Generate frameworkcontroller job resource config object
        const frameworkcontrollerJobConfig: any =
          await this.generateFrameworkControllerJobConfig(trialJobId, trialWorkingFolder, frameworkcontrollerJobName, podResources);

        return Promise.resolve(frameworkcontrollerJobConfig);
    }

    private generateContainerPort(): void {
        if (this.fcTrialConfig === undefined) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }

        let port: number = 4000; //The default port used in container
        for (const index of this.fcTrialConfig.taskRoles.keys()) {
            this.fcContainerPortMap.set(this.fcTrialConfig.taskRoles[index].name, port);
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
        if (this.fcClusterConfig === undefined) {
            throw new Error('frameworkcontroller Cluster config is not initialized');
        }

        if (this.fcTrialConfig === undefined) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }

        const taskRoles: any = [];
        for (const index of this.fcTrialConfig.taskRoles.keys()) {
            const containerPort: number | undefined = this.fcContainerPortMap.get(this.fcTrialConfig.taskRoles[index].name);
            if (containerPort === undefined) {
                throw new Error('Container port is not initialized');
            }
            
            const taskRole: any = this.generateTaskRoleConfig(
                trialWorkingFolder,
                this.fcTrialConfig.taskRoles[index].image,
                `run_${this.fcTrialConfig.taskRoles[index].name}.sh`,
                podResources[index],
                containerPort,
                await this.createRegistrySecret(this.fcTrialConfig.taskRoles[index].privateRegistryAuthPath)
            );
            taskRoles.push({
                name: this.fcTrialConfig.taskRoles[index].name,
                taskNumber: this.fcTrialConfig.taskRoles[index].taskNum,
                frameworkAttemptCompletionPolicy: {
                    minFailedTaskCount: this.fcTrialConfig.taskRoles[index].frameworkAttemptCompletionPolicy.minFailedTaskCount,
                    minSucceededTaskCount: this.fcTrialConfig.taskRoles[index].frameworkAttemptCompletionPolicy.minSucceededTaskCount
                },
                task: taskRole
            });
        }

        return Promise.resolve({
            apiVersion: `frameworkcontroller.microsoft.com/v1`,
            kind: 'Framework',
            metadata: {
                name: frameworkcontrollerJobName,
                namespace: 'default',
                labels: {
                    app: this.NNI_KUBERNETES_TRIAL_LABEL,
                    expId: getExperimentId(),
                    trialId: trialJobId
                }
            },
            spec: {
                executionType: 'Start',
                taskRoles: taskRoles
            }
        });
    }

    private  generateTaskRoleConfig(trialWorkingFolder: string, replicaImage: string, runScriptFile: string,
                                   podResources: any, containerPort: number, privateRegistrySecretName: string | undefined): any {
        if (this.fcClusterConfig === undefined) {
            throw new Error('frameworkcontroller Cluster config is not initialized');
        }

        if (this.fcTrialConfig === undefined) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }

        const volumeSpecMap: Map<string, object> = new Map<string, object>();
        if (this.fcClusterConfig.storageType === 'azureStorage') {
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
            const frameworkcontrollerClusterConfigNFS: FrameworkControllerClusterConfigNFS =
              <FrameworkControllerClusterConfigNFS> this.fcClusterConfig;
            volumeSpecMap.set('nniVolumes', [
            {
                name: 'nni-vol',
                nfs: {
                    server: `${frameworkcontrollerClusterConfigNFS.nfs.server}`,
                    path: `${frameworkcontrollerClusterConfigNFS.nfs.path}`
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
        if(privateRegistrySecretName) {
            spec.imagePullSecrets = [
                {
                    name: privateRegistrySecretName 
                }
             ]
        }

        if (this.fcClusterConfig.serviceAccountName !== undefined) {
            spec.serviceAccountName = this.fcClusterConfig.serviceAccountName;
        }

        return {
            pod: {
                spec: spec
            }
        };
    }
}

export { FrameworkControllerTrainingService };
