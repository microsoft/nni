/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';

import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as path from 'path';
import * as component from '../../../common/component';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import {
    JobApplicationForm, NNIManagerIpConfig, TrialJobApplicationForm, TrialJobDetail
} from '../../../common/trainingService';
import { delay, generateParamFileName, getExperimentRootDir, uniqueString } from '../../../common/utils';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../../common/containerJobData';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { validateCodeDir } from '../../common/util';
import { AzureStorageClientUtility } from '../azureStorageClientUtils';
import { NFSConfig } from '../kubernetesConfig';
import { KubernetesTrialJobDetail } from '../kubernetesData';
import { KubernetesTrainingService } from '../kubernetesTrainingService';
import { FrameworkControllerClient } from './frameworkcontrollerApiClient';
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
        this.nextTrialSequenceId = -1;
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

    public async submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
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

        const trialJobId: string = uniqueString(5);
        const curTrialSequenceId: number = this.generateSequenceId();
        // Set trial's NFS working folder
        const trialWorkingFolder: string = path.join(this.CONTAINER_MOUNT_PATH, 'nni', getExperimentId(), trialJobId);
        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        const frameworkcontrollerJobName: string = `nniexp${this.experimentId}trial${trialJobId}`.toLowerCase();
        //Generate the port used for taskRole
        this.generateContainerPort();
        await this.prepareRunScript(trialLocalTempFolder, curTrialSequenceId, trialJobId, trialWorkingFolder, form);

        //upload code files
        const trialJobOutputUrl: string = await this.uploadCodeFiles(trialJobId, trialLocalTempFolder);

        const trialJobDetail: KubernetesTrialJobDetail = new KubernetesTrialJobDetail(
            trialJobId,
            'WAITING',
            Date.now(),
            trialWorkingFolder,
            form,
            frameworkcontrollerJobName,
            curTrialSequenceId,
            trialJobOutputUrl
        );

        // Set trial job detail until create frameworkcontroller job successfully
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        // Create frameworkcontroller job based on generated frameworkcontroller job resource config
        // tslint:disable-next-line:no-any
        const frameworkcontrollerJobConfig: any = await this.prepareFrameworkControllerConfig(
            trialJobId, trialWorkingFolder, frameworkcontrollerJobName);
        await this.kubernetesCRDClient.createKubernetesJob(frameworkcontrollerJobConfig);

        // Set trial job detail until create frameworkcontroller job successfully
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    // tslint:disable:no-redundant-jsdoc no-any no-unsafe-any
    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;
            case TrialConfigMetadataKey.FRAMEWORKCONTROLLER_CLUSTER_CONFIG:
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
                        azureFrameworkControllerClusterConfig.keyVault.name,
                        azureFrameworkControllerClusterConfig.azureStorage.accountName,
                        azureFrameworkControllerClusterConfig.azureStorage.azureShare
                    );
                } else if (this.fcClusterConfig.storageType === 'nfs') {
                    const nfsFrameworkControllerClusterConfig: FrameworkControllerClusterConfigNFS =
                      <FrameworkControllerClusterConfigNFS>this.fcClusterConfig;
                    await this.createNFSStorage(
                        nfsFrameworkControllerClusterConfig.nfs.server,
                        nfsFrameworkControllerClusterConfig.nfs.path
                    );
                }
                this.kubernetesCRDClient = FrameworkControllerClient.generateFrameworkControllerClient();
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                const frameworkcontrollerTrialJsonObjsect: any = JSON.parse(value);

                this.fcTrialConfig = new FrameworkControllerTrialConfig(
                    frameworkcontrollerTrialJsonObjsect.codeDir,
                    frameworkcontrollerTrialJsonObjsect.taskRoles
                );

                // Validate to make sure codeDir doesn't have too many files
                try {
                    await validateCodeDir(this.fcTrialConfig.codeDir);
                } catch (error) {
                    this.log.error(error);

                    return Promise.reject(new Error(error));
                }
                break;
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
    // tslint:enable: no-any no-unsafe-any

    /**
     * upload code files to nfs or azureStroage
     * @param trialJobId
     * @param trialLocalTempFolder
     * return: trialJobOutputUrl
     */
    private async uploadCodeFiles(trialJobId: string, trialLocalTempFolder: string): Promise<string> {
        if (this.fcClusterConfig === undefined) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        let trialJobOutputUrl: string = '';

        if (this.fcClusterConfig.storageType === 'azureStorage') {
            if (this.azureStorageClient === undefined) {
                throw new Error('azureStorageClient is not initialized');
            }
            try {
                //upload local files to azure storage
                await AzureStorageClientUtility.uploadDirectory(
                    this.azureStorageClient, `nni/${getExperimentId()}/${trialJobId}`, this.azureStorageShare, `${trialLocalTempFolder}`);

                trialJobOutputUrl = `https://${this.azureStorageAccountName}.file.core.windows.net/\
                ${this.azureStorageShare}/${path.join('nni', getExperimentId(), trialJobId, 'output')}`;
            } catch (error) {
                this.log.error(error);

                return Promise.reject(error);
            }
        } else if (this.fcClusterConfig.storageType === 'nfs') {
            const nfsFrameworkControllerClusterConfig: FrameworkControllerClusterConfigNFS =
              <FrameworkControllerClusterConfigNFS>this.fcClusterConfig;
            // Creat work dir for current trial in NFS directory
            await cpp.exec(`mkdir -p ${this.trialLocalNFSTempFolder}/nni/${getExperimentId()}/${trialJobId}`);
            // Copy code files from local dir to NFS mounted dir
            await cpp.exec(`cp -r ${trialLocalTempFolder}/* ${this.trialLocalNFSTempFolder}/nni/${getExperimentId()}/${trialJobId}/.`);

            const nfsConfig: NFSConfig = nfsFrameworkControllerClusterConfig.nfs;
            trialJobOutputUrl = `nfs://${nfsConfig.server}:${path.join(nfsConfig.path, 'nni', getExperimentId(), trialJobId, 'output')}`;
        }

        return Promise.resolve(trialJobOutputUrl);
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

    private async prepareRunScript(trialLocalTempFolder: string, curTrialSequenceId: number, trialJobId: string,
                                   trialWorkingFolder: string, form: JobApplicationForm): Promise<void> {
        if (this.fcTrialConfig === undefined) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }

        await cpp.exec(`mkdir -p ${path.dirname(trialLocalTempFolder)}`);
        await cpp.exec(`cp -r ${this.fcTrialConfig.codeDir} ${trialLocalTempFolder}`);
        const installScriptContent : string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), installScriptContent, { encoding: 'utf8' });
        // Create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${trialLocalTempFolder}`);

        for (const taskRole of this.fcTrialConfig.taskRoles) {
            const runScriptContent: string =
              await this.generateRunScript('frameworkcontroller', trialJobId, trialWorkingFolder,
                                           this.generateCommandScript(taskRole.command), curTrialSequenceId.toString(),
                                           taskRole.name, taskRole.gpuNum);
            await fs.promises.writeFile(path.join(trialLocalTempFolder, `run_${taskRole.name}.sh`), runScriptContent, { encoding: 'utf8' });
        }

        // Write file content ( parameter.cfg ) to local tmp folders
        const trialForm : TrialJobApplicationForm = (<TrialJobApplicationForm>form);
        if (trialForm !== undefined && trialForm.hyperParameters !== undefined) {
            await fs.promises.writeFile(path.join(trialLocalTempFolder, generateParamFileName(trialForm.hyperParameters)),
                                        trialForm.hyperParameters.value, { encoding: 'utf8' });
        }
    }

    // tslint:disable: no-any no-unsafe-any
    private async prepareFrameworkControllerConfig(trialJobId: string, trialWorkingFolder: string, frameworkcontrollerJobName: string):
     Promise<any> {

        if (this.fcTrialConfig === undefined) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }

        const podResources : any = [];
        for (const taskRole of this.fcTrialConfig.taskRoles) {
            const resource: any = {};
            resource.requests = this.generatePodResource(taskRole.memoryMB, taskRole.cpuNum, taskRole.gpuNum);
            resource.limits = {...resource.requests};
            podResources.push(resource);
        }
        // Generate frameworkcontroller job resource config object
        const frameworkcontrollerJobConfig: any =
          this.generateFrameworkControllerJobConfig(trialJobId, trialWorkingFolder, frameworkcontrollerJobName, podResources);

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
    private generateFrameworkControllerJobConfig(trialJobId: string, trialWorkingFolder: string,
                                                 frameworkcontrollerJobName : string, podResources : any) : any {
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
                containerPort
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

        return {
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
        };
    }

    private generateTaskRoleConfig(trialWorkingFolder: string, replicaImage: string, runScriptFile: string,
                                   podResources: any, containerPort: number): any {
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
        if (this.fcClusterConfig.serviceAccountName !== undefined) {
            spec.serviceAccountName = this.fcClusterConfig.serviceAccountName;
        }

        return {
            pod: {
                spec: spec
            }
        };
    }
    // tslint:enable: no-any no-unsafe-any
}

export { FrameworkControllerTrainingService };
