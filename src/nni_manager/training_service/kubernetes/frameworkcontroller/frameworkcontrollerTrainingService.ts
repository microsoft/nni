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

'use strict'

import * as component from '../../../common/component';
import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as path from 'path';

import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../../common/containerJobData';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import {
    JobApplicationForm, TrialJobApplicationForm,
    TrialJobDetail, NNIManagerIpConfig
} from '../../../common/trainingService';
import { delay, generateParamFileName, getExperimentRootDir, uniqueString } from '../../../common/utils';
import { NFSConfig, KubernetesClusterConfigNFS, KubernetesClusterConfigAzure, KubernetesClusterConfigFactory } from '../kubernetesConfig'
import { KubernetesTrialJobDetail } from '../kubernetesData';
import { validateCodeDir } from '../../common/util';
import { AzureStorageClientUtility } from '../azureStorageClientUtils';
import { KubernetesTrainingService } from '../kubernetesTrainingService';
import { FrameworkControllerTrialConfig } from './frameworkcontrollerConfig';
import { FrameworkControllerJobRestServer } from './frameworkcontrollerJobRestServer';
import { FrameworkControllerClient } from './frameworkcontrollerApiClient';
import { FrameworkControllerJobInfoCollector } from './frameworkcontrollerJobInfoCollector';

/**
 * Training Service implementation for frameworkcontroller
 */
@component.Singleton
class FrameworkControllerTrainingService extends KubernetesTrainingService implements KubernetesTrainingService {
    private frameworkcontrollerTrialConfig?: FrameworkControllerTrialConfig;
    private frameworkcontrollerJobInfoCollector: FrameworkControllerJobInfoCollector;

    constructor() {
        super();
        this.frameworkcontrollerJobInfoCollector = new FrameworkControllerJobInfoCollector(this.trialJobsMap);
        this.experimentId = getExperimentId();      
        this.nextTrialSequenceId = -1;
    }

    public async run(): Promise<void> {
        this.kubernetesJobRestServer = component.get(FrameworkControllerJobRestServer);
        if(!this.kubernetesJobRestServer) {
            throw new Error('kubernetesJobRestServer not initialized!');
        }
        await this.kubernetesJobRestServer.start();
        this.log.info(`frameworkcontroller Training service rest server listening on: ${this.kubernetesJobRestServer.endPoint}`);
        while (!this.stopping) {
            // collect metrics for frameworkcontroller jobs by interacting with Kubernetes API server  
            await delay(3000);
            await this.frameworkcontrollerJobInfoCollector.retrieveTrialStatus(this.kubernetesCRDClient);
        }
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        if(!this.kubernetesClusterConfig) {
            throw new Error('frameworkcontrollerClusterConfig is not initialized');
        }
        if(!this.kubernetesCRDClient) {
            throw new Error('kubernetesCRDClient is undefined');
        }

        if(!this.kubernetesRestServerPort) {
            const restServer: FrameworkControllerJobRestServer = component.get(FrameworkControllerJobRestServer);
            this.kubernetesRestServerPort = restServer.clusterRestServerPort;
        }

        const trialJobId: string = uniqueString(5);
        const curTrialSequenceId: number = this.generateSequenceId();
        // Set trial's NFS working folder
        const trialWorkingFolder: string = path.join(this.CONTAINER_MOUNT_PATH, 'nni', getExperimentId(), trialJobId);
        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        const frameworkcontrollerJobName = `nniexp${this.experimentId}trial${trialJobId}`.toLowerCase();
        
        await this.prepareRunScript(trialLocalTempFolder, curTrialSequenceId, trialJobId, trialWorkingFolder, form);
        
        //upload code files
        let trialJobOutputUrl: string = await this.uploadCodeFiles(trialJobId, trialLocalTempFolder);
        
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
        const frameworkcontrollerJobConfig = await this.prepareFrameworkControllerConfig(trialJobId, trialWorkingFolder, frameworkcontrollerJobName);
        await this.kubernetesCRDClient.createKubernetesJob(frameworkcontrollerJobConfig);

        // Set trial job detail until create frameworkcontroller job successfully 
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    /**
     * upload code files to nfs or azureStroage
     * @param trialJobId 
     * @param trialLocalTempFolder 
     * return: trialJobOutputUrl
     */
    private async uploadCodeFiles(trialJobId: string, trialLocalTempFolder: string): Promise<string> {
        if(!this.kubernetesClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        let trialJobOutputUrl: string = '';

        if(this.kubernetesClusterConfig.storageType === 'azureStorage') {
            try{
                //upload local files to azure storage
                await AzureStorageClientUtility.uploadDirectory(this.azureStorageClient, 
                    `nni/${getExperimentId()}/${trialJobId}`, this.azureStorageShare, `${trialLocalTempFolder}`);

                trialJobOutputUrl = `https://${this.azureStorageAccountName}.file.core.windows.net/${this.azureStorageShare}/${path.join('nni', getExperimentId(), trialJobId, 'output')}`
            }catch(error){
                this.log.error(error);
                return Promise.reject(error);
            }
        } else if(this.kubernetesClusterConfig.storageType === 'nfs') {
            let nfsFrameworkControllerClusterConfig: KubernetesClusterConfigNFS = <KubernetesClusterConfigNFS>this.kubernetesClusterConfig;
            // Creat work dir for current trial in NFS directory 
            await cpp.exec(`mkdir -p ${this.trialLocalNFSTempFolder}/nni/${getExperimentId()}/${trialJobId}`);
            // Copy code files from local dir to NFS mounted dir
            await cpp.exec(`cp -r ${trialLocalTempFolder}/* ${this.trialLocalNFSTempFolder}/nni/${getExperimentId()}/${trialJobId}/.`);
        
            const nfsConfig: NFSConfig = nfsFrameworkControllerClusterConfig.nfs;
            trialJobOutputUrl = `nfs://${nfsConfig.server}:${path.join(nfsConfig.path, 'nni', getExperimentId(), trialJobId, 'output')}`
        }
        return Promise.resolve(trialJobOutputUrl);
    }
    
    private async prepareRunScript(trialLocalTempFolder: string, curTrialSequenceId: number, trialJobId: string, trialWorkingFolder: string, form: JobApplicationForm): Promise<void> {
        if(!this.frameworkcontrollerTrialConfig) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }

        await cpp.exec(`mkdir -p ${path.dirname(trialLocalTempFolder)}`);
        await cpp.exec(`cp -r ${this.frameworkcontrollerTrialConfig.codeDir} ${trialLocalTempFolder}`);
        const runScriptContent : string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });
        // Create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${trialLocalTempFolder}`);

        for(let taskRole of this.frameworkcontrollerTrialConfig.taskRoles) {
            const runScriptContent: string = this.generateRunScript('frameworkcontroller', trialJobId, trialWorkingFolder, 
                taskRole.command, curTrialSequenceId.toString(), taskRole.name, taskRole.gpuNum);
            await fs.promises.writeFile(path.join(trialLocalTempFolder, `run_${taskRole.name}.sh`), runScriptContent, { encoding: 'utf8' });
        }

        // Write file content ( parameter.cfg ) to local tmp folders
        const trialForm : TrialJobApplicationForm = (<TrialJobApplicationForm>form)
        if(trialForm && trialForm.hyperParameters) {
            await fs.promises.writeFile(path.join(trialLocalTempFolder, generateParamFileName(trialForm.hyperParameters)), 
                            trialForm.hyperParameters.value, { encoding: 'utf8' });
        }
    }
    
    private async prepareFrameworkControllerConfig(trialJobId: string, trialWorkingFolder: string, frameworkcontrollerJobName: string): Promise<any> {

        if(!this.frameworkcontrollerTrialConfig) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }

        const podResources : any = [];
        for(let taskRole of this.frameworkcontrollerTrialConfig.taskRoles) {
            let resource: any = {};
            resource.requests = this.generatePodResource(taskRole.memoryMB, taskRole.cpuNum, taskRole.gpuNum);
            resource.limits = Object.assign({}, resource.requests);
            podResources.push(resource);
        }
        // Generate frameworkcontroller job resource config object        
        const frameworkcontrollerJobConfig: any = this.generateFrameworkControllerJobConfig(trialJobId, trialWorkingFolder, frameworkcontrollerJobName, podResources);

        return Promise.resolve(frameworkcontrollerJobConfig);
    } 

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;
            
            case TrialConfigMetadataKey.FRAMEWORKCONTROLLER_CLUSTER_CONFIG:
                let frameworkcontrollerClusterJsonObject = JSON.parse(value);
                this.kubernetesClusterConfig = KubernetesClusterConfigFactory.generateKubernetesClusterConfig(frameworkcontrollerClusterJsonObject);
                if(this.kubernetesClusterConfig.storageType === 'azureStorage') {
                    let azureFrameworkControllerClusterConfig = <KubernetesClusterConfigAzure>this.kubernetesClusterConfig;
                    this.azureStorageAccountName = azureFrameworkControllerClusterConfig.azureStorage.accountName;
                    this.azureStorageShare = azureFrameworkControllerClusterConfig.azureStorage.azureShare;
                    await this.createAzureStorage(
                        azureFrameworkControllerClusterConfig.keyVault.vaultName,
                        azureFrameworkControllerClusterConfig.keyVault.name,
                        azureFrameworkControllerClusterConfig.azureStorage.accountName,
                        azureFrameworkControllerClusterConfig.azureStorage.azureShare
                    );
                } else if(this.kubernetesClusterConfig.storageType === 'nfs') {
                    let nfsFrameworkControllerClusterConfig = <KubernetesClusterConfigNFS>this.kubernetesClusterConfig;
                    await this.createNFSStorage(
                        nfsFrameworkControllerClusterConfig.nfs.server,
                        nfsFrameworkControllerClusterConfig.nfs.path
                    );
                } 
                this.kubernetesCRDClient = FrameworkControllerClient.generateFrameworkControllerClient();
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                let frameworkcontrollerTrialJsonObjsect = JSON.parse(value);

                this.frameworkcontrollerTrialConfig = new FrameworkControllerTrialConfig(
                    frameworkcontrollerTrialJsonObjsect.codeDir,
                    frameworkcontrollerTrialJsonObjsect.taskRoles
                );

                // Validate to make sure codeDir doesn't have too many files
                try {
                    await validateCodeDir(this.frameworkcontrollerTrialConfig.codeDir);
                } catch(error) {
                    this.log.error(error);
                    return Promise.reject(new Error(error));                    
                }
                break;
            default:
                break;
        }

        return Promise.resolve();
    }

    /**
     * Generate frameworkcontroller resource config file
     * @param trialJobId trial job id
     * @param trialWorkingFolder working folder
     * @param frameworkcontrollerJobName job name
     * @param podResources  pod template
     */
    private generateFrameworkControllerJobConfig(trialJobId: string, trialWorkingFolder: string, frameworkcontrollerJobName : string, podResources : any) : any {
        if(!this.kubernetesClusterConfig) {
            throw new Error('frameworkcontroller Cluster config is not initialized');
        }

        if(!this.frameworkcontrollerTrialConfig) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }
        
        let taskRoles = [];
        for(let index in this.frameworkcontrollerTrialConfig.taskRoles) {
            let taskRole = this.generateTaskRoleConfig(
                trialWorkingFolder, 
                this.frameworkcontrollerTrialConfig.taskRoles[index].image, 
                `run_${this.frameworkcontrollerTrialConfig.taskRoles[index].name}.sh`,
                podResources[index]
            );
            taskRoles.push({
                name: this.frameworkcontrollerTrialConfig.taskRoles[index].name,
                taskNumber: this.frameworkcontrollerTrialConfig.taskRoles[index].taskNum,
                frameworkAttemptCompletionPolicy: {
                    minFailedTaskCount: this.frameworkcontrollerTrialConfig.taskRoles[index].frameworkAttemptCompletionPolicy.minFailedTaskCount, 
                    minSucceededTaskCount: this.frameworkcontrollerTrialConfig.taskRoles[index].frameworkAttemptCompletionPolicy.minSucceededTaskCount
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

    private generateTaskRoleConfig(trialWorkingFolder: string, replicaImage: string, runScriptFile: string, podResources: any): any {
        if(!this.kubernetesClusterConfig) {
            throw new Error('frameworkcontroller Cluster config is not initialized');
        }

        if(!this.frameworkcontrollerTrialConfig) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }

        let volumeSpecMap = new Map<string, object>();
        if(this.kubernetesClusterConfig.storageType === 'azureStorage'){
            volumeSpecMap.set('nniVolumes', [
            {
                    name: 'nni-vol',
                    azureFile: {
                        secretName: `${this.azureStorageSecretName}`,
                        shareName: `${this.azureStorageShare}`,
                        readonly: false
                    }
            }])
        }else {
            let frameworkcontrollerClusterConfigNFS: KubernetesClusterConfigNFS = <KubernetesClusterConfigNFS> this.kubernetesClusterConfig;
            volumeSpecMap.set('nniVolumes', [
            {
                name: 'nni-vol',
                nfs: {
                    server: `${frameworkcontrollerClusterConfigNFS.nfs.server}`,
                    path: `${frameworkcontrollerClusterConfigNFS.nfs.path}`
                }
            }])
        }
        
        let taskRole = {
            pod: {
                spec: {
                    containers: [
                    {
                        name: 'framework',
                        image: replicaImage,
                        args: ["sh", `${path.join(trialWorkingFolder, runScriptFile)}`],
                        volumeMounts: [
                        {
                            name: 'nni-vol',
                            mountPath: this.CONTAINER_MOUNT_PATH
                        }],
                        resources: podResources
                    }],
                    restartPolicy: 'OnFailure',
                    volumes: volumeSpecMap.get('nniVolumes')
                }
            }
        }
        return taskRole;
    }
}

export { FrameworkControllerTrainingService }
