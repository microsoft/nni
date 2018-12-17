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

import * as assert from 'assert';
import * as azureStorage from 'azure-storage';
import * as component from '../../common/component';
import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as path from 'path';

import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { EventEmitter } from 'events';
import { getExperimentId, getInitTrialSequenceId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import { MethodNotImplementedError } from '../../common/errors';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import {
    JobApplicationForm, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric, NNIManagerIpConfig
} from '../../common/trainingService';
import { delay, generateParamFileName, getExperimentRootDir, getIPV4Address, uniqueString, getJobCancelStatus } from '../../common/utils';
import { NFSConfig } from '../kubernetes/kubernetesConfig'
import { KubernetesTrialJobDetail } from '../kubernetes/kubernetesData';
import { KubernetesTrialConfig, KubernetesClusterConfig } from '../kubernetes/kubernetesConfig';
import { validateCodeDir } from '../common/util';
import { AzureStorageClientUtility } from '../kubernetes/azureStorageClientUtils';
import { KubernetesTrainingService } from '../kubernetes/kubernetesTrainingService';
import { FrameworkControllerClusterConfigAzure, FrameworkControllerClusterConfigNFS, FrameworkControllerTrialConfig, 
    FrameworkControllerClusterConfigFactory } from './frameworkcontrollerConfig';
import { GeneralK8sClient } from '../kubernetes/kubernetesApiClient';
import { FrameworkControllerJobRestServer } from './frameworkcontrollerJobRestServer';
import { FrameworkControllerClient } from './frameworkcontrollerApiClient';
import { FrameworkControllerJobInfoCollector } from './frameworkcontrollerJobInfoCollector';


var azure = require('azure-storage');
var base64 = require('js-base64').Base64;

/**
 * Training Service implementation for Kubeflow
 * Refer https://github.com/kubeflow/kubeflow for more info about Kubeflow
 */
@component.Singleton
class FrameworkControllerainingService extends KubernetesTrainingService {
    private frameworkcontrollerClusterConfig?: KubernetesClusterConfig;
    private frameworkcotrollerAPIClient?: FrameworkControllerClient;
    private frameworkcontrollerJobInfoCollector: FrameworkControllerJobInfoCollector;

    constructor() {
        super();
        this.frameworkcontrollerJobInfoCollector = new FrameworkControllerJobInfoCollector(this.trialJobsMap);
        this.experimentId = getExperimentId();      
        this.nextTrialSequenceId = -1;
    }

    public async run(): Promise<void> {
        const restServer: FrameworkControllerJobRestServer = component.get(FrameworkControllerJobRestServer);
        await restServer.start();
        this.log.info(`Kubeflow Training service rest server listening on: ${restServer.endPoint}`);
        while (!this.stopping) {
            // collect metrics for Kubeflow jobs by interacting with Kubernetes API server  
            await delay(3000);
            await this.frameworkcontrollerJobInfoCollector.retrieveTrialStatus(this.frameworkcotrollerAPIClient);
        }
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        if(!this.frameworkcontrollerClusterConfig) {
            throw new Error('frameworkcontrollerClusterConfig is not initialized');
        }

        if(!this.kubernetesTrialConfig) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        if(!this.frameworkcotrollerAPIClient) {
            throw new Error('Kubeflow job operator client is undefined');
        }

        if(!this.kubernetesRestServerPort) {
            const restServer: FrameworkControllerJobRestServer = component.get(FrameworkControllerJobRestServer);
            this.kubernetesRestServerPort = restServer.clusterRestServerPort;
        }
        // initialize kubeflow trial config to specific type
        let frameworkcontrollerTrialConfig;
        frameworkcontrollerTrialConfig = <FrameworkControllerTrialConfig>this.kubernetesTrialConfig;

        const trialJobId: string = uniqueString(5);
        const curTrialSequenceId: number = this.generateSequenceId();
        // Set trial's NFS working folder
        const trialWorkingFolder: string = path.join(this.CONTAINER_MOUNT_PATH, 'nni', getExperimentId(), trialJobId);
        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        //create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${path.dirname(trialLocalTempFolder)}`);
        await cpp.exec(`cp -r ${frameworkcontrollerTrialConfig.codeDir} ${trialLocalTempFolder}`);
        const runScriptContent : string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });
        // Create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${trialLocalTempFolder}`);

        // Write worker file content run_worker.sh to local tmp folders
        if(frameworkcontrollerTrialConfig.worker) {
            const workerRunScriptContent: string = this.generateRunScript(trialJobId, trialWorkingFolder, 
                frameworkcontrollerTrialConfig.worker.command, curTrialSequenceId.toString(), 'worker', frameworkcontrollerTrialConfig.worker.gpuNum);

            await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_worker.sh'), workerRunScriptContent, { encoding: 'utf8' });
        }
  
        if(frameworkcontrollerTrialConfig.ps){
            const psRunScriptContent: string = this.generateRunScript(trialJobId, trialWorkingFolder, 
                frameworkcontrollerTrialConfig.ps.command, curTrialSequenceId.toString(), 'ps', frameworkcontrollerTrialConfig.ps.gpuNum);
            await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_ps.sh'), psRunScriptContent, { encoding: 'utf8' });
        }
        
        // Write file content ( parameter.cfg ) to local tmp folders
        const trialForm : TrialJobApplicationForm = (<TrialJobApplicationForm>form)
        if(trialForm && trialForm.hyperParameters) {
            await fs.promises.writeFile(path.join(trialLocalTempFolder, generateParamFileName(trialForm.hyperParameters)), 
                            trialForm.hyperParameters.value, { encoding: 'utf8' });
        }
        const kubeflowJobName = `nniexp${this.experimentId}trial${trialJobId}`.toLowerCase();
        
        const workerPodResources : any = {};
        if(frameworkcontrollerTrialConfig.worker) {
            workerPodResources.requests = this.generatePodResource(frameworkcontrollerTrialConfig.worker.memoryMB, frameworkcontrollerTrialConfig.worker.cpuNum, 
                frameworkcontrollerTrialConfig.worker.gpuNum)
        }
        workerPodResources.limits = Object.assign({}, workerPodResources.requests);

        let nonWorkerResources : any = {};
        if (frameworkcontrollerTrialConfig.ps) {
            nonWorkerResources.requests = this.generatePodResource(frameworkcontrollerTrialConfig.ps.memoryMB, frameworkcontrollerTrialConfig.ps.cpuNum, 
                frameworkcontrollerTrialConfig.ps.gpuNum)
                nonWorkerResources.limits = Object.assign({}, nonWorkerResources.requests); 
        }      

        //The output url used in trialJobDetail
        let trialJobOutputUrl: string = '';

        assert(!this.frameworkcontrollerClusterConfig.storage 
            || this.frameworkcontrollerClusterConfig.storage === 'azureStorage' 
            || this.frameworkcontrollerClusterConfig.storage === 'nfs');

        if(this.frameworkcontrollerClusterConfig.storage === 'azureStorage') {
            try{
                //upload local files to azure storage
                await AzureStorageClientUtility.uploadDirectory(this.azureStorageClient, 
                    `nni/${getExperimentId()}/${trialJobId}`, this.azureStorageShare, `${trialLocalTempFolder}`);

                trialJobOutputUrl = `https://${this.azureStorageAccountName}.file.core.windows.net/${this.azureStorageShare}/${path.join('nni', getExperimentId(), trialJobId, 'output')}`
            }catch(error){
                this.log.error(error);
                return Promise.reject(error);
            }
        } else if(this.frameworkcontrollerClusterConfig.storage === 'nfs' || this.frameworkcontrollerClusterConfig.storage === undefined) {
            let nfsKubeflowClusterConfig: FrameworkControllerClusterConfigNFS = <FrameworkControllerClusterConfigNFS>this.frameworkcontrollerClusterConfig;
            // Creat work dir for current trial in NFS directory 
            await cpp.exec(`mkdir -p ${this.trialLocalNFSTempFolder}/nni/${getExperimentId()}/${trialJobId}`);
            // Copy code files from local dir to NFS mounted dir
            await cpp.exec(`cp -r ${trialLocalTempFolder}/* ${this.trialLocalNFSTempFolder}/nni/${getExperimentId()}/${trialJobId}/.`);
        
            const nfsConfig: NFSConfig = nfsKubeflowClusterConfig.nfs;
            trialJobOutputUrl = `nfs://${nfsConfig.server}:${path.join(nfsConfig.path, 'nni', getExperimentId(), trialJobId, 'output')}`
        }

        const trialJobDetail: KubernetesTrialJobDetail = new KubernetesTrialJobDetail(
            trialJobId,
            'WAITING',
            Date.now(),
            trialWorkingFolder,
            form,
            kubeflowJobName,
            curTrialSequenceId,
            trialJobOutputUrl
        );

        // Set trial job detail until create Kubeflow job successfully 
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        // Generate kubeflow job resource config object        
        const kubeflowJobConfig: any = this.generateKubeflowJobConfig(trialJobId, trialWorkingFolder, kubeflowJobName, workerPodResources, nonWorkerResources);

        // Create kubeflow job based on generated kubeflow job resource config
        await this.frameworkcotrollerAPIClient.createKubeflowJob(kubeflowJobConfig);

        // Set trial job detail until create Kubeflow job successfully 
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trialJobDetail : KubernetesTrialJobDetail | undefined =  this.trialJobsMap.get(trialJobId);
        if(!trialJobDetail) {
            const errorMessage: string = `CancelTrialJob: trial job id ${trialJobId} not found`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }        
        if(!this.frameworkcotrollerAPIClient) {
            const errorMessage: string = `CancelTrialJob: trial job id ${trialJobId} failed because operatorClient is undefined`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }

        try {
            await this.frameworkcotrollerAPIClient.deleteKubeflowJob(new Map(
                [
                    ['app', this.NNI_KUBEFLOW_TRIAL_LABEL],
                    ['expId', getExperimentId()],
                    ['trialId', trialJobId]
                ]
            ));
        } catch(err) {
            const errorMessage: string = `Delete trial ${trialJobId} failed: ${err}`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }

        trialJobDetail.endTime = Date.now();
        trialJobDetail.status = getJobCancelStatus(isEarlyStopped);

        return Promise.resolve();
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;
            
            case TrialConfigMetadataKey.FRAMEWORKCONTROLLER_CLUSTER_CONFIG:
                let frameworkcontrollerClusterJsonObject = JSON.parse(value);
                this.frameworkcontrollerClusterConfig = FrameworkControllerClusterConfigFactory.generateFrameworkControllerClusterConfig(frameworkcontrollerClusterJsonObject);
                if(this.frameworkcontrollerClusterConfig.storageType === 'azureStorage') {
                    let azureKubeflowClusterConfig = <FrameworkControllerClusterConfigAzure>this.frameworkcontrollerClusterConfig;
                    this.azureStorageAccountName = azureKubeflowClusterConfig.azureStorage.accountName;
                    this.azureStorageShare = azureKubeflowClusterConfig.azureStorage.azureShare;
                    await this.createAzureStorage(
                        azureKubeflowClusterConfig.keyVault.vaultName,
                        azureKubeflowClusterConfig.keyVault.name,
                        azureKubeflowClusterConfig.azureStorage.accountName,
                        azureKubeflowClusterConfig.azureStorage.azureShare
                    );
                } else if(this.frameworkcontrollerClusterConfig.storageType === 'nfs') {
                    let nfsKubeflowClusterConfig = <FrameworkControllerClusterConfigNFS>this.frameworkcontrollerClusterConfig;
                    await this.createNFSStorage(
                        nfsKubeflowClusterConfig.nfs.server,
                        nfsKubeflowClusterConfig.nfs.path
                    );
                } 
                this.frameworkcotrollerAPIClient = FrameworkControllerClient.generateOperatorClient();
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG:
            if (!this.frameworkcontrollerClusterConfig){
                this.log.error('kubeflow cluster config is not initialized');
                return Promise.reject(new Error('kubeflow cluster config is not initialized'));                    
            }

            assert(this.frameworkcontrollerClusterConfig !== undefined)
            let kubeflowTrialJsonObjsect = JSON.parse(value);

            this.kubernetesTrialConfig = new FrameworkControllerTrialConfig(kubeflowTrialJsonObjsect.codeDir, 
                kubeflowTrialJsonObjsect.worker, kubeflowTrialJsonObjsect.ps);
            if (!this.kubernetesTrialConfig){
                this.log.error('kubeflow kubeflow TrialConfig is not initialized');
                return Promise.reject(new Error('kubeflow kubeflow TrialConfig is not initialized'));                    
            }

            // Validate to make sure codeDir doesn't have too many files
            try {
                await validateCodeDir(this.kubernetesTrialConfig.codeDir);
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

    public async cleanUp(): Promise<void> {
        this.stopping = true;

        // First, cancel all running kubeflow jobs
        for(let [trialJobId, kubernetesTrialJob] of this.trialJobsMap) {
            if(['RUNNING', 'WAITING', 'UNKNOWN'].includes(kubernetesTrialJob.status)) {
                try {
                    await this.cancelTrialJob(trialJobId);
                } catch(error) {} // DONT throw error during cleanup
                kubernetesTrialJob.status = 'SYS_CANCELED';
            }
        }
        
        // Delete all kubeflow jobs whose expId label is current experiment id 
        try {
            if(this.frameworkcotrollerAPIClient) {
                await this.frameworkcotrollerAPIClient.deleteKubeflowJob(new Map(
                    [
                        ['app', this.NNI_KUBEFLOW_TRIAL_LABEL],
                        ['expId', getExperimentId()]
                    ]
                ));
            }
        } catch(error) {
            this.log.error(`Delete kubeflow job with label: app=${this.NNI_KUBEFLOW_TRIAL_LABEL},expId=${getExperimentId()} failed, error is ${error}`);
        }

        // Unmount NFS
        try {
            await cpp.exec(`sudo umount ${this.trialLocalNFSTempFolder}`);
        } catch(error) {
            this.log.error(`Unmount ${this.trialLocalNFSTempFolder} failed, error is ${error}`);
        }

        // Stop Kubeflow rest server 
        const restServer: FrameworkControllerJobRestServer = component.get(FrameworkControllerJobRestServer);
        try {
            await restServer.stop();
            this.log.info('Kubeflow Training service rest server stopped successfully.');
        } catch (error) {
            this.log.error(`Kubeflow Training service rest server stopped failed, error: ${error.message}`);
            Promise.reject(error);
        }

        return Promise.resolve();
    }

       /**
     * Generate kubeflow resource config file
     * @param trialJobId trial job id
     * @param trialWorkingFolder working folder
     * @param kubeflowJobName job name
     * @param workerPodResources worker pod template
     * @param nonWorkerPodResources non-worker pod template, like ps or master
     */
    private generateKubeflowJobConfig(trialJobId: string, trialWorkingFolder: string, kubeflowJobName : string, workerPodResources : any, nonWorkerPodResources?: any) : any {
        if(!this.frameworkcontrollerClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if(!this.kubernetesTrialConfig) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        let replicaSpecsObj: any;

        let frameworkcontrollerTrialConfig = <FrameworkControllerTrialConfig>this.kubernetesTrialConfig;
        replicaSpecsObj = this.generateReplicaConfig(trialWorkingFolder, frameworkcontrollerTrialConfig.worker.replicas, 
            frameworkcontrollerTrialConfig.worker.image, 'run_worker.sh', workerPodResources);

        return {
            apiVersion: `frameworkcontroller.microsoft.com/v1`,
            kind: 'Framework',
            metadata: { 
                name: kubeflowJobName,
                namespace: 'default',
                labels: {
                    app: this.NNI_KUBEFLOW_TRIAL_LABEL,
                    expId: getExperimentId(),
                    trialId: trialJobId
                }
            },
            spec: replicaSpecsObj
        };     
    }

    /**
     * Generate tf-operator's tfjobs replica config section
     * @param trialWorkingFolder trial working folder
     * @param replicaNumber replica number
     * @param replicaImage image
     * @param runScriptFile script file name
     * @param podResources pod resource config section
     */
    private generateReplicaConfig(trialWorkingFolder: string, replicaNumber: number, replicaImage: string, runScriptFile: string, podResources: any): any {
        if(!this.frameworkcontrollerClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if(!this.kubernetesTrialConfig) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        let volumeSpecMap = new Map<string, object>();
        if(this.frameworkcontrollerClusterConfig.storage && this.frameworkcontrollerClusterConfig.storage === 'azureStorage'){
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
            let nfsKubeflowClusterConfig: FrameworkControllerClusterConfigNFS = <FrameworkControllerClusterConfigNFS> this.kubernetesClusterConfig;
            volumeSpecMap.set('nniVolumes', [
            {
                name: 'nni-vol',
                nfs: {
                    server: `${nfsKubeflowClusterConfig.nfs.server}`,
                    path: `${nfsKubeflowClusterConfig.nfs.path}`
                }
            }])
        }

        return {
            executionType: 'Start',
            taskRoles: [
                {
                    name: 'worker',
                    taskNumber: replicaNumber,
                    task: {
                        pod: {
                            spec: {
                                containers: [
                                {
                                    // Kubeflow tensorflow operator requires that containers' name must be tensorflow
                                    // TODO: change the name based on operator's type
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
                }
            ]
        };
    }
}

export { FrameworkControllerainingService }
