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
import { DistTrainRole, KubeflowClusterConfigBase, KubeflowClusterConfigNFS, KubeflowClusterConfigAzure, KubeflowTrialConfigBase,
     KubeflowTrialConfigPytorch, KubeflowTrialConfigTensorflow, NFSConfig } from './kubeflowConfig';
import { KubeflowTrialJobDetail } from './kubeflowData';
import { KubeflowJobRestServer } from './kubeflowJobRestServer';
import { KubeflowJobInfoCollector } from './kubeflowJobInfoCollector';
import { validateCodeDir } from '../common/util';
import { AzureStorageClientUtility } from './azureStorageClientUtils';
import { GeneralK8sClient, KubeflowOperatorClient } from './kubernetesApiClient';

var azure = require('azure-storage');
var base64 = require('js-base64').Base64;

/**
 * Training Service implementation for Kubeflow
 * Refer https://github.com/kubeflow/kubeflow for more info about Kubeflow
 */
@component.Singleton
class KubeflowTrainingService implements TrainingService {
    private readonly NNI_KUBEFLOW_TRIAL_LABEL: string = 'nni-kubeflow-trial';
    private readonly log!: Logger;
    private readonly metricsEmitter: EventEmitter;
    private readonly trialJobsMap: Map<string, KubeflowTrialJobDetail>;
    /**  experiment root dir in NFS */
    private readonly trialLocalNFSTempFolder: string;
    private stopping: boolean = false;
    private experimentId! : string;
    private nextTrialSequenceId: number;
    private kubeflowClusterConfig?: KubeflowClusterConfigBase;
    private kubeflowTrialConfig?: KubeflowTrialConfigBase;
    private kubeflowJobInfoCollector: KubeflowJobInfoCollector;
    private kubeflowRestServerPort?: number;
    private operatorClient?: KubeflowOperatorClient;
    private readonly genericK8sClient: GeneralK8sClient;
    private readonly CONTAINER_MOUNT_PATH: string;
    private azureStorageClient?: azureStorage.FileService;
    private azureStorageShare?: string;
    private azureStorageSecretName?: string;
    private azureStorageAccountName?: string;
    private nniManagerIpConfig?: NNIManagerIpConfig;
    
    constructor() {        
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, KubeflowTrialJobDetail>();
        this.genericK8sClient = new GeneralK8sClient();
        this.kubeflowJobInfoCollector = new KubeflowJobInfoCollector(this.trialJobsMap);
        this.trialLocalNFSTempFolder = path.join(getExperimentRootDir(), 'trials-nfs-tmp');
        this.experimentId = getExperimentId();      
        this.nextTrialSequenceId = -1;
        this.CONTAINER_MOUNT_PATH = '/tmp/mount';
    }

    public async run(): Promise<void> {
        const restServer: KubeflowJobRestServer = component.get(KubeflowJobRestServer);
        await restServer.start();
        this.log.info(`Kubeflow Training service rest server listening on: ${restServer.endPoint}`);
        while (!this.stopping) {
            // collect metrics for Kubeflow jobs by interacting with Kubernetes API server  
            await delay(3000);
            await this.kubeflowJobInfoCollector.retrieveTrialStatus(this.operatorClient);
        }
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if(!this.kubeflowTrialConfig) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        if(!this.operatorClient) {
            throw new Error('Kubeflow job operator client is undefined');
        }

        if(!this.kubeflowRestServerPort) {
            const restServer: KubeflowJobRestServer = component.get(KubeflowJobRestServer);
            this.kubeflowRestServerPort = restServer.clusterRestServerPort;
        }
        // initialize kubeflow trial config to specific type
        let kubeflowTrialConfig;
        if(this.kubeflowClusterConfig.operator === 'tf-operator') {
            kubeflowTrialConfig = <KubeflowTrialConfigTensorflow>this.kubeflowTrialConfig;
        }else if(this.kubeflowClusterConfig.operator === 'pytorch-operator'){
            kubeflowTrialConfig = <KubeflowTrialConfigPytorch>this.kubeflowTrialConfig;
        }else {
            throw Error(`operator ${this.kubeflowClusterConfig.operator} is invalid`)
        }

        const trialJobId: string = uniqueString(5);
        const curTrialSequenceId: number = this.generateSequenceId();
        // Set trial's NFS working folder
        const trialWorkingFolder: string = path.join(this.CONTAINER_MOUNT_PATH, 'nni', getExperimentId(), trialJobId);
        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        //create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${path.dirname(trialLocalTempFolder)}`);
        await cpp.exec(`cp -r ${kubeflowTrialConfig.codeDir} ${trialLocalTempFolder}`);
        const runScriptContent : string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });
        // Create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${trialLocalTempFolder}`);

        // Write worker file content run_worker.sh to local tmp folders
        if(kubeflowTrialConfig.worker) {
            const workerRunScriptContent: string = this.generateRunScript(trialJobId, trialWorkingFolder, 
                    kubeflowTrialConfig.worker.command, curTrialSequenceId.toString(), 'worker', kubeflowTrialConfig.worker.gpuNum);

            await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_worker.sh'), workerRunScriptContent, { encoding: 'utf8' });
        }
        // Write parameter server file content run_ps.sh to local tmp folders
        if(this.kubeflowClusterConfig.operator === 'tf-operator') {
            let tensorflowTrialConfig: KubeflowTrialConfigTensorflow = <KubeflowTrialConfigTensorflow>this.kubeflowTrialConfig;
            if(tensorflowTrialConfig.ps){
                const psRunScriptContent: string = this.generateRunScript(trialJobId, trialWorkingFolder, 
                    tensorflowTrialConfig.ps.command, curTrialSequenceId.toString(), 'ps', tensorflowTrialConfig.ps.gpuNum);
                await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_ps.sh'), psRunScriptContent, { encoding: 'utf8' });
            }
        }
        else if(this.kubeflowClusterConfig.operator === 'pytorch-operator') {
            let pytorchTrialConfig: KubeflowTrialConfigPytorch = <KubeflowTrialConfigPytorch>this.kubeflowTrialConfig;
            if(pytorchTrialConfig.master){
                const masterRunScriptContent: string = this.generateRunScript(trialJobId, trialWorkingFolder, 
                    pytorchTrialConfig.master.command, curTrialSequenceId.toString(), 'master', pytorchTrialConfig.master.gpuNum);
                await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_master.sh'), masterRunScriptContent, { encoding: 'utf8' });
            }
        }
        // Write file content ( parameter.cfg ) to local tmp folders
        const trialForm : TrialJobApplicationForm = (<TrialJobApplicationForm>form)
        if(trialForm && trialForm.hyperParameters) {
            await fs.promises.writeFile(path.join(trialLocalTempFolder, generateParamFileName(trialForm.hyperParameters)), 
                            trialForm.hyperParameters.value, { encoding: 'utf8' });
        }
        const kubeflowJobName = `nni-exp-${this.experimentId}-trial-${trialJobId}`.toLowerCase();
        
        const workerPodResources : any = {};
        if(kubeflowTrialConfig.worker) {
            workerPodResources.requests = this.generatePodResource(kubeflowTrialConfig.worker.memoryMB, kubeflowTrialConfig.worker.cpuNum, 
                kubeflowTrialConfig.worker.gpuNum)
        }
        workerPodResources.limits = Object.assign({}, workerPodResources.requests);

        let nonWorkerResources : any = {};
        if(this.kubeflowClusterConfig.operator === 'tf-operator') {
            let tensorflowTrialConfig: KubeflowTrialConfigTensorflow = <KubeflowTrialConfigTensorflow>this.kubeflowTrialConfig;
            if (tensorflowTrialConfig.ps) {
                nonWorkerResources.requests = this.generatePodResource(tensorflowTrialConfig.ps.memoryMB, tensorflowTrialConfig.ps.cpuNum, 
                    tensorflowTrialConfig.ps.gpuNum)
                    nonWorkerResources.limits = Object.assign({}, nonWorkerResources.requests); 
            }
        }else if(this.kubeflowClusterConfig.operator === 'pytorch-operator'){
            let pyTorchTrialConfig: KubeflowTrialConfigPytorch = <KubeflowTrialConfigPytorch>this.kubeflowTrialConfig;
            nonWorkerResources.requests = this.generatePodResource(pyTorchTrialConfig.master.memoryMB, pyTorchTrialConfig.master.cpuNum, 
                pyTorchTrialConfig.master.gpuNum)
                nonWorkerResources.limits = Object.assign({}, nonWorkerResources.requests); 
            
        }       

        //The output url used in trialJobDetail
        let trialJobOutputUrl: string = '';

        assert(!this.kubeflowClusterConfig.storage 
            || this.kubeflowClusterConfig.storage === 'azureStorage' 
            || this.kubeflowClusterConfig.storage === 'nfs');

        if(this.kubeflowClusterConfig.storage === 'azureStorage') {
            try{
                //upload local files to azure storage
                await AzureStorageClientUtility.uploadDirectory(this.azureStorageClient, 
                    `nni/${getExperimentId()}/${trialJobId}`, this.azureStorageShare, `${trialLocalTempFolder}`);

                trialJobOutputUrl = `https://${this.azureStorageAccountName}.file.core.windows.net/${this.azureStorageShare}/${path.join('nni', getExperimentId(), trialJobId, 'output')}`
            }catch(error){
                this.log.error(error);
                return Promise.reject(error);
            }
        } else if(this.kubeflowClusterConfig.storage === 'nfs' || this.kubeflowClusterConfig.storage === undefined) {
            let nfsKubeflowClusterConfig: KubeflowClusterConfigNFS = <KubeflowClusterConfigNFS>this.kubeflowClusterConfig;
            // Creat work dir for current trial in NFS directory 
            await cpp.exec(`mkdir -p ${this.trialLocalNFSTempFolder}/nni/${getExperimentId()}/${trialJobId}`);
            // Copy code files from local dir to NFS mounted dir
            await cpp.exec(`cp -r ${trialLocalTempFolder}/* ${this.trialLocalNFSTempFolder}/nni/${getExperimentId()}/${trialJobId}/.`);
        
            const nfsConfig: NFSConfig = nfsKubeflowClusterConfig.nfs;
            trialJobOutputUrl = `nfs://${nfsConfig.server}:${path.join(nfsConfig.path, 'nni', getExperimentId(), trialJobId, 'output')}`
        }

        const trialJobDetail: KubeflowTrialJobDetail = new KubeflowTrialJobDetail(
            trialJobId,
            'WAITING',
            Date.now(),
            trialWorkingFolder,
            form,
            kubeflowJobName,
            curTrialSequenceId,
            trialJobOutputUrl
        );

        // Generate kubeflow job resource config object        
        const kubeflowJobConfig: any = this.generateKubeflowJobConfig(trialJobId, trialWorkingFolder, kubeflowJobName, workerPodResources, nonWorkerResources);

        // Create kubeflow job based on generated kubeflow job resource config
        await this.operatorClient.createKubeflowJob(kubeflowJobConfig);

        // Set trial job detail until create Kubeflow job successfully 
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    public generatePodResource(memory: number, cpuNum: number, gpuNum: number) {
        return {
            'memory': `${memory}Mi`,
            'cpu': `${cpuNum}`,
            'nvidia.com/gpu': `${gpuNum}`
        }
    }

    public updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<TrialJobDetail> {
        throw new MethodNotImplementedError();
    }

    public listTrialJobs(): Promise<TrialJobDetail[]> {
        const jobs: TrialJobDetail[] = [];
        
        this.trialJobsMap.forEach(async (value: KubeflowTrialJobDetail, key: string) => {
            if (value.form.jobType === 'TRIAL') {
                jobs.push(await this.getTrialJob(key));
            }
        });

        return Promise.resolve(jobs);
    }

    public getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        const kubeflowTrialJob: TrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (!kubeflowTrialJob) {
            return Promise.reject(`trial job ${trialJobId} not found`)
        }        

        return Promise.resolve(kubeflowTrialJob);
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void) {
        this.metricsEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void) {
        this.metricsEmitter.off('metric', listener);
    }
 
    public get isMultiPhaseJobSupported(): boolean {
        return false;
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trialJobDetail : KubeflowTrialJobDetail | undefined =  this.trialJobsMap.get(trialJobId);
        if(!trialJobDetail) {
            const errorMessage: string = `CancelTrialJob: trial job id ${trialJobId} not found`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }        
        if(!this.operatorClient) {
            const errorMessage: string = `CancelTrialJob: trial job id ${trialJobId} failed because operatorClient is undefined`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }

        try {
            await this.operatorClient.deleteKubeflowJob(new Map(
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
            
            case TrialConfigMetadataKey.KUBEFLOW_CLUSTER_CONFIG:
                let kubeflowClusterJsonObject = JSON.parse(value);
                let kubeflowClusterConfigBase: KubeflowClusterConfigBase 
                        = new KubeflowClusterConfigBase(kubeflowClusterJsonObject.operator, kubeflowClusterJsonObject.apiVersion, kubeflowClusterJsonObject.storage);
                
                if(kubeflowClusterConfigBase && kubeflowClusterConfigBase.storage === 'azureStorage') {
                    const azureKubeflowClusterConfig: KubeflowClusterConfigAzure = 
                        new KubeflowClusterConfigAzure(kubeflowClusterJsonObject.operator, 
                            kubeflowClusterJsonObject.apiVersion,
                            kubeflowClusterJsonObject.keyVault, 
                            kubeflowClusterJsonObject.azureStorage, kubeflowClusterJsonObject.storage);

                    const vaultName = azureKubeflowClusterConfig.keyVault.vaultName;
                    const valutKeyName = azureKubeflowClusterConfig.keyVault.name;
                    this.azureStorageAccountName = azureKubeflowClusterConfig.azureStorage.accountName;
                    this.azureStorageShare = azureKubeflowClusterConfig.azureStorage.azureShare;
                    try {
                        const result = await cpp.exec(`az keyvault secret show --name ${valutKeyName} --vault-name ${vaultName}`);
                        if(result.stderr) {
                            const errorMessage: string = result.stderr;
                            this.log.error(errorMessage);
                            return Promise.reject(errorMessage);
                        }
                        const storageAccountKey =JSON.parse(result.stdout).value;
                        //create storage client
                        this.azureStorageClient = azure.createFileService(this.azureStorageAccountName, storageAccountKey);
                        await AzureStorageClientUtility.createShare(this.azureStorageClient, this.azureStorageShare);
                        //create sotrage secret
                        this.azureStorageSecretName = 'nni-secret-' + uniqueString(8).toLowerCase();
                        await this.genericK8sClient.createSecret(
                            {
                                apiVersion: 'v1',
                                kind: 'Secret',
                                metadata: { 
                                    name: this.azureStorageSecretName,
                                    namespace: 'default',
                                    labels: {
                                        app: this.NNI_KUBEFLOW_TRIAL_LABEL,
                                        expId: getExperimentId()
                                    }
                                },
                                type: 'Opaque',
                                data: {
                                    azurestorageaccountname: base64.encode(this.azureStorageAccountName),
                                    azurestorageaccountkey: base64.encode(storageAccountKey)
                                }
                            }
                        );
                    } catch(error) {
                        this.log.error(error);
                        throw new Error(error);
                    }

                    this.kubeflowClusterConfig = azureKubeflowClusterConfig;
                } else if(kubeflowClusterConfigBase && (kubeflowClusterConfigBase.storage === 'nfs' || kubeflowClusterConfigBase.storage === undefined)) {
                    //Check and mount NFS mount point here
                    //If storage is undefined, the default value is nfs
                    const nfsKubeflowClusterConfig: KubeflowClusterConfigNFS = 
                                 new KubeflowClusterConfigNFS(kubeflowClusterJsonObject.operator, 
                                            kubeflowClusterJsonObject.apiVersion,
                                            kubeflowClusterJsonObject.nfs, 
                                            kubeflowClusterJsonObject.storage);

                    await cpp.exec(`mkdir -p ${this.trialLocalNFSTempFolder}`);
                    const nfsServer: string = nfsKubeflowClusterConfig.nfs.server;
                    const nfsPath: string = nfsKubeflowClusterConfig.nfs.path;

                    try {
                        await cpp.exec(`sudo mount ${nfsServer}:${nfsPath} ${this.trialLocalNFSTempFolder}`);
                    } catch(error) {
                        const mountError: string = `Mount NFS ${nfsServer}:${nfsPath} to ${this.trialLocalNFSTempFolder} failed, error is ${error}`;
                        this.log.error(mountError);
                        throw new Error(mountError);
                    }
                    this.kubeflowClusterConfig = nfsKubeflowClusterConfig;
                } else {
                    const error: string = `kubeflowClusterConfig format error!`;
                    this.log.error(error);
                    throw new Error(error);
                }

                this.operatorClient = KubeflowOperatorClient.generateOperatorClient(this.kubeflowClusterConfig.operator,
                                                                                     this.kubeflowClusterConfig.apiVersion);
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG:
                if (!this.kubeflowClusterConfig){
                    this.log.error('kubeflow cluster config is not initialized');
                    return Promise.reject(new Error('kubeflow cluster config is not initialized'));                    
                }

                assert(this.kubeflowClusterConfig !== undefined)
                let kubeflowTrialJsonObjsect = JSON.parse(value);
                if(this.kubeflowClusterConfig.operator === 'tf-operator'){
                    this.kubeflowTrialConfig = new KubeflowTrialConfigTensorflow(kubeflowTrialJsonObjsect.codeDir, 
                        kubeflowTrialJsonObjsect.worker, kubeflowTrialJsonObjsect.ps);
                }else if(this.kubeflowClusterConfig.operator === 'pytorch-operator'){
                    this.kubeflowTrialConfig = new KubeflowTrialConfigPytorch(kubeflowTrialJsonObjsect.codeDir, 
                        kubeflowTrialJsonObjsect.master, kubeflowTrialJsonObjsect.worker);
                }

                if (!this.kubeflowTrialConfig){
                    this.log.error('kubeflow kubeflow TrialConfig is not initialized');
                    return Promise.reject(new Error('kubeflow kubeflow TrialConfig is not initialized'));                    
                }

                // Validate to make sure codeDir doesn't have too many files
                try {
                    await validateCodeDir(this.kubeflowTrialConfig.codeDir);
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

    public getClusterMetadata(key: string): Promise<string> {
        return Promise.resolve('');
    }

    public async cleanUp(): Promise<void> {
        this.stopping = true;

        // First, cancel all running kubeflow jobs
        for(let [trialJobId, kubeflowTrialJob] of this.trialJobsMap) {
            if(['RUNNING', 'WAITING', 'UNKNOWN'].includes(kubeflowTrialJob.status)) {
                try {
                    await this.cancelTrialJob(trialJobId);
                } catch(error) {} // DONT throw error during cleanup
                kubeflowTrialJob.status = 'SYS_CANCELED';
            }
        }
        
        // Delete all kubeflow jobs whose expId label is current experiment id 
        try {
            if(this.operatorClient) {
                await this.operatorClient.deleteKubeflowJob(new Map(
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
        const restServer: KubeflowJobRestServer = component.get(KubeflowJobRestServer);
        try {
            await restServer.stop();
            this.log.info('Kubeflow Training service rest server stopped successfully.');
        } catch (error) {
            this.log.error(`Kubeflow Training service rest server stopped failed, error: ${error.message}`);
            Promise.reject(error);
        }

        return Promise.resolve();
    }

    public get MetricsEmitter() : EventEmitter {
        return this.metricsEmitter;
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
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if(!this.kubeflowTrialConfig) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        if(!this.operatorClient) {
            throw new Error('Kubeflow operator client is not initialized');
        }

        const replicaSpecsObj: any = {};
        let replicaSpecsObjMap = new Map<string, object>();

        if(this.kubeflowClusterConfig.operator === 'tf-operator') {
            let tensorflowTrialConfig: KubeflowTrialConfigTensorflow = <KubeflowTrialConfigTensorflow>this.kubeflowTrialConfig;
            replicaSpecsObj.Worker = this.generateReplicaConfig(trialWorkingFolder, tensorflowTrialConfig.worker.replicas, 
                tensorflowTrialConfig.worker.image, 'run_worker.sh', workerPodResources);
            
            if (tensorflowTrialConfig.ps){
                replicaSpecsObj.Ps = this.generateReplicaConfig(trialWorkingFolder, tensorflowTrialConfig.ps.replicas, 
                    tensorflowTrialConfig.ps.image, 'run_ps.sh', nonWorkerPodResources);
            }
            replicaSpecsObjMap.set(this.operatorClient.jobKind, {'tfReplicaSpecs': replicaSpecsObj})
        }
        else if(this.kubeflowClusterConfig.operator === 'pytorch-operator') {
            let pytorchTrialConfig: KubeflowTrialConfigPytorch = <KubeflowTrialConfigPytorch>this.kubeflowTrialConfig;
            if(pytorchTrialConfig.worker) {
                replicaSpecsObj.Worker = this.generateReplicaConfig(trialWorkingFolder, pytorchTrialConfig.worker.replicas, 
                    pytorchTrialConfig.worker.image, 'run_worker.sh', workerPodResources);
            }
            replicaSpecsObj.Master = this.generateReplicaConfig(trialWorkingFolder, pytorchTrialConfig.master.replicas, 
                pytorchTrialConfig.master.image, 'run_master.sh', nonWorkerPodResources);
            
            replicaSpecsObjMap.set(this.operatorClient.jobKind, {'pytorchReplicaSpecs': replicaSpecsObj})
        }

        return {
            apiVersion: `kubeflow.org/${this.operatorClient.apiVersion}`,
            kind: this.operatorClient.jobKind,
            metadata: { 
                name: kubeflowJobName,
                namespace: 'default',
                labels: {
                    app: this.NNI_KUBEFLOW_TRIAL_LABEL,
                    expId: getExperimentId(),
                    trialId: trialJobId
                }
            },
            spec: replicaSpecsObjMap.get(this.operatorClient.jobKind)
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
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if(!this.kubeflowTrialConfig) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        if(!this.operatorClient) {
            throw new Error('Kubeflow operator client is not initialized');
        }

        let volumeSpecMap = new Map<string, object>();
        if(this.kubeflowClusterConfig.storage && this.kubeflowClusterConfig.storage === 'azureStorage'){
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
            let nfsKubeflowClusterConfig: KubeflowClusterConfigNFS = <KubeflowClusterConfigNFS> this.kubeflowClusterConfig;
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
            replicas: replicaNumber,
            template: {
                metadata: {
                    creationTimestamp: null
                },
                spec: {
                    containers: [
                    {
                        // Kubeflow tensorflow operator requires that containers' name must be tensorflow
                        // TODO: change the name based on operator's type
                        name: this.operatorClient.containerName,
                        image: replicaImage,
                        args: ["sh", `${path.join(trialWorkingFolder, runScriptFile)}`],
                        volumeMounts: [
                        {
                            name: 'nni-vol',
                            mountPath: this.CONTAINER_MOUNT_PATH
                        }],
                        resources: podResources
                    }],
                    restartPolicy: 'ExitCode',
                    volumes: volumeSpecMap.get('nniVolumes')
                }
            }
        };
    }

    /**
     * Genereate run script for different roles(like worker or ps)
     * @param trialJobId trial job id
     * @param trialWorkingFolder working folder
     * @param command 
     * @param trialSequenceId sequence id
     */
    private generateRunScript(trialJobId: string, trialWorkingFolder: string, 
                command: string, trialSequenceId: string, roleType: DistTrainRole, gpuNum: number): string {
        const runScriptLines: string[] = [];

        runScriptLines.push('#!/bin/bash');
        runScriptLines.push('export NNI_PLATFORM=kubeflow');
        runScriptLines.push(`export NNI_SYS_DIR=$PWD/nni/${trialJobId}`);
        runScriptLines.push(`export NNI_OUTPUT_DIR=${path.join(trialWorkingFolder, 'output', `${roleType}_output`)}`);
        runScriptLines.push('export MULTI_PHASE=false');
        runScriptLines.push(`export NNI_TRIAL_JOB_ID=${trialJobId}`);
        runScriptLines.push(`export NNI_EXP_ID=${getExperimentId()}`);
        runScriptLines.push(`export NNI_CODE_DIR=${trialWorkingFolder}`);
        runScriptLines.push(`export NNI_TRIAL_SEQ_ID=${trialSequenceId}`);
        
        // Nvidia devcie plugin for K8S has a known issue that requesting zero GPUs allocates all GPUs
        // Refer https://github.com/NVIDIA/k8s-device-plugin/issues/61
        // So we have to explicitly set CUDA_VISIBLE_DEVICES to empty if user sets gpuNum to 0 in NNI config file
        if(gpuNum === 0) {
            runScriptLines.push(`export CUDA_VISIBLE_DEVICES=''`);
        }

        const nniManagerIp = this.nniManagerIpConfig?this.nniManagerIpConfig.nniManagerIp:getIPV4Address();
        runScriptLines.push('mkdir -p $NNI_SYS_DIR');
        runScriptLines.push('mkdir -p $NNI_OUTPUT_DIR');
        runScriptLines.push('cp -rT $NNI_CODE_DIR $NNI_SYS_DIR');
        runScriptLines.push('cd $NNI_SYS_DIR');
        runScriptLines.push('sh install_nni.sh # Check and install NNI pkg');
        runScriptLines.push(`python3 -m nni_trial_tool.trial_keeper --trial_command '${command}' `
        + `--nnimanager_ip '${nniManagerIp}' --nnimanager_port '${this.kubeflowRestServerPort}' `
        + `1>$NNI_OUTPUT_DIR/trialkeeper_stdout 2>$NNI_OUTPUT_DIR/trialkeeper_stderr`);

        return runScriptLines.join('\n');
    }

    private generateSequenceId(): number {
        if (this.nextTrialSequenceId === -1) {
            this.nextTrialSequenceId = getInitTrialSequenceId();
        }

        return this.nextTrialSequenceId++;
    }
}

export { KubeflowTrainingService }
