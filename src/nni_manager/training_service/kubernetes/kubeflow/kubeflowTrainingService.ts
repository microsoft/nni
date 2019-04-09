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
import { KubeflowClusterConfigNFS, KubeflowClusterConfigAzure,
     KubeflowTrialConfigPytorch, KubeflowTrialConfigTensorflow, KubeflowClusterConfigFactory, KubeflowTrialConfigFactory,
     KubeflowTrialConfig, KubeflowClusterConfig } from './kubeflowConfig';
import { NFSConfig } from '../kubernetesConfig'
import { KubernetesTrialJobDetail } from '../kubernetesData';
import { KubeflowJobRestServer } from './kubeflowJobRestServer';
import { validateCodeDir } from '../../common/util';
import { AzureStorageClientUtility } from '../azureStorageClientUtils';
import { KubeflowOperatorClient } from './kubeflowApiClient';
import { KubernetesTrainingService } from '../kubernetesTrainingService'
import { KubeflowJobInfoCollector } from './kubeflowJobInfoCollector';


/**
 * Training Service implementation for Kubeflow
 * Refer https://github.com/kubeflow/kubeflow for more info about Kubeflow
 */
@component.Singleton
class KubeflowTrainingService extends KubernetesTrainingService implements KubernetesTrainingService {
    private kubeflowClusterConfig?: KubeflowClusterConfig;
    private kubeflowTrialConfig?: KubeflowTrialConfig;
    private kubeflowJobInfoCollector: KubeflowJobInfoCollector;

    constructor() {
        super();  
        this.kubeflowJobInfoCollector = new KubeflowJobInfoCollector(this.trialJobsMap);
        this.experimentId = getExperimentId();      
        this.nextTrialSequenceId = -1;
        this.log.info('Construct Kubeflow training service.');
    }

    public async run(): Promise<void> {
        this.log.info('Run Kubeflow training service.');
        this.kubernetesJobRestServer = component.get(KubeflowJobRestServer);
        if(!this.kubernetesJobRestServer) {
            throw new Error('kubernetesJobRestServer not initialized!');
        }
        await this.kubernetesJobRestServer.start();
        this.kubernetesJobRestServer.setEnableVersionCheck = this.versionCheck;
        this.log.info(`Kubeflow Training service rest server listening on: ${this.kubernetesJobRestServer.endPoint}`);
        while (!this.stopping) {
            // collect metrics for Kubeflow jobs by interacting with Kubernetes API server  
            await delay(3000);
            await this.kubeflowJobInfoCollector.retrieveTrialStatus(this.kubernetesCRDClient);
            if(this.kubernetesJobRestServer.getErrorMessage) {
                throw new Error(this.kubernetesJobRestServer.getErrorMessage);
                this.stopping = true;
            }
        }
        this.log.info('Kubeflow training service exit.');
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        if(!this.kubernetesCRDClient) {
            throw new Error('Kubeflow job operator client is undefined');
        }

        if(!this.kubernetesRestServerPort) {
            const restServer: KubeflowJobRestServer = component.get(KubeflowJobRestServer);
            this.kubernetesRestServerPort = restServer.clusterRestServerPort;
        }
        const trialJobId: string = uniqueString(5);
        const trialWorkingFolder: string = path.join(this.CONTAINER_MOUNT_PATH, 'nni', getExperimentId(), trialJobId);
        const kubeflowJobName = `nni-exp-${this.experimentId}-trial-${trialJobId}`.toLowerCase();
        const curTrialSequenceId: number = this.generateSequenceId();
        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        //prepare the runscript
        await this.prepareRunScript(trialLocalTempFolder, trialJobId, trialWorkingFolder, curTrialSequenceId, form);
        //upload files to sotrage
        const trialJobOutputUrl: string = await this.uploadCodeFiles(trialJobId, trialLocalTempFolder);
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
       
        // Generate kubeflow job resource config object        
        const kubeflowJobConfig: any = await this.prepareKubeflowConfig(trialJobId, trialWorkingFolder, kubeflowJobName);
        // Create kubeflow job based on generated kubeflow job resource config
        await this.kubernetesCRDClient.createKubernetesJob(kubeflowJobConfig);

        // Set trial job detail until create Kubeflow job successfully 
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
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

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

        return Promise.resolve(trialJobOutputUrl);
    }
    
    private async prepareRunScript(trialLocalTempFolder: string, trialJobId: string, trialWorkingFolder: string, curTrialSequenceId: number, form: JobApplicationForm): Promise<void> {
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
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
           const workerRunScriptContent: string = await this.generateRunScript('kubeflow', trialJobId, trialWorkingFolder, 
                   kubeflowTrialConfig.worker.command, curTrialSequenceId.toString(), 'worker', kubeflowTrialConfig.worker.gpuNum);

           await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_worker.sh'), workerRunScriptContent, { encoding: 'utf8' });
       }
       // Write parameter server file content run_ps.sh to local tmp folders
       if(this.kubeflowClusterConfig.operator === 'tf-operator') {
           let tensorflowTrialConfig: KubeflowTrialConfigTensorflow = <KubeflowTrialConfigTensorflow>this.kubeflowTrialConfig;
           if(tensorflowTrialConfig.ps){
               const psRunScriptContent: string = await this.generateRunScript('kubeflow', trialJobId, trialWorkingFolder, 
                   tensorflowTrialConfig.ps.command, curTrialSequenceId.toString(), 'ps', tensorflowTrialConfig.ps.gpuNum);
               await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_ps.sh'), psRunScriptContent, { encoding: 'utf8' });
           }
       }
       else if(this.kubeflowClusterConfig.operator === 'pytorch-operator') {
           let pytorchTrialConfig: KubeflowTrialConfigPytorch = <KubeflowTrialConfigPytorch>this.kubeflowTrialConfig;
           if(pytorchTrialConfig.master){
               const masterRunScriptContent: string = await this.generateRunScript('kubeflow', trialJobId, trialWorkingFolder, 
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
    }
    
    private async prepareKubeflowConfig(trialJobId: string, trialWorkingFolder: string, kubeflowJobName: string): Promise<any> {
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if(!this.kubeflowTrialConfig) {
            throw new Error('Kubeflow trial config is not initialized');
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

        // Generate kubeflow job resource config object        
        const kubeflowJobConfig: any = this.generateKubeflowJobConfig(trialJobId, trialWorkingFolder, kubeflowJobName, workerPodResources, nonWorkerResources);

        return Promise.resolve(kubeflowJobConfig);
    } 

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;
            
            case TrialConfigMetadataKey.KUBEFLOW_CLUSTER_CONFIG:
                let kubeflowClusterJsonObject = JSON.parse(value);
                this.kubeflowClusterConfig = KubeflowClusterConfigFactory.generateKubeflowClusterConfig(kubeflowClusterJsonObject);
                if(this.kubeflowClusterConfig.storageType === 'azureStorage') {
                    let azureKubeflowClusterConfig = <KubeflowClusterConfigAzure>this.kubeflowClusterConfig;
                    this.azureStorageAccountName = azureKubeflowClusterConfig.azureStorage.accountName;
                    this.azureStorageShare = azureKubeflowClusterConfig.azureStorage.azureShare;
                    await this.createAzureStorage(
                        azureKubeflowClusterConfig.keyVault.vaultName,
                        azureKubeflowClusterConfig.keyVault.name,
                        azureKubeflowClusterConfig.azureStorage.accountName,
                        azureKubeflowClusterConfig.azureStorage.azureShare
                    );
                } else if(this.kubeflowClusterConfig.storageType === 'nfs') {
                    let nfsKubeflowClusterConfig = <KubeflowClusterConfigNFS>this.kubeflowClusterConfig;
                    await this.createNFSStorage(
                        nfsKubeflowClusterConfig.nfs.server,
                        nfsKubeflowClusterConfig.nfs.path
                    );
                } 
                this.kubernetesCRDClient = KubeflowOperatorClient.generateOperatorClient(this.kubeflowClusterConfig.operator,
                                                                                     this.kubeflowClusterConfig.apiVersion);
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG:
                if (!this.kubeflowClusterConfig){
                    this.log.error('kubeflow cluster config is not initialized');
                    return Promise.reject(new Error('kubeflow cluster config is not initialized'));                    
                }

                assert(this.kubeflowClusterConfig !== undefined)
                let kubeflowTrialJsonObjsect = JSON.parse(value);
                this.kubeflowTrialConfig = KubeflowTrialConfigFactory.generateKubeflowTrialConfig(
                    kubeflowTrialJsonObjsect, 
                    this.kubeflowClusterConfig.operator
                );

                // Validate to make sure codeDir doesn't have too many files
                try {
                    await validateCodeDir(this.kubeflowTrialConfig.codeDir);
                } catch(error) {
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
                break;
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
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if(!this.kubeflowTrialConfig) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        if(!this.kubernetesCRDClient) {
            throw new Error('Kubeflow operator client is not initialized');
        }

        const replicaSpecsObj: any = {};
        let replicaSpecsObjMap = new Map<string, object>();

        if(this.kubeflowTrialConfig.operatorType === 'tf-operator') {
            let tensorflowTrialConfig: KubeflowTrialConfigTensorflow = <KubeflowTrialConfigTensorflow>this.kubeflowTrialConfig;
            replicaSpecsObj.Worker = this.generateReplicaConfig(trialWorkingFolder, tensorflowTrialConfig.worker.replicas, 
                tensorflowTrialConfig.worker.image, 'run_worker.sh', workerPodResources);
            
            if (tensorflowTrialConfig.ps){
                replicaSpecsObj.Ps = this.generateReplicaConfig(trialWorkingFolder, tensorflowTrialConfig.ps.replicas, 
                    tensorflowTrialConfig.ps.image, 'run_ps.sh', nonWorkerPodResources);
            }
            replicaSpecsObjMap.set(this.kubernetesCRDClient.jobKind, {'tfReplicaSpecs': replicaSpecsObj})
        }
        else if(this.kubeflowTrialConfig.operatorType === 'pytorch-operator') {
            let pytorchTrialConfig: KubeflowTrialConfigPytorch = <KubeflowTrialConfigPytorch>this.kubeflowTrialConfig;
            if(pytorchTrialConfig.worker) {
                replicaSpecsObj.Worker = this.generateReplicaConfig(trialWorkingFolder, pytorchTrialConfig.worker.replicas, 
                    pytorchTrialConfig.worker.image, 'run_worker.sh', workerPodResources);
            }
            replicaSpecsObj.Master = this.generateReplicaConfig(trialWorkingFolder, pytorchTrialConfig.master.replicas, 
                pytorchTrialConfig.master.image, 'run_master.sh', nonWorkerPodResources);
            
            replicaSpecsObjMap.set(this.kubernetesCRDClient.jobKind, {'pytorchReplicaSpecs': replicaSpecsObj})
        }

        return {
            apiVersion: `kubeflow.org/${this.kubernetesCRDClient.apiVersion}`,
            kind: this.kubernetesCRDClient.jobKind,
            metadata: { 
                name: kubeflowJobName,
                namespace: 'default',
                labels: {
                    app: this.NNI_KUBERNETES_TRIAL_LABEL,
                    expId: getExperimentId(),
                    trialId: trialJobId
                }
            },
            spec: replicaSpecsObjMap.get(this.kubernetesCRDClient.jobKind)
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

        if(!this.kubernetesCRDClient) {
            throw new Error('Kubeflow operator client is not initialized');
        }

        let volumeSpecMap = new Map<string, object>();
        if(this.kubeflowClusterConfig.storageType === 'azureStorage'){
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
                        name: this.kubernetesCRDClient.containerName,
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
}

export { KubeflowTrainingService }
