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
    TrialJobDetail, TrialJobMetric
} from '../../common/trainingService';
import { delay, generateParamFileName, getExperimentRootDir, getIPV4Address, uniqueString } from '../../common/utils';
import { KubeflowClusterConfig, kubeflowOperatorMap, KubeflowTrialConfig, NFSConfig } from './kubeflowConfig';
import { KubeflowTrialJobDetail } from './kubeflowData';
import { KubeflowJobRestServer } from './kubeflowJobRestServer';
import { KubeflowJobInfoCollector } from './kubeflowJobInfoCollector';

var yaml = require('node-yaml');

type DistTrainRole = 'worker' | 'ps';

/**
 * Training Service implementation for Kubeflow
 * Refer https://github.com/kubeflow/kubeflow for more info about Kubeflow
 */
@component.Singleton
class KubeflowTrainingService implements TrainingService {
    private readonly NNI_KUBEFLOW_TRIAL_LABEL = 'nni-kubeflow-trial';
    private readonly log!: Logger;
    private readonly metricsEmitter: EventEmitter;
    private readonly trialJobsMap: Map<string, KubeflowTrialJobDetail>;
    /**  experiment root dir in NFS */
    private readonly trialLocalNFSTempFolder: string;
    private stopping: boolean = false;
    private experimentId! : string;
    private nextTrialSequenceId: number;
    private kubeflowClusterConfig?: KubeflowClusterConfig;
    private kubeflowTrialConfig?: KubeflowTrialConfig;
    private kubeflowJobInfoCollector: KubeflowJobInfoCollector;
    private kubeflowRestServerPort?: number;
    private kubeflowJobPlural?: string;
    private readonly CONTAINER_MOUNT_PATH: string;    
    
    constructor() {        
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, KubeflowTrialJobDetail>();
        this.kubeflowJobInfoCollector = new KubeflowJobInfoCollector(this.trialJobsMap);
        this.trialLocalNFSTempFolder = path.join(getExperimentRootDir(), 'trials-nfs-tmp');
        this.experimentId = getExperimentId();      
        this.nextTrialSequenceId = -1;
        this.CONTAINER_MOUNT_PATH = '/tmp/nfs';
    }

    public async run(): Promise<void> {
        const restServer: KubeflowJobRestServer = component.get(KubeflowJobRestServer);
        await restServer.start();
        this.log.info(`Kubeflow Training service rest server listening on: ${restServer.endPoint}`);
        while (!this.stopping) {
            // collect metrics by calling 'kubectl get' command on Kubeflow jobs 
            await delay(3000);
            await this.kubeflowJobInfoCollector.retrieveTrialStatus();            
        }
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if(!this.kubeflowTrialConfig || !this.kubeflowTrialConfig.worker) {
            throw new Error('Kubeflow trial config or worker config is not initialized');
        }

        if(!this.kubeflowJobPlural) {
            throw new Error('Kubeflow job plural name is undefined');
        }

        if(!this.kubeflowRestServerPort) {
            const restServer: KubeflowJobRestServer = component.get(KubeflowJobRestServer);
            this.kubeflowRestServerPort = restServer.clusterRestServerPort;
        }

        const trialJobId: string = uniqueString(5);
        const curTrialSequenceId: number = this.generateSequenceId();
        // Set trial's NFS working folder
        const trialWorkingFolder: string = path.join(this.CONTAINER_MOUNT_PATH, 'nni', getExperimentId(), trialJobId);
        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        //create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${path.dirname(trialLocalTempFolder)}`);
        await cpp.exec(`cp -r ${this.kubeflowTrialConfig.codeDir} ${trialLocalTempFolder}`);

        const runScriptContent : string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });

        // Create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${trialLocalTempFolder}`);

        // Write worker file content run_worker.sh to local tmp folders
        if(this.kubeflowTrialConfig.worker) {
            const workerRunScriptContent: string = this.genereateRunScript(trialJobId, trialWorkingFolder, 
                    this.kubeflowTrialConfig.worker.command, curTrialSequenceId.toString(), 'worker');

            await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_worker.sh'), workerRunScriptContent, { encoding: 'utf8' });
        }

        // Write parameter server file content run_ps.sh to local tmp folders
        if(this.kubeflowTrialConfig.ps) {
            const psRunScriptContent: string = this.genereateRunScript(trialJobId, trialWorkingFolder, 
                this.kubeflowTrialConfig.ps.command, curTrialSequenceId.toString(), 'ps');

            await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run_ps.sh'), psRunScriptContent, { encoding: 'utf8' });
        }

        // Write file content ( parameter.cfg ) to local tmp folders
        const trialForm : TrialJobApplicationForm = (<TrialJobApplicationForm>form)
        if(trialForm && trialForm.hyperParameters) {
            await fs.promises.writeFile(path.join(trialLocalTempFolder, generateParamFileName(trialForm.hyperParameters)), 
                            trialForm.hyperParameters.value, { encoding: 'utf8' });
        }

        const kubeflowJobYamlPath = path.join(trialLocalTempFolder, `kubeflow-job-${trialJobId}.yaml`);
        const kubeflowJobName = `nni-exp-${this.experimentId}-trial-${trialJobId}`.toLowerCase();
        const workerPodResources : any = {};
        workerPodResources.requests = {
            'memory': `${this.kubeflowTrialConfig.worker.memoryMB}Mi`,
            'cpu': `${this.kubeflowTrialConfig.worker.cpuNum}`,
            'nvidia.com/gpu': `${this.kubeflowTrialConfig.worker.gpuNum}`
        }
        workerPodResources.limits = Object.assign({}, workerPodResources.requests);

        let psPodResources : any = undefined;
        if(this.kubeflowTrialConfig.ps) {
            psPodResources = {};
            psPodResources.requests = {
                'memory': `${this.kubeflowTrialConfig.ps.memoryMB}Mi`,
                'cpu': `${this.kubeflowTrialConfig.ps.cpuNum}`,
                'nvidia.com/gpu': `${this.kubeflowTrialConfig.ps.gpuNum}`
            }
            psPodResources.limits = Object.assign({}, psPodResources.requests);
        }        

        // Generate kubeflow job resource yaml file for K8S
        yaml.write(
            kubeflowJobYamlPath,
            this.generateKubeflowJobConfig(trialJobId, trialWorkingFolder, kubeflowJobName, workerPodResources, psPodResources),
            'utf-8'
        );

        // Creat work dir for current trial in NFS directory 
        await cpp.exec(`mkdir -p ${this.trialLocalNFSTempFolder}/nni/${getExperimentId()}/${trialJobId}`);
        // Copy code files from local dir to NFS mounted dir
        await cpp.exec(`cp -r ${trialLocalTempFolder}/* ${this.trialLocalNFSTempFolder}/nni/${getExperimentId()}/${trialJobId}/.`);

        const nfsConfig: NFSConfig = this.kubeflowClusterConfig.nfs;
        const trialJobDetail: KubeflowTrialJobDetail = new KubeflowTrialJobDetail(
            trialJobId,
            'WAITING',
            Date.now(),
            trialWorkingFolder,
            form,
            kubeflowJobName,
            curTrialSequenceId,
            `nfs://${nfsConfig.server}:${path.join(nfsConfig.path, 'nni', getExperimentId(), trialJobId, 'output')}`,
            this.kubeflowJobPlural
            );

        // Create kubeflow training jobs
        await cpp.exec(`kubectl create -f ${kubeflowJobYamlPath}`);

        // Set trial job detail until kubectl create resource successfully 
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    public updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<TrialJobDetail> {
        throw new MethodNotImplementedError();
    }

    public listTrialJobs(): Promise<TrialJobDetail[]> {
        throw new MethodNotImplementedError();
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

    public async cancelTrialJob(trialJobId: string): Promise<void> {
        const trialJobDetail : KubeflowTrialJobDetail | undefined =  this.trialJobsMap.get(trialJobId);
        if(!trialJobDetail) {
            const errorMessage: string = `CancelTrialJob: trial job id ${trialJobId} not found`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }
        if(!this.kubeflowJobPlural) {
            const errorMessage: string = `CancelTrialJob: trial job id ${trialJobId} failed because kubeflowJobPlural is undefined`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }

        const result: cpp.childProcessPromise.Result = await cpp.exec(`kubectl delete ${this.kubeflowJobPlural} -l app=${this.NNI_KUBEFLOW_TRIAL_LABEL},expId=${getExperimentId()},trialId=${trialJobId}`);
        if(result.stderr) {
            const errorMessage: string = `kubectl delete ${this.kubeflowJobPlural} for trial ${trialJobId} failed: ${result.stderr}`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }

        trialJobDetail.endTime = Date.now();
        trialJobDetail.status = 'USER_CANCELED';        

        return Promise.resolve();
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.KUBEFLOW_CLUSTER_CONFIG:
                this.kubeflowClusterConfig = <KubeflowClusterConfig>JSON.parse(value);

                // If NFS config section is valid in config file, proceed to mount and config NFS
                if(this.kubeflowClusterConfig.nfs) {
                    //Check and mount NFS mount point here
                    await cpp.exec(`mkdir -p ${this.trialLocalNFSTempFolder}`);
                    const nfsServer: string = this.kubeflowClusterConfig.nfs.server;
                    const nfsPath: string = this.kubeflowClusterConfig.nfs.path;

                    try {
                        await cpp.exec(`sudo mount ${nfsServer}:${nfsPath} ${this.trialLocalNFSTempFolder}`);
                    } catch(error) {
                        const mountError: string = `Mount NFS ${nfsServer}:${nfsPath} to ${this.trialLocalNFSTempFolder} failed, error is ${error}`;
                        this.log.error(mountError);
                        throw new Error(mountError);
                    }
                }

                this.kubeflowJobPlural = kubeflowOperatorMap.get(this.kubeflowClusterConfig.operator);
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG:
                if (!this.kubeflowClusterConfig){
                    this.log.error('kubeflow cluster config is not initialized');
                    return Promise.reject(new Error('kubeflow cluster config is not initialized'));                    
                }

                this.kubeflowTrialConfig = <KubeflowTrialConfig>JSON.parse(value);
                assert(this.kubeflowClusterConfig !== undefined && this.kubeflowTrialConfig.worker !== undefined);
                break;
            default:
                break;
        }

        return Promise.resolve();
    }

    public getClusterMetadata(key: string): Promise<string> {
        throw new MethodNotImplementedError();
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

        assert(this.kubeflowJobPlural !== undefined);
        
        // Delete all kubeflow jobs whose expId label is current experiment id 
        try {
            await cpp.exec(`kubectl delete ${this.kubeflowJobPlural} -l app=${this.NNI_KUBEFLOW_TRIAL_LABEL},expId=${getExperimentId()}`);
        } catch(error) {
            this.log.error(`Delete ${this.kubeflowJobPlural} with label: app=${this.NNI_KUBEFLOW_TRIAL_LABEL},expId=${getExperimentId()} failed, error is ${error}`);
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
     * @param psPodResources ps pod template
     */
    private generateKubeflowJobConfig(trialJobId: string, trialWorkingFolder: string, kubeflowJobName : string, workerPodResources : any, psPodResources?: any) : any {
        if(!this.kubeflowClusterConfig) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }

        if(!this.kubeflowTrialConfig) {
            throw new Error('Kubeflow trial config is not initialized');
        }

        const tfReplicaSpecsObj: any = {};
        tfReplicaSpecsObj.Worker = this.generateReplicaConfig(trialWorkingFolder, this.kubeflowTrialConfig.worker.replicas, 
            this.kubeflowTrialConfig.worker.image, 'run_worker.sh', workerPodResources);

        if(this.kubeflowTrialConfig.ps) {
            tfReplicaSpecsObj.Ps = this.generateReplicaConfig(trialWorkingFolder, this.kubeflowTrialConfig.ps.replicas, 
                this.kubeflowTrialConfig.ps.image, 'run_ps.sh', psPodResources);
        }

        return {
            apiVersion: 'kubeflow.org/v1alpha2',
            kind: 'TFJob',
            metadata: { 
                name: kubeflowJobName,
                namespace: 'default',
                labels: {
                    app: this.NNI_KUBEFLOW_TRIAL_LABEL,
                    expId: getExperimentId(),
                    trialId: trialJobId
                }
            },
            spec: {
                tfReplicaSpecs: tfReplicaSpecsObj
            }                
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
                        name: 'tensorflow',
                        image: replicaImage,
                        args: ["sh", `${path.join(trialWorkingFolder, runScriptFile)}`],
                        volumeMounts: [{
                            name: 'nni-nfs-vol',
                            mountPath: this.CONTAINER_MOUNT_PATH
                        }],
                        resources: podResources
                    }],
                    restartPolicy: 'ExitCode',
                    volumes: [{
                        name: 'nni-nfs-vol',
                        nfs: {
                            server: `${this.kubeflowClusterConfig.nfs.server}`,
                            path: `${this.kubeflowClusterConfig.nfs.path}`
                        }
                    }]
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
    private genereateRunScript(trialJobId: string, trialWorkingFolder: string, 
                command: string, trialSequenceId: string, roleType: DistTrainRole): string {
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
        if(this.kubeflowTrialConfig) {
            switch(roleType) {
                case 'ps':
                    if(this.kubeflowTrialConfig.ps && this.kubeflowTrialConfig.ps.gpuNum == 0) {
                        runScriptLines.push(`export CUDA_VISIBLE_DEVICES=''`);
                    }
                    break;
                case 'worker':
                    if(this.kubeflowTrialConfig.worker && this.kubeflowTrialConfig.worker.gpuNum == 0) {
                        runScriptLines.push(`export CUDA_VISIBLE_DEVICES=''`);
                    }
                    break;
                default:
                    break;
            }
        }

        runScriptLines.push('mkdir -p $NNI_SYS_DIR');
        runScriptLines.push('mkdir -p $NNI_OUTPUT_DIR');
        runScriptLines.push('cp -rT $NNI_CODE_DIR $NNI_SYS_DIR');
        runScriptLines.push('cd $NNI_SYS_DIR');
        runScriptLines.push('sh install_nni.sh # Check and install NNI pkg');
        runScriptLines.push(`python3 -m nni_trial_tool.trial_keeper --trial_command '${command}' --nnimanager_ip '${getIPV4Address()}' --nnimanager_port '${this.kubeflowRestServerPort}' 1>$NNI_OUTPUT_DIR/trialkeeper_stdout 2>$NNI_OUTPUT_DIR/trialkeeper_stderr`);

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