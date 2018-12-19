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

import * as component from '../../common/component';
import * as cpp from 'child-process-promise';
import * as path from 'path';

import { EventEmitter } from 'events';
import { getExperimentId, getInitTrialSequenceId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import { MethodNotImplementedError } from '../../common/errors';
import { getExperimentRootDir, getIPV4Address, uniqueString, getJobCancelStatus } from '../../common/utils';
import {
    JobApplicationForm, TrainingService,
    TrialJobDetail, TrialJobMetric, NNIManagerIpConfig
} from '../../common/trainingService';
import { KubernetesTrialJobDetail } from './kubernetesData';
import { KubernetesClusterConfig } from './kubernetesConfig';
import { GeneralK8sClient, KubernetesCRDClient } from './kubernetesApiClient';
import { AzureStorageClientUtility } from './azureStorageClientUtils';
import { KubernetesJobRestServer } from './kubernetesJobRestServer';

import * as azureStorage from 'azure-storage';
var azure = require('azure-storage');
var base64 = require('js-base64').Base64;

type DistTrainRole = 'worker' | 'ps' | 'master';

/**
 * Training Service implementation for Kubeflow
 * Refer https://github.com/kubeflow/kubeflow for more info about Kubeflow
 */
@component.Singleton
class KubernetesTrainingService implements TrainingService {
    protected readonly NNI_KUBERNETES_TRIAL_LABEL: string = 'nni-kubernetes-trial';
    protected readonly log!: Logger;
    protected readonly metricsEmitter: EventEmitter;
    protected readonly trialJobsMap: Map<string, KubernetesTrialJobDetail>;
    /**  experiment root dir in NFS */
    protected readonly trialLocalNFSTempFolder: string;
    protected stopping: boolean = false;
    protected experimentId! : string;
    protected nextTrialSequenceId: number;
    protected kubernetesRestServerPort?: number;
    protected readonly CONTAINER_MOUNT_PATH: string;
    protected azureStorageClient?: azureStorage.FileService;
    protected azureStorageShare?: string;
    protected azureStorageSecretName?: string;
    protected azureStorageAccountName?: string;
    protected nniManagerIpConfig?: NNIManagerIpConfig;
    protected readonly genericK8sClient: GeneralK8sClient;
    protected kubernetesCRDClient?: KubernetesCRDClient;
    protected kubernetesJobRestServer?: KubernetesJobRestServer;
    protected kubernetesClusterConfig?: KubernetesClusterConfig;
    
    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, KubernetesTrialJobDetail>();
        this.trialLocalNFSTempFolder = path.join(getExperimentRootDir(), 'trials-nfs-tmp');
        this.experimentId = getExperimentId();      
        this.nextTrialSequenceId = -1;
        this.CONTAINER_MOUNT_PATH = '/tmp/mount';
        this.genericK8sClient = new GeneralK8sClient();
    }

    public async run(): Promise<void> {
        throw new MethodNotImplementedError();
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        throw new MethodNotImplementedError();
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        throw new MethodNotImplementedError();
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
        
        this.trialJobsMap.forEach(async (value: KubernetesTrialJobDetail, key: string) => {
            if (value.form.jobType === 'TRIAL') {
                jobs.push(await this.getTrialJob(key));
            }
        });

        return Promise.resolve(jobs);
    }

    public getTrialJob(trialJobId: string): Promise<TrialJobDetail> {

        const kubernetesTrialJob: TrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (!kubernetesTrialJob) {
            return Promise.reject(`trial job ${trialJobId} not found`)
        }        

        return Promise.resolve(kubernetesTrialJob);
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

    public getClusterMetadata(key: string): Promise<string> {
        return Promise.resolve('');
    }

    public get MetricsEmitter() : EventEmitter {
        return this.metricsEmitter;
    }

    /**
     * Genereate run script for different roles(like worker or ps)
     * @param trialJobId trial job id
     * @param trialWorkingFolder working folder
     * @param command 
     * @param trialSequenceId sequence id
     */
    protected generateRunScript(trialJobId: string, trialWorkingFolder: string, 
                command: string, trialSequenceId: string, roleName: string, gpuNum: number): string {
        const runScriptLines: string[] = [];

        runScriptLines.push('#!/bin/bash');
        runScriptLines.push('export NNI_PLATFORM=kubeflow');
        runScriptLines.push(`export NNI_SYS_DIR=$PWD/nni/${trialJobId}`);
        runScriptLines.push(`export NNI_OUTPUT_DIR=${path.join(trialWorkingFolder, 'output', `${roleName}_output`)}`);
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
        + `--nnimanager_ip '${nniManagerIp}' --nnimanager_port '${this.kubernetesRestServerPort}' `
        + `1>$NNI_OUTPUT_DIR/trialkeeper_stdout 2>$NNI_OUTPUT_DIR/trialkeeper_stderr`);

        return runScriptLines.join('\n');
    }

    protected generateSequenceId(): number {
        if (this.nextTrialSequenceId === -1) {
            this.nextTrialSequenceId = getInitTrialSequenceId();
        }

        return this.nextTrialSequenceId++;
    }

    protected async createAzureStorage(vaultName: string, valutKeyName: string, accountName: string, azureShare: string): Promise<void> {
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
                            app: this.NNI_KUBERNETES_TRIAL_LABEL,
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
            return Promise.reject(error);
        }
        return Promise.resolve();
    }

    protected async createNFSStorage(nfsServer: string, nfsPath: string): Promise<void> {
        await cpp.exec(`mkdir -p ${this.trialLocalNFSTempFolder}`);
        try {
            await cpp.exec(`sudo mount ${nfsServer}:${nfsPath} ${this.trialLocalNFSTempFolder}`);
        } catch(error) {
            const mountError: string = `Mount NFS ${nfsServer}:${nfsPath} to ${this.trialLocalNFSTempFolder} failed, error is ${error}`;
            this.log.error(mountError);
            return Promise.reject(mountError);
        }
        return Promise.resolve();
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trialJobDetail : KubernetesTrialJobDetail | undefined =  this.trialJobsMap.get(trialJobId);
        if(!trialJobDetail) {
            const errorMessage: string = `CancelTrialJob: trial job id ${trialJobId} not found`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }        
        if(!this.kubernetesCRDClient) {
            const errorMessage: string = `CancelTrialJob: trial job id ${trialJobId} failed because operatorClient is undefined`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }

        try {
            await this.kubernetesCRDClient.deleteKubernetesJob(new Map(
                [
                    ['app', this.NNI_KUBERNETES_TRIAL_LABEL],
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
            if(this.kubernetesCRDClient) {
                await this.kubernetesCRDClient.deleteKubernetesJob(new Map(
                    [
                        ['app', this.NNI_KUBERNETES_TRIAL_LABEL],
                        ['expId', getExperimentId()]
                    ]
                ));
            }
        } catch(error) {
            this.log.error(`Delete kubeflow job with label: app=${this.NNI_KUBERNETES_TRIAL_LABEL},expId=${getExperimentId()} failed, error is ${error}`);
        }

        // Unmount NFS
        try {
            await cpp.exec(`sudo umount ${this.trialLocalNFSTempFolder}`);
        } catch(error) {
            this.log.error(`Unmount ${this.trialLocalNFSTempFolder} failed, error is ${error}`);
        }

        // Stop Kubeflow rest server 
        if(!this.kubernetesJobRestServer) {
            throw new Error('kubernetesJobRestServer not initialized!');
        }
        try {
            await this.kubernetesJobRestServer.stop();
            this.log.info('Kubeflow Training service rest server stopped successfully.');
        } catch (error) {
            this.log.error(`Kubeflow Training service rest server stopped failed, error: ${error.message}`);
            Promise.reject(error);
        }

        return Promise.resolve();
    }
}

export { KubernetesTrainingService }
