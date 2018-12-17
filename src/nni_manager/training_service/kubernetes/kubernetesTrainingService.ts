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
import { delay, generateParamFileName, getExperimentRootDir, getIPV4Address, uniqueString, getJobCancelStatus } from '../../common/utils';
import {
    JobApplicationForm, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric, NNIManagerIpConfig
} from '../../common/trainingService';
import { KubernetesTrialJobDetail } from './kubernetesData';
import { KubernetesClusterConfig, KubernetesTrialConfig, KubernetesStorageKind, keyVaultConfig, AzureStorage} from './kubernetesConfig';
import { GeneralK8sClient }from './kubernetesApiClient';

import * as azureStorage from 'azure-storage';
var azure = require('azure-storage');

type DistTrainRole = 'worker' | 'ps' | 'master';

/**
 * Training Service implementation for Kubeflow
 * Refer https://github.com/kubeflow/kubeflow for more info about Kubeflow
 */
@component.Singleton
class KubernetesTrainingService implements TrainingService {
    protected readonly NNI_KUBEFLOW_TRIAL_LABEL: string = 'nni-kubeflow-trial';
    protected readonly log!: Logger;
    protected readonly metricsEmitter: EventEmitter;
    protected readonly trialJobsMap: Map<string, KubernetesTrialJobDetail>;
    /**  experiment root dir in NFS */
    protected readonly trialLocalNFSTempFolder: string;
    protected stopping: boolean = false;
    protected experimentId! : string;
    protected nextTrialSequenceId: number;
    protected kubernetesClusterConfig?: KubernetesClusterConfig;
    protected kubernetesRestServerPort?: number;
    protected readonly CONTAINER_MOUNT_PATH: string;
    protected azureStorageClient?: azureStorage.FileService;
    protected azureStorageShare?: string;
    protected azureStorageSecretName?: string;
    protected azureStorageAccountName?: string;
    protected nniManagerIpConfig?: NNIManagerIpConfig;
    protected readonly genericK8sClient: GeneralK8sClient;
    protected kubernetesTrialConfig?: KubernetesTrialConfig;

    
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
        return Promise.resolve();
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        let tmp:any;
        return Promise.resolve(tmp);
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        return Promise.resolve();
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        return Promise.resolve();
    }

    public async cleanUp(): Promise<void> {
        return Promise.resolve();
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
}

export { KubernetesTrainingService }
