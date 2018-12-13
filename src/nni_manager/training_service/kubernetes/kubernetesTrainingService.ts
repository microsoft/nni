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
import * as component from 'common/component';
import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as path from 'path';

import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { EventEmitter } from 'events';
import { getExperimentId, getInitTrialSequenceId } from 'common/experimentStartupInfo';
import { getLogger, Logger } from 'common/log';
import { MethodNotImplementedError } from 'common/errors';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { delay, generateParamFileName, getExperimentRootDir, getIPV4Address, uniqueString, getJobCancelStatus } from 'common/utils';
import {
    JobApplicationForm, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric, NNIManagerIpConfig
} from 'common/trainingService';
import { KubernetesTrialJobDetail } from './kubernetesData';
import { KubernetesClusterConfig, KubernetesStorageKind, keyVaultConfig, AzureStorage} from './kubernetesConfig';

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

    
    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, KubernetesTrialJobDetail>();
        this.trialLocalNFSTempFolder = path.join(getExperimentRootDir(), 'trials-nfs-tmp');
        this.experimentId = getExperimentId();      
        this.nextTrialSequenceId = -1;
        this.CONTAINER_MOUNT_PATH = '/tmp/mount';
    }

    public async run(): Promise<void> {
        
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        let tmp:any;
        return Promise.resolve(tmp);
    }

    public updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<TrialJobDetail> {
        throw new MethodNotImplementedError();
    }

    public listTrialJobs(): Promise<TrialJobDetail[]> {
        let tmp: any;
        return Promise.resolve(tmp);
    }

    public getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        let tmp: any;
        return Promise.resolve(tmp);
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void) {
        
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void) {
       
    }
 
    public get isMultiPhaseJobSupported(): boolean {
        return false;
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        return Promise.resolve();
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        return Promise.resolve();
    }

    public getClusterMetadata(key: string): Promise<string> {
        return Promise.resolve('');
    }

    public async cleanUp(): Promise<void> {
        return Promise.resolve();
    }

    public get MetricsEmitter() : EventEmitter {
        let tmp: any;
        return tmp;
    }
}

export { KubernetesTrainingService }
