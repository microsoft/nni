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
    TrialJobDetail, TrialJobMetric, NNIManagerIpConfig
} from '../../common/trainingService';


var azure = require('azure-storage');

type DistTrainRole = 'worker' | 'ps' | 'master';

/**
 * Training Service implementation for Kubeflow
 * Refer https://github.com/kubeflow/kubeflow for more info about Kubeflow
 */
@component.Singleton
class FrameworkControllerTrainingService implements TrainingService {
    
    
    constructor() {        

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

export { FrameworkControllerTrainingService }
