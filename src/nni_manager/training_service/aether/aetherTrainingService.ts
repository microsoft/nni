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
import * as fs from 'fs';
import * as path from 'path';
import * as request from 'request';

import { EventEmitter } from 'events';
import {
    HostJobApplicationForm, JobApplicationForm, HyperParameters, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric, TrialJobStatus
} from '../../common/trainingService';

class AetherTrialJobDetail implements TrialJobDetail {
    readonly id:string;
    readonly status: TrialJobStatus;
    readonly submitTime: number;
    readonly url: string;
    readonly guid: string;  // GUID of Aether Experiment
    readonly workingDirectory: string;
    readonly form: JobApplicationForm;
    readonly sequenceId: number;
}

/**
 * Aether Training Service
 */

@component.Singleton
class aetherTrainingService extends TrainingService {
    private readonly AetherClientPath: string;
    private readonly metricsEmitter: EventEmitter;
    private readonly trialJobsMap: Map<string, AetherTrialJobDetail>;

    public async listTrialJobs(): Promise<AetherTrialJobDetail[]> {
    }

    public async getTrialJob(trialJobId: string): Promise<AetherTrialJobDetail> {

    }

    public async submitTrialJob(form: JobApplicationForm): Promise<AetherTrialJobDetail> {

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
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {

    }

    public async getClusterMetadata(key: string): Promise<string> {

    }

    public async cleanUp(): Promise<void> {

    }

    public async run(): Promise<void> {

    }

}