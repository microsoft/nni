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
import { Logger, getLogger } from 'common/log';
import { stringify } from 'querystring';
import { Deferred } from 'ts-deferred';
import { MethodNotImplementedError } from 'common/errors';

class AetherTrialJobDetail implements TrialJobDetail {
    readonly id:string;
    readonly status: TrialJobStatus;
    readonly submitTime: number;
    readonly url: string;
    readonly guid: string;  // GUID of Aether Experiment
    readonly workingDirectory: string;
    readonly form: JobApplicationForm;
    readonly sequenceId: number;

    constructor(id: string, status: TrialJobStatus, submitTime: number, guid: string, 
        workingDirectory: string, form: JobApplicationForm, sequenceId: number) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.url = `aether://experiments/${guid}`;
        this.guid = guid;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.sequenceId = sequenceId;
    }
}

/**
 * Aether Training Service
 */

@component.Singleton
class AetherTrainingService implements TrainingService {
    private readonly log!: Logger;
    private readonly AETHER_EXE_PATH: string = '/fake/aether/exe/path';
    private readonly metricsEmitter: EventEmitter;
    private readonly trialJobsMap: Map<string, AetherTrialJobDetail>;

    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, AetherTrialJobDetail>();
    }
    public async listTrialJobs(): Promise<AetherTrialJobDetail[]> {
        const deferred: Deferred<AetherTrialJobDetail[]> = new Deferred<AetherTrialJobDetail[]>();

        return deferred.promise;
    }

    public async getTrialJob(trialJobId: string): Promise<AetherTrialJobDetail> {
        const deferred: Deferred<AetherTrialJobDetail> = new Deferred<AetherTrialJobDetail>();

        return deferred.promise;
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<AetherTrialJobDetail> {
        const deferred: Deferred<AetherTrialJobDetail> = new Deferred<AetherTrialJobDetail>();

        return deferred.promise;
    }

    public updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<AetherTrialJobDetail> {
        throw new MethodNotImplementedError();
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
        const deferred: Deferred<string> = new Deferred<string>();

        return deferred.promise;
    }

    public async cleanUp(): Promise<void> {
        
    }

    public async run(): Promise<void> {

    }

    public get MetricsEmitter() : EventEmitter {
        return this.metricsEmitter;
    }
}

export { AetherTrainingService }