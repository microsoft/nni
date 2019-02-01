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

'use strict';

/**
 * define TrialJobStatus
 */
type TrialJobStatus = 'UNKNOWN' | 'WAITING' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'USER_CANCELED' | 'SYS_CANCELED' | 'EARLY_STOPPED';
type JobType = 'TRIAL' | 'HOST';

interface TrainingServiceMetadata {
    readonly key: string;
    readonly value: string;
}

/**
 * define JobApplicationForm
 */
interface JobApplicationForm {
    readonly jobType: JobType;
}

interface HyperParameters {
    readonly value: string;
    readonly index: number;
}

/**
 * define TrialJobApplicationForm
 */
interface TrialJobApplicationForm extends JobApplicationForm {
    readonly hyperParameters: HyperParameters;
}

/**
 * define HostJobApplicationForm
 */
interface HostJobApplicationForm extends JobApplicationForm {
    readonly host: string;
    readonly cmd: string;
}

/**
 * define TrialJobDetail
 */
interface TrialJobDetail {
    readonly id: string;
    readonly status: TrialJobStatus;
    readonly submitTime: number;
    readonly startTime?: number;
    readonly endTime?: number;
    readonly tags?: string[];
    readonly url?: string;
    readonly workingDirectory: string;
    readonly form: JobApplicationForm;
    readonly sequenceId: number;
    isEarlyStopped?: boolean;
}

interface HostJobDetail {
    readonly id: string;
    readonly status: string;
}

/**
 * define TrialJobMetric
 */
interface TrialJobMetric {
    readonly id: string;
    readonly data: string;
}

/**
 * define TrainingServiceError
 */
class TrainingServiceError extends Error {
    private errCode: number;

    constructor(errorCode: number, errorMessage: string) {
        super(errorMessage);
        this.errCode = errorCode;
    }

    get errorCode(): number {
        return this.errCode;
    }
}

/**
 * define TrainingService
 */
abstract class TrainingService {
    public abstract listTrialJobs(): Promise<TrialJobDetail[]>;
    public abstract getTrialJob(trialJobId: string): Promise<TrialJobDetail>;
    public abstract addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void;
    public abstract removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void;
    public abstract submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail>;
    public abstract updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<TrialJobDetail>;
    public abstract get isMultiPhaseJobSupported(): boolean;
    public abstract cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean): Promise<void>;
    public abstract setClusterMetadata(key: string, value: string): Promise<void>;
    public abstract getClusterMetadata(key: string): Promise<string>;
    public abstract cleanUp(): Promise<void>;
    public abstract run(): Promise<void>;
}

/**
 * the ip of nni manager
 */
class NNIManagerIpConfig {
    public readonly nniManagerIp: string;
    constructor(nniManagerIp: string){
        this.nniManagerIp = nniManagerIp;
    }
}

export {
    TrainingService, TrainingServiceError, TrialJobStatus, TrialJobApplicationForm,
    TrainingServiceMetadata, TrialJobDetail, TrialJobMetric, HyperParameters,
    HostJobApplicationForm, JobApplicationForm, JobType, NNIManagerIpConfig
};


