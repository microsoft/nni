// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { ExperimentConfig } from './manager';

/**
 * define TrialJobStatus
 */
type TrialJobStatus = 'UNKNOWN' | 'WAITING' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'USER_CANCELED' | 'SYS_CANCELED' | 'EARLY_STOPPED';

type LogType = 'TRIAL_LOG' | 'TRIAL_ERROR';

interface TrainingServiceMetadata {
    readonly key: string;
    readonly value: string;
}

interface HyperParameters {
    readonly value: string;
    readonly index: number;
}

/**
 * define TrialJobApplicationForm
 */
interface TrialJobApplicationForm {
    readonly sequenceId: number;
    readonly hyperParameters: HyperParameters;
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
    readonly form: TrialJobApplicationForm;
    isEarlyStopped?: boolean;
    message?: string;
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
    public abstract submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail>;
    public abstract updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail>;
    public get isMultiPhaseJobSupported(): boolean { return false; }
    public abstract cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean): Promise<void>;
    public abstract getTrialLog(trialJobId: string, logType: LogType): Promise<string>;
    public async setClusterMetadata(_key: string, _value: string): Promise<void> { throw new Error(); }
    public async getClusterMetadata(_key: string): Promise<string> { throw new Error(); }
    public abstract cleanUp(): Promise<void>;
    public abstract run(): Promise<void>;
    public async initConfig(_config: ExperimentConfig): Promise<void> { return; }
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
    NNIManagerIpConfig, LogType
};
