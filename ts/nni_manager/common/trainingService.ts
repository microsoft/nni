// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

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
    public abstract get isMultiPhaseJobSupported(): boolean;
    public abstract cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean): Promise<void>;
    public abstract getTrialLog(trialJobId: string, logType: LogType): Promise<string>;
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
    NNIManagerIpConfig, LogType
};
