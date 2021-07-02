// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

/**
 * define TrialJobStatus
 */
type TrialJobStatus = 'UNKNOWN' | 'WAITING' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'USER_CANCELED' | 'SYS_CANCELED' | 'EARLY_STOPPED';

type LogType = 'TRIAL_LOG' | 'TRIAL_STDOUT' | 'TRIAL_ERROR';

interface TrainingServiceMetadata {
    readonly key: string;
    readonly value: string;
}

interface HyperParameters {
    readonly value: string;
    readonly index: number;
}

type PlacementConstraintType = 'None' | 'Topology' | 'Device'
interface PlacementConstraint{
    readonly type: PlacementConstraintType;
    readonly gpus: Array<number|[number,number]>;
    /**
     * Topology constraint is in form of Array<number>, e.g., [2,2,2] means find three servers each with two GPUs
     * Device constraint is in form of Array<[number,number]>, e.g., [(0,1),(1,0)] means it must be placed on 
     *      Node-0's GPU-1 and Node-1's GPU-0
     */
}
/**
 * define TrialJobApplicationForm
 */
interface TrialJobApplicationForm {
    readonly sequenceId: number;
    readonly hyperParameters: HyperParameters;
    readonly placementConstraint?: PlacementConstraint;
}

interface TrialCommandContent {
    readonly parameter_id: string;
    readonly parameters: string;
    readonly parameter_source: string;
    readonly placement_constraint: PlacementConstraint
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

interface GPUStatus {
    readonly nodeId: number;
    readonly gpuId: number;
    readonly status: string;
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
    public abstract addGPUStatusUpdateListener(listener: (status: Array<GPUStatus>) => void): void;
    public abstract removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void;
    public abstract removeGPUStatusUpdateListener(listener: (status: Array<GPUStatus>) => void): void;
    public abstract submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail>;
    public abstract updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail>;
    public abstract cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean): Promise<void>;
    public abstract getTrialLog(trialJobId: string, logType: LogType): Promise<string>;
    public abstract setClusterMetadata(key: string, value: string): Promise<void>;
    public abstract getClusterMetadata(key: string): Promise<string>;
    public abstract getTrialOutputLocalPath(trialJobId: string): Promise<string>;
    public abstract fetchTrialOutput(trialJobId: string, subpath: string): Promise<void>;
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
    NNIManagerIpConfig, LogType, GPUStatus, PlacementConstraint, TrialCommandContent
};
