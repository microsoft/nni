// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { MetricDataRecord, MetricType, TrialJobInfo } from './datastore';
import { TrialJobStatus, LogType } from './trainingService';

type ProfileUpdateType = 'TRIAL_CONCURRENCY' | 'MAX_EXEC_DURATION' | 'SEARCH_SPACE' | 'MAX_TRIAL_NUM';
type ExperimentStatus = 'INITIALIZED' | 'RUNNING' | 'ERROR' | 'STOPPING' | 'STOPPED' | 'DONE' | 'NO_MORE_TRIAL' | 'TUNER_NO_MORE_TRIAL';
namespace ExperimentStartUpMode {
    export const NEW = 'new';
    export const RESUME = 'resume';
}

interface ExperimentParams {
    authorName: string;
    experimentName: string;
    description?: string;
    trialConcurrency: number;
    maxExecDuration: number; //seconds
    maxTrialNum: number;
    searchSpace: string;
    trainingServicePlatform: string;
    multiPhase?: boolean;
    multiThread?: boolean;
    versionCheck?: boolean;
    logCollection?: string;
    tuner?: {
        className?: string;
        builtinTunerName?: string;
        codeDir?: string;
        classArgs?: any;
        classFileName?: string;
        checkpointDir: string;
        includeIntermediateResults?: boolean;
        gpuIndices?: string;
    };
    assessor?: {
        className?: string;
        builtinAssessorName?: string;
        codeDir?: string;
        classArgs?: any;
        classFileName?: string;
        checkpointDir: string;
    };
    advisor?: {
        className?: string;
        builtinAdvisorName?: string;
        codeDir?: string;
        classArgs?: any;
        classFileName?: string;
        checkpointDir: string;
        gpuIndices?: string;
    };
    clusterMetaData?: {
        key: string;
        value: string;
    }[];
}

class AlgorithmConfig {
    name?: string;
    className?: string;
    codeDirectory?: string;
    classArgs?: object;

    get includeIntermediateResults(): boolean { return false; }
}

class ExperimentConfig {
    experimentName?: string;
    searchSpace: any;
    trialCommand: string;
    trialCodeDirectory: string;
    trialConcurrency: number;
    trialGpuNumber?: number;
    maxExperimentDuration?: string;
    maxTrialNumber?: number;
    nniManagerIp?: string;
    useAnnotation?: boolean;
    debug?: boolean;
    logLevel?: string;
    experimentWorkingDirectory?: string;
    tunerGpuIndices?: number[];
    tuner?: AlgorithmConfig;
    assessor?: AlgorithmConfig;
    advisor?: AlgorithmConfig;
    trainingService: any;

    get multiPhase(): boolean { return false; }
    get versionCheck(): boolean { return false; }
    get logCollection(): boolean { return false; }

    constructor() { throw new Error('trying to construct ExperimentConfig'); }
}

interface ExperimentProfile {
    params: ExperimentParams | ExperimentConfig;
    id: string;
    execDuration: number;
    logDir?: string;
    startTime?: number;
    endTime?: number;
    nextSequenceId: number;
    revision: number;
}

interface TrialJobStatistics {
    trialJobStatus: TrialJobStatus;
    trialJobNumber: number;
}

interface NNIManagerStatus {
    status: ExperimentStatus;
    errors: string[];
}

abstract class Manager {
    public abstract startExperiment(experimentParams: ExperimentParams): Promise<string>;
    public abstract resumeExperiment(readonly: boolean): Promise<void>;
    public abstract stopExperiment(): Promise<void>;
    public abstract getExperimentProfile(): Promise<ExperimentProfile>;
    public abstract updateExperimentProfile(experimentProfile: ExperimentProfile, updateType: ProfileUpdateType): Promise<void>;
    public abstract importData(data: string): Promise<void>;
    public abstract getImportedData(): Promise<string[]>;
    public abstract exportData(): Promise<string>;

    public abstract addCustomizedTrialJob(hyperParams: string): Promise<number>;
    public abstract cancelTrialJobByUser(trialJobId: string): Promise<void>;

    public abstract listTrialJobs(status?: TrialJobStatus): Promise<TrialJobInfo[]>;
    public abstract getTrialJob(trialJobId: string): Promise<TrialJobInfo>;
    public abstract setClusterMetadata(key: string, value: string): Promise<void>;
    public abstract getClusterMetadata(key: string): Promise<string>;

    public abstract getMetricData(trialJobId?: string, metricType?: MetricType): Promise<MetricDataRecord[]>;
    public abstract getMetricDataByRange(minSeqId: number, maxSeqId: number): Promise<MetricDataRecord[]>;
    public abstract getLatestMetricData(): Promise<MetricDataRecord[]>;

    public abstract getTrialLog(trialJobId: string, logType: LogType): Promise<string>;

    public abstract getTrialJobStatistics(): Promise<TrialJobStatistics[]>;
    public abstract getTrialJobMessage(trialJobId: string): string | undefined;
    public abstract getStatus(): NNIManagerStatus;
}

export { Manager, ExperimentConfig, ExperimentParams, ExperimentProfile, TrialJobStatistics, ProfileUpdateType, NNIManagerStatus, ExperimentStatus, ExperimentStartUpMode };
