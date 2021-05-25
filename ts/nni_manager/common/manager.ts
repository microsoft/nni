// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { MetricDataRecord, MetricType, TrialJobInfo } from './datastore';
import { TrialJobStatus, LogType } from './trainingService';
import { ExperimentConfig } from './experimentConfig';

type ProfileUpdateType = 'TRIAL_CONCURRENCY' | 'MAX_EXEC_DURATION' | 'SEARCH_SPACE' | 'MAX_TRIAL_NUM';
type ExperimentStatus = 'INITIALIZED' | 'RUNNING' | 'ERROR' | 'STOPPING' | 'STOPPED' | 'DONE' | 'NO_MORE_TRIAL' | 'TUNER_NO_MORE_TRIAL' | 'VIEWED';
namespace ExperimentStartUpMode {
    export const NEW = 'new';
    export const RESUME = 'resume';
}

interface ExperimentProfile {
    params: ExperimentConfig;
    id: string;
    execDuration: number;
    logDir: string;
    startTime: number;
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
    public abstract startExperiment(experimentConfig: ExperimentConfig): Promise<string>;
    public abstract resumeExperiment(readonly: boolean): Promise<void>;
    public abstract stopExperiment(): Promise<void>;
    public abstract stopExperimentTopHalf(): Promise<void>;
    public abstract stopExperimentBottomHalf(): Promise<void>;
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
    public abstract getStatus(): NNIManagerStatus;

    public abstract getTrialOutputLocalPath(trialJobId: string): Promise<string>;
    public abstract fetchTrialOutput(trialJobId: string, subpath: string): Promise<void>;
}

export { Manager, ExperimentConfig, ExperimentProfile, TrialJobStatistics, ProfileUpdateType, NNIManagerStatus, ExperimentStatus, ExperimentStartUpMode };
