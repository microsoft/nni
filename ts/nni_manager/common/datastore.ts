// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { ExperimentProfile, TrialJobStatistics } from './manager';
import { TrialJobDetail, TrialJobStatus } from './trainingService';

type TrialJobEvent = TrialJobStatus | 'USER_TO_CANCEL' | 'ADD_CUSTOMIZED' | 'ADD_HYPERPARAMETER' | 'IMPORT_DATA' |'ADD_RESUMED';
type MetricType = 'PERIODICAL' | 'FINAL' | 'CUSTOM' | 'REQUEST_PARAMETER';

interface ExperimentProfileRecord {
    readonly timestamp: number;
    readonly experimentId: number;
    readonly revision: number;
    readonly data: ExperimentProfile;
}

interface TrialJobEventRecord {
    readonly timestamp: number;
    readonly trialJobId: string;
    readonly event: TrialJobEvent;
    readonly data?: string;
    readonly logPath?: string;
    readonly sequenceId?: number;
    readonly message?: string;
    readonly envId?: string;
}

interface MetricData {
    readonly parameter_id: string;
    readonly trial_job_id: string;
    readonly type: MetricType;
    readonly sequence: number;
    readonly value: any;
}

interface MetricDataRecord {
    readonly timestamp: number;
    readonly trialJobId: string;
    readonly parameterId: string;
    readonly type: MetricType;
    readonly sequence: number;
    readonly data: any;
}

interface TrialJobInfo {
    trialJobId: string;
    sequenceId?: number;
    status: TrialJobStatus;
    message?: string;
    startTime?: number;
    endTime?: number;
    hyperParameters?: string[];
    logPath?: string;
    finalMetricData?: MetricDataRecord[];
    stderrPath?: string;
    envId?: string;
}

interface HyperParameterFormat {
    parameter_source: string;
    parameters: Record<string, any>;
    parameter_id: number;
}

interface ExportedDataFormat {
    parameter: Record<string, any>;
    value: Record<string, any>;
    trialJobId: string;
}

abstract class DataStore {
    public abstract init(): Promise<void>;
    public abstract close(): Promise<void>;
    public abstract storeExperimentProfile(experimentProfile: ExperimentProfile): Promise<void>;
    public abstract getExperimentProfile(experimentId: string): Promise<ExperimentProfile>;
    public abstract storeTrialJobEvent(
        event: TrialJobEvent, trialJobId: string, hyperParameter?: string, jobDetail?: TrialJobDetail): Promise<void>;
    public abstract getTrialJobStatistics(): Promise<TrialJobStatistics[]>;
    public abstract listTrialJobs(status?: TrialJobStatus): Promise<TrialJobInfo[]>;
    public abstract getTrialJob(trialJobId: string): Promise<TrialJobInfo>;
    public abstract storeMetricData(trialJobId: string, data: string): Promise<void>;
    public abstract getMetricData(trialJobId?: string, metricType?: MetricType): Promise<MetricDataRecord[]>;
    public abstract exportTrialHpConfigs(): Promise<string>;
    public abstract getImportedData(): Promise<string[]>;
}

abstract class Database {
    public abstract init(createNew: boolean, dbDir: string): Promise<void>;
    public abstract close(): Promise<void>;
    public abstract storeExperimentProfile(experimentProfile: ExperimentProfile): Promise<void>;
    public abstract queryExperimentProfile(experimentId: string, revision?: number): Promise<ExperimentProfile[]>;
    public abstract queryLatestExperimentProfile(experimentId: string): Promise<ExperimentProfile>;
    public abstract storeTrialJobEvent(
        event: TrialJobEvent, trialJobId: string, timestamp: number, hyperParameter?: string, jobDetail?: TrialJobDetail): Promise<void>;
    public abstract queryTrialJobEvent(trialJobId?: string, event?: TrialJobEvent): Promise<TrialJobEventRecord[]>;
    public abstract storeMetricData(trialJobId: string, data: string): Promise<void>;
    public abstract queryMetricData(trialJobId?: string, type?: MetricType): Promise<MetricDataRecord[]>;
}

export {
    DataStore, Database, TrialJobEvent, MetricType, MetricData, TrialJobInfo,
    ExperimentProfileRecord, TrialJobEventRecord, MetricDataRecord, HyperParameterFormat, ExportedDataFormat
};
