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

import { ExperimentProfile, TrialJobStatistics } from './manager';
import { TrialJobDetail, TrialJobStatus } from './trainingService';

type TrialJobEvent = TrialJobStatus | 'USER_TO_CANCEL' | 'ADD_CUSTOMIZED' | 'ADD_HYPERPARAMETER' | 'IMPORT_DATA';
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
    id: string;
    sequenceId?: number;
    status: TrialJobStatus;
    startTime?: number;
    endTime?: number;
    hyperParameters?: string[];
    logPath?: string;
    finalMetricData?: MetricDataRecord[];
    stderrPath?: string;
}

interface HyperParameterFormat {
    parameter_source: string;
    parameters: Object;
    parameter_id: number;
}

interface ExportedDataFormat {
    parameter: Object;
    value: Object;
    id: string;
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
