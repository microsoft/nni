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

import { MetricDataRecord, MetricType, TrialJobInfo } from './datastore';
import { TrialJobStatus } from './trainingService';

type ProfileUpdateType = 'TRIAL_CONCURRENCY' | 'MAX_EXEC_DURATION' | 'SEARCH_SPACE' | 'MAX_TRIAL_NUM';
type ExperimentStatus = 'INITIALIZED' | 'RUNNING' | 'ERROR' | 'STOPPING' | 'STOPPED' | 'DONE' | 'NO_MORE_TRIAL' | 'TUNER_NO_MORE_TRIAL';

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
        className: string;
        builtinTunerName?: string;
        codeDir?: string;
        classArgs?: any;
        classFileName?: string;
        checkpointDir: string;
        gpuNum?: number;
        includeIntermediateResults?: boolean;
    };
    assessor?: {
        className: string;
        builtinAssessorName?: string;
        codeDir?: string;
        classArgs?: any;
        classFileName?: string;
        checkpointDir: string;
        gpuNum?: number;
    };
    advisor?: {
        className: string;
        builtinAdvisorName?: string;
        codeDir?: string;
        classArgs?: any;
        classFileName?: string;
        checkpointDir: string;
        gpuNum?: number;
    };
    clusterMetaData?: {
        key: string;
        value: string;
    }[];
}

interface ExperimentProfile {
    params: ExperimentParams;
    id: string;
    execDuration: number;
    logDir?: string;
    startTime?: number;
    endTime?: number;
    maxSequenceId: number;
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
    public abstract resumeExperiment(): Promise<void>;
    public abstract stopExperiment(): Promise<void>;
    public abstract getExperimentProfile(): Promise<ExperimentProfile>;
    public abstract updateExperimentProfile(experimentProfile: ExperimentProfile, updateType: ProfileUpdateType): Promise<void>;
    public abstract importData(data: string): Promise<void>;
    public abstract exportData(): Promise<string>;

    public abstract addCustomizedTrialJob(hyperParams: string): Promise<void>;
    public abstract cancelTrialJobByUser(trialJobId: string): Promise<void>;

    public abstract listTrialJobs(status?: TrialJobStatus): Promise<TrialJobInfo[]>;
    public abstract getTrialJob(trialJobId: string): Promise<TrialJobInfo>;
    public abstract setClusterMetadata(key: string, value: string): Promise<void>;
    public abstract getClusterMetadata(key: string): Promise<string>;

    public abstract getMetricData(trialJobId?: string, metricType?: MetricType): Promise<MetricDataRecord[]>;
    public abstract getTrialJobStatistics(): Promise<TrialJobStatistics[]>;
    public abstract getStatus(): NNIManagerStatus;
}

export { Manager, ExperimentParams, ExperimentProfile, TrialJobStatistics, ProfileUpdateType, NNIManagerStatus, ExperimentStatus };
