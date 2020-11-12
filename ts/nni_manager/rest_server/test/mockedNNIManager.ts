// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { Deferred } from 'ts-deferred';
import { Provider } from 'typescript-ioc';

import { MetricDataRecord, MetricType, TrialJobInfo } from '../../common/datastore';
import { MethodNotImplementedError } from '../../common/errors';
import {
    ExperimentParams, ExperimentProfile, Manager, ProfileUpdateType,
    TrialJobStatistics, NNIManagerStatus
} from '../../common/manager';
import {
    TrialJobApplicationForm, TrialJobDetail, TrialJobStatus, LogType
} from '../../common/trainingService';

export const testManagerProvider: Provider = {
    get: (): Manager => { return new MockedNNIManager(); }
};

export class MockedNNIManager extends Manager {
    public getStatus(): NNIManagerStatus {
        return {
            status: 'RUNNING',
            errors: []
        }
    }
    public updateExperimentProfile(experimentProfile: ExperimentProfile, updateType: ProfileUpdateType): Promise<void> {
        return Promise.resolve();
    }
    public importData(data: string): Promise<void> {
        return Promise.resolve();
    }
    public getImportedData(): Promise<string[]> {
        const ret: string[] = ["1", "2"];
        return Promise.resolve(ret);
    }
    public async exportData(): Promise<string> {
        const ret: string = '';
        return Promise.resolve(ret);
    }
    public getTrialJobStatistics(): Promise<TrialJobStatistics[]> {
        const deferred: Deferred<TrialJobStatistics[]> = new Deferred<TrialJobStatistics[]>();
        deferred.resolve([{
            trialJobStatus: 'RUNNING',
            trialJobNumber: 2
        }, {
            trialJobStatus: 'FAILED',
            trialJobNumber: 1
        }]);

        return deferred.promise;
    }
    public addCustomizedTrialJob(hyperParams: string): Promise<number> {
        return Promise.resolve(99);
    }

    public resumeExperiment(): Promise<void> {
        return Promise.resolve();
    }

    public submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const deferred: Deferred<TrialJobDetail> = new Deferred<TrialJobDetail>();
        const jobDetail: TrialJobDetail = {
            id: '1234',
            status: 'RUNNING',
            submitTime: Date.now(),
            startTime: Date.now(),
            endTime: Date.now(),
            tags: ['test'],
            url: 'http://test',
            workingDirectory: '/tmp/mocked',
            form: {
                sequenceId: 0,
                hyperParameters: { value: '', index: 0 }
            }
        };
        deferred.resolve(jobDetail);

        return deferred.promise;
    }

    public cancelTrialJobByUser(trialJobId: string): Promise<void> {
        return Promise.resolve();
    }

    public getClusterMetadata(key: string): Promise<string> {
        return Promise.resolve('METAVALUE1');
    }

    public startExperiment(experimentParams: ExperimentParams): Promise<string> {
        return Promise.resolve('id-1234');
    }

    public setClusterMetadata(key: string, value: string): Promise<void> {
        return Promise.resolve();
    }

    public getTrialJob(trialJobId: string): Promise<TrialJobInfo> {
        const deferred: Deferred<TrialJobInfo> = new Deferred<TrialJobInfo>();
        const jobInfo: TrialJobInfo = {
            trialJobId: '1234',
            status: 'SUCCEEDED',
            startTime: Date.now(),
            endTime: Date.now()
        };
        deferred.resolve(jobInfo);

        return deferred.promise;
    }
    public stopExperiment(): Promise<void> {
        throw new MethodNotImplementedError();
    }
    public getMetricData(trialJobId: string, metricType: MetricType): Promise<MetricDataRecord[]> {
        throw new MethodNotImplementedError();
    }
    public getMetricDataByRange(minSeqId: number, maxSeqId: number): Promise<MetricDataRecord[]> {
        throw new MethodNotImplementedError();
    }
    public getLatestMetricData(): Promise<MetricDataRecord[]> {
        throw new MethodNotImplementedError();
    }
    public getTrialLog(trialJobId: string, logType: LogType): Promise<string> {
        throw new MethodNotImplementedError();
    }
    public getExperimentProfile(): Promise<ExperimentProfile> {
        const profile: ExperimentProfile = {
            params: {
                authorName: 'test',
                experimentName: 'exp1',
                trialConcurrency: 2,
                maxExecDuration: 30,
                maxTrialNum: 3,
                trainingServicePlatform: 'local',
                searchSpace: '{lr: 0.01}',
                tuner: {
                    className: 'testTuner',
                    checkpointDir: ''
                }
            },
            id: '2345',
            execDuration: 0,
            startTime: Date.now(),
            endTime: Date.now(),
            nextSequenceId: 0,
            revision: 0
        };

        return Promise.resolve(profile);
    }
    public listTrialJobs(status?: TrialJobStatus): Promise<TrialJobInfo[]> {
        const job1: TrialJobInfo = {
            trialJobId: '1234',
            status: 'SUCCEEDED',
            startTime: Date.now(),
            endTime: Date.now(),
            finalMetricData: [{
                timestamp: 0,
                trialJobId: '3456',
                parameterId: '123',
                type: 'FINAL',
                sequence: 0,
                data: '0.2'
            }]
        };
        const job2: TrialJobInfo = {
            trialJobId: '3456',
            status: 'FAILED',
            startTime: Date.now(),
            endTime: Date.now(),
            finalMetricData: [{
                timestamp: 0,
                trialJobId: '3456',
                parameterId: '123',
                type: 'FINAL',
                sequence: 0,
                data: '0.2'
            }]
        };

        return Promise.resolve([job1, job2]);
    }
}
