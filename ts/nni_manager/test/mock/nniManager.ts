// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { Deferred } from 'ts-deferred';

import { MetricDataRecord, MetricType, TrialJobInfo } from '../../common/datastore';
import { MethodNotImplementedError } from '../../common/errors';
import {
    ExperimentConfig, ExperimentProfile, Manager, ProfileUpdateType,
    TrialJobStatistics, NNIManagerStatus
} from '../../common/manager';
import {
    TrialJobApplicationForm, TrialJobDetail, TrialJobStatus
} from '../../common/trainingService';

export class MockedNNIManager extends Manager {
    public getStatus(): NNIManagerStatus {
        return {
            status: 'RUNNING',
            errors: []
        }
    }
    public updateExperimentProfile(_experimentProfile: ExperimentProfile, _updateType: ProfileUpdateType): Promise<void> {
        return Promise.resolve();
    }
    public importData(_data: string): Promise<void> {
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
    public addCustomizedTrialJob(_hyperParams: string): Promise<number> {
        return Promise.resolve(99);
    }

    public resumeExperiment(): Promise<void> {
        return Promise.resolve();
    }

    public submitTrialJob(_form: TrialJobApplicationForm): Promise<TrialJobDetail> {
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

    public cancelTrialJobByUser(_trialJobId: string): Promise<void> {
        return Promise.resolve();
    }

    public getClusterMetadata(_key: string): Promise<string> {
        return Promise.resolve('METAVALUE1');
    }

    public startExperiment(_experimentParams: ExperimentConfig): Promise<string> {
        return Promise.resolve('id-1234');
    }

    public setClusterMetadata(_key: string, _value: string): Promise<void> {
        return Promise.resolve();
    }

    public getTrialJob(_trialJobId: string): Promise<TrialJobInfo> {
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
    public stopExperimentTopHalf(): Promise<void> {
        throw new MethodNotImplementedError();
    }
    public stopExperimentBottomHalf(): Promise<void> {
        throw new MethodNotImplementedError();
    }
    public getMetricData(_trialJobId: string, _metricType: MetricType): Promise<MetricDataRecord[]> {
        throw new MethodNotImplementedError();
    }
    public getMetricDataByRange(_minSeqId: number, _maxSeqId: number): Promise<MetricDataRecord[]> {
        throw new MethodNotImplementedError();
    }
    public getLatestMetricData(): Promise<MetricDataRecord[]> {
        throw new MethodNotImplementedError();
    }
    public getTrialFile(_trialJobId: string, _fileName: string): Promise<string> {
        throw new MethodNotImplementedError();
    }
    public getExperimentProfile(): Promise<ExperimentProfile> {
        const profile: ExperimentProfile = <any>{
            params: {
                experimentName: 'exp1',
                trialConcurrency: 2,
                maxExperimentDuration: '30s',
                maxTrialNumber: 3,
                trainingService: {
                    platform: 'local'
                },
                searchSpace: '{lr: 0.01}',
                tuner: {
                    className: 'testTuner',
                },
                trialCommand: '',
                trialCodeDirectory: '',
                debug: true
            },
            id: '2345',
            execDuration: 0,
            logDir: '',
            startTime: Date.now(),
            endTime: Date.now(),
            nextSequenceId: 0,
            revision: 0
        };

        return Promise.resolve(profile);
    }
    public listTrialJobs(_status?: TrialJobStatus): Promise<TrialJobInfo[]> {
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

    public async getTrialOutputLocalPath(_trialJobId: string): Promise<string> {
        throw new MethodNotImplementedError();
    }

    public async fetchTrialOutput(_trialJobId: string, _subpath: string): Promise<void> {
        throw new MethodNotImplementedError();
    }
}
