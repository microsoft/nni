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

import { Deferred } from 'ts-deferred';
import { Provider } from 'typescript-ioc';

import { MetricDataRecord, MetricType, TrialJobInfo } from '../../common/datastore';
import { MethodNotImplementedError } from '../../common/errors';
import {
    ExperimentParams, ExperimentProfile, Manager, ProfileUpdateType,
    TrialJobStatistics, NNIManagerStatus
} from '../../common/manager';
import {
    TrialJobApplicationForm, TrialJobDetail, TrialJobStatus
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
    public addCustomizedTrialJob(hyperParams: string): Promise<void> {
        return Promise.resolve();
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
            // tslint:disable-next-line:no-http-string
            url: 'http://test',
            workingDirectory: '/tmp/mocked',
            sequenceId: 0,
            form: {
                jobType: 'TRIAL'
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
            id: '1234',
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
            maxSequenceId: 0,
            revision: 0
        };

        return Promise.resolve(profile);
    }
    public listTrialJobs(status?: TrialJobStatus): Promise<TrialJobInfo[]> {
        const job1: TrialJobInfo = {
            id: '1234',
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
            id: '3456',
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
