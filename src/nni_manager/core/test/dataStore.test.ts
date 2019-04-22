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

import { expect } from 'chai';
import { Container, Scope } from 'typescript-ioc';

import * as component from '../../common/component';
import { Database, DataStore, TrialJobInfo } from '../../common/datastore';
import { setExperimentStartupInfo } from '../../common/experimentStartupInfo';
import { ExperimentProfile, TrialJobStatistics } from '../../common/manager';
import { TrialJobStatus } from '../../common/trainingService';
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { NNIDataStore } from '../nniDataStore';
import { SqlDB } from '../sqlDatabase';

describe('Unit test for dataStore', () => {
    let ds: DataStore;
    before(async () => {
        prepareUnitTest();
        Container.bind(Database).to(SqlDB).scope(Scope.Singleton);
        Container.bind(DataStore).to(NNIDataStore).scope(Scope.Singleton);
        ds = component.get(DataStore);
        await ds.init();
    });

    after(() => {
        ds.close();
        cleanupUnitTest();
    });

    it('test emtpy experiment profile', async () => {
        const result: ExperimentProfile = await ds.getExperimentProfile('abc');
        expect(result).to.equal(undefined, 'Should not get any profile');
    });

    it('test experiment profiles CRUD', async () => {
        const profile: ExperimentProfile = {
            params: {
                authorName: 'test1',
                experimentName: 'exp1',
                trialConcurrency: 2,
                maxExecDuration: 10,
                maxTrialNum: 5,
                trainingServicePlatform: 'local',
                searchSpace: `{
                    "dropout_rate": {
                        "_type": "uniform",
                        "_value": [0.1, 0.5]
                    },
                    "batch_size": {
                        "_type": "choice",
                        "_value": [50, 250, 500]
                    }
                }`,
                tuner: {
                    className: 'testTuner',
                    checkpointDir: '/tmp/cp',
                    gpuNum: 0
                }
            },
            id: 'exp123',
            execDuration: 0,
            startTime: Date.now(),
            endTime: Date.now(),
            maxSequenceId: 0,
            revision: 0
        }
        const id: string = profile.id;
        for (let i: number = 0; i < 5; i++) {
            await ds.storeExperimentProfile(profile);
            profile.revision += 1;
        }
        const result: ExperimentProfile = await ds.getExperimentProfile(id);
        expect(result.revision).to.equal(4);
    });

    const testEventRecords: {
        event: string;
        jobId: string;
        data?: string;
    }[] = [
        {
            event: 'WAITING',
            jobId: '111'
        },
        {
            event: 'WAITING',
            jobId: '222'
        },
        {
            event: 'RUNNING',
            jobId: '111'
        },
        {
            event: 'RUNNING',
            jobId: '222'
        },
        {
            event: 'SUCCEEDED',
            jobId: '111',
            data: 'lr: 0.001'
        },
        {
            event: 'FAILED',
            jobId: '222'
        }
    ];

    // tslint:disable-next-line:no-any
    const metricsData: any = [
        {
            trial_job_id: '111',
            parameter_id: 'abc',
            type: 'PERIODICAL',
            value: 'acc: 0.88',
            timestamp: Date.now()
        },
        {
            trial_job_id: '111',
            parameter_id: 'abc',
            type: 'FINAL',
            value: 'acc: 0.88',
            timestamp: Date.now()
        }
    ];

    it('test trial job events store /query', async () => {
        for (const event of testEventRecords) {
            await ds.storeTrialJobEvent(<TrialJobStatus>event.event, event.jobId, event.data);
        }
        for (const metrics of metricsData) {
            await ds.storeMetricData(metrics.trial_job_id, JSON.stringify(metrics));
        }
        const jobs: TrialJobInfo[] = await ds.listTrialJobs();
        expect(jobs.length).to.equals(2, 'There should be 2 jobs');

        const statistics: TrialJobStatistics[] = await ds.getTrialJobStatistics();
        expect(statistics.length).to.equals(2, 'There should be 2 statistics');
    });
});
