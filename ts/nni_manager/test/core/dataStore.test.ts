// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { expect } from 'chai';

import { IocShim } from 'common/ioc_shim';
import { Database, DataStore, TrialJobInfo } from '../../common/datastore';
import { ExperimentProfile, TrialJobStatistics } from '../../common/manager';
import { TrialJobStatus } from '../../common/trainingService';
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import { NNIDataStore } from '../../core/nniDataStore';
import { SqlDB } from '../../core/sqlDatabase';

describe('Unit test for dataStore', () => {
    let ds: DataStore;
    before(async () => {
        prepareUnitTest();
        IocShim.bind(Database, SqlDB);
        IocShim.bind(DataStore, NNIDataStore);
        ds = IocShim.get(DataStore);
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
        const profile: ExperimentProfile = <ExperimentProfile>{
            params: {
                experimentName: 'exp1',
                trialConcurrency: 2,
                maxExperimentDuration: '10s',
                maxTrialNumber: 5,
                trainingService: {
                    platform: 'local'
                },
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
                    className: 'testTuner'
                },
                trialCommand: '',
                trialCodeDirectory: '',
                debug: true
            },
            id: 'exp123',
            execDuration: 0,
            logDir: '',
            startTime: Date.now(),
            endTime: Date.now(),
            nextSequenceId: 0,
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
