// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import * as os from 'os';
import * as path from 'path';
import { Database, MetricDataRecord, TrialJobEvent, TrialJobEventRecord } from '../../common/datastore';
import { ExperimentConfig, ExperimentProfile } from '../../common/manager';
import { cleanupUnitTest, getDefaultDatabaseDir, mkDirP, prepareUnitTest } from '../../common/utils';
import { SqlDB } from '../../core/sqlDatabase';

const expParams1: ExperimentConfig = <any>{
    experimentName: 'Exp1',
    trialConcurrency: 3,
    maxExperimentDuration: '100s',
    maxTrialNumber: 5,
    trainingService: {
        platform: 'local'
    },
    searchSpace: 'SS',
    tuner: {
        className: 'testTuner'
    },
    trialCommand: '',
    trialCodeDirectory: '',
    debug: true
};

const expParams2: ExperimentConfig = <any>{
    experimentName: 'Exp2',
    trialConcurrency: 5,
    maxExperimentDuration: '1000s',
    maxTrialNumber: 5,
    trainingService: {
        platform: 'local'
    },
    searchSpace: '',
    tuner: {
        className: 'testTuner'
    },
    assessor: {
        className: 'testAssessor'
    },
    trialCommand: '',
    trialCodeDirectory: '',
    debug: true
};

const profiles: ExperimentProfile[] = [
    { params: expParams1, id: '#1', execDuration: 0, logDir: '/log', startTime: Date.now(), endTime: undefined, nextSequenceId: 0, revision: 1,},
    { params: expParams1, id: '#1', execDuration: 0, logDir: '/log', startTime: Date.now(), endTime: Date.now(), nextSequenceId: 1, revision: 2 },
    { params: expParams2, id: '#2', execDuration: 0, logDir: '/log', startTime: Date.now(), endTime: Date.now(), nextSequenceId: 0, revision: 2 },
    { params: expParams2, id: '#2', execDuration: 0, logDir: '/log', startTime: Date.now(), endTime: Date.now(), nextSequenceId: 2, revision: 3 }
];

const events: TrialJobEventRecord[] = [
    { timestamp: Date.now(), event: 'WAITING', trialJobId: 'A', data: 'hello' },     // 0
    { timestamp: Date.now(), event: 'UNKNOWN', trialJobId: 'B', data: 'world' },     // 1
    { timestamp: Date.now(), event: 'RUNNING', trialJobId: 'B', data: undefined },   // 2
    { timestamp: Date.now(), event: 'RUNNING', trialJobId: 'A', data: '123' },       // 3
    { timestamp: Date.now(), event: 'FAILED', trialJobId: 'A', data: undefined }     // 4
];

const metrics: MetricDataRecord[] = [
    { timestamp: Date.now(), trialJobId: 'A', parameterId: '1', type: 'PERIODICAL', sequence: 0, data: 1.1 },   // 0
    { timestamp: Date.now(), trialJobId: 'B', parameterId: '2', type: 'PERIODICAL', sequence: 0, data: 2.1 },   // 1
    { timestamp: Date.now(), trialJobId: 'A', parameterId: '1', type: 'PERIODICAL', sequence: 1, data: 1.2 },   // 2
    { timestamp: Date.now(), trialJobId: 'A', parameterId: '1', type: 'FINAL', sequence: 0, data: 1.3 },        // 3
    { timestamp: Date.now(), trialJobId: 'C', parameterId: '2', type: 'PERIODICAL', sequence: 1, data: 2.1 },   // 4
    { timestamp: Date.now(), trialJobId: 'C', parameterId: '2', type: 'FINAL', sequence: 0, data: 2.2 }         // 5
];

function assertRecordEqual(record: any, value: any): void {
    assert.ok(record.timestamp > new Date(2018, 6, 1).getTime());
    assert.ok(record.timestamp < Date.now());

    for (const key in value) {
        if (key !== 'timestamp') {
            assert.equal(record[key], value[key]);
        }
    }
}

function assertRecordsEqual(records: any[], inputs: any[], indices: number[]): void {
    assert.equal(records.length, indices.length);
    for (let i: number = 0; i < records.length; i++) {
        assertRecordEqual(records[i], inputs[indices[i]]);
    }
}

describe('core/sqlDatabase', () => {
    let db: SqlDB | undefined;

    before(async () => {
        prepareUnitTest();
        const dbDir: string = getDefaultDatabaseDir();
        await mkDirP(dbDir);
        db = new SqlDB();
        await (<SqlDB>db).init(true, dbDir);
        for (const profile of profiles) {
            await (<SqlDB>db).storeExperimentProfile(profile);
        }
        for (const event of events) {
            await (<SqlDB>db).storeTrialJobEvent(<TrialJobEvent>event.event, event.trialJobId, Date.now(), event.data);
        }
        for (const metric of metrics) {
            await (<SqlDB>db).storeMetricData(metric.trialJobId, JSON.stringify(metric));
        }
    });

    after(() => {
        cleanupUnitTest();
    });

    it('queryExperimentProfile without revision', async () => {
        const records: ExperimentProfile[] = await (<SqlDB>db).queryExperimentProfile('#1');
        assert.equal(records.length, 2);
        assert.deepEqual(records[0], profiles[1]);
        assert.deepEqual(records[1], profiles[0]);
    });

    it('queryExperimentProfile with revision', async () => {
        const records: ExperimentProfile[] = await (<SqlDB>db).queryExperimentProfile('#1', 2);
        assert.equal(records.length, 1);
        assert.deepEqual(records[0], profiles[1]);
    });

    it('queryLatestExperimentProfile', async () => {
        const record: ExperimentProfile = await (<SqlDB>db).queryLatestExperimentProfile('#2');
        assert.deepEqual(record, profiles[3]);
    });

    it('queryTrialJobEventByEvent without trialJobId', async () => {
        const records: TrialJobEventRecord[] = await (<SqlDB>db).queryTrialJobEvent(undefined, 'RUNNING');
        assertRecordsEqual(records, events, [2, 3]);
    });

    it('queryTrialJobEventByEvent with trialJobId', async () => {
        const records: TrialJobEventRecord[] = await (<SqlDB>db).queryTrialJobEvent('A', 'RUNNING');
        assertRecordsEqual(records, events, [3]);
    });

    it('queryTrialJobEventById', async () => {
        const records: TrialJobEventRecord[] = await (<SqlDB>db).queryTrialJobEvent('B');
        assertRecordsEqual(records, events, [1, 2]);
    });

    it('queryMetricDataByType without trialJobId', async () => {
        const records: MetricDataRecord[] = await (<SqlDB>db).queryMetricData(undefined, 'FINAL');
        assertRecordsEqual(records, metrics, [3, 5]);
    });

    it('queryMetricDataByType with trialJobId', async () => {
        const records: MetricDataRecord[] = await (<SqlDB>db).queryMetricData('A', 'PERIODICAL');
        assertRecordsEqual(records, metrics, [0, 2]);
    });

    it('queryMetricDataById', async () => {
        const records: MetricDataRecord[] = await (<SqlDB>db).queryMetricData('B');
        assertRecordsEqual(records, metrics, [1]);
    });

    it('empty result', async () => {
        const records: MetricDataRecord[] = await (<SqlDB>db).queryMetricData('X');
        assert.equal(records.length, 0);
    });

});
