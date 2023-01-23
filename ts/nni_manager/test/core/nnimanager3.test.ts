// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as os from 'os';
import { assert, expect } from 'chai';
import { Container, Scope } from 'typescript-ioc';

import * as component from '../../common/component';
import { Database, DataStore, TrialJobInfo } from '../../common/datastore';
import { Manager, ExperimentProfile, TrialJobStatistics} from '../../common/manager';
import { TrainingService, TrialJobDetail } from '../../common/trainingService';
import { cleanupUnitTest, prepareUnitTest, killPid } from '../../common/utils';
import { NNIManager } from '../../core/nnimanager';
import { SqlDB } from '../../core/sqlDatabase';
import { NNIDataStore } from '../../core/nniDataStore';
import { MockedTrainingService } from '../mock/trainingService';
import { TensorboardManager } from '../../common/tensorboardManager';
import { NNITensorboardManager } from '../../extensions/nniTensorboardManager';
import { initExperimentsManager } from '../../extensions/experiments_manager';
import * as path from 'path';
import { UnitTestHelpers } from 'core/ipcInterface';
import { RestServer } from '../../rest_server';
import globals, { initGlobals } from '../../common/globals';
import * as timersPromises from 'timers/promises';
// import { before, it } from 'node:test';

let nniManager: NNIManager;
let experimentParams: any = {
    experimentName: 'naive_experiment',
    trialConcurrency: 3,
    maxExperimentDuration: '10s',
    maxTrialNumber: 3,
    trainingService: {
        platform: 'local'
    },
    searchSpace: {'lr': {'_type': 'choice', '_value': [0.01,0.001,0.002,0.003,0.004]}},
    // use Random because no metric data from mocked training service
    tuner: {
        name: 'Random'
    },
    // skip assessor
    // assessor: {
    //     name: 'Medianstop'
    // },
    // trialCommand does not take effect in mocked training service
    trialCommand: 'sleep 2',
    trialCodeDirectory: '',
    debug: true
}
let experimentProfile: any = {
    params: experimentParams,
    // the experiment profile can only keep params,
    // because the update logic only touch key-values in params.
    // it violates the type of ExperimentProfile, but it is okay.
}
let mockedInfo = {
    "id": "unittest",
    "port": 8080,
    "startTime": 1605246730756,
    "endTime": "N/A",
    "status": "INITIALIZED",
    "platform": "local",
    "experimentName": "testExp",
    "tag": [],
    "pid": 11111,
    "webuiUrl": [],
    "logDir": null
}

let restServer: RestServer;

async function initContainer(): Promise<void> {
    console.log(globals);
    restServer = new RestServer(globals.args.port, globals.args.urlPrefix);
    await restServer.start();
    Container.bind(Manager).to(NNIManager).scope(Scope.Singleton);
    Container.bind(Database).to(SqlDB).scope(Scope.Singleton);
    Container.bind(DataStore).to(NNIDataStore).scope(Scope.Singleton);
    Container.bind(TensorboardManager).to(NNITensorboardManager).scope(Scope.Singleton);
    await component.get<DataStore>(DataStore).init();
}

async function prepareExperiment(): Promise<void> {
    console.log('hello0');
    // clean the db file under the unittest experiment directory.
    // NOTE: cannot remove the whole exp directory, it seems the directory is created before this line.
    const unittestPath = path.join(globals.args.experimentsDirectory, globals.args.experimentId, 'db');
    fs.rmSync(unittestPath, { recursive: true, force: true });

    // Write the experiment info to ~/nni-experiments/.experiment before experiment start.
    // Do not use file lock for simplicity.
    // The ut also works if not updating experiment info but ExperimentsManager will complain.
    const fileInfo: Buffer = fs.readFileSync(globals.paths.experimentsList);
    let experimentsInformation = JSON.parse(fileInfo.toString());
    experimentsInformation['unittest'] = mockedInfo;
    fs.writeFileSync(globals.paths.experimentsList, JSON.stringify(experimentsInformation, null, 4));

    console.log('hello');
    await initContainer();
    console.log('hello2');
    nniManager = component.get(Manager);
    console.log('hello4');

    // if trainingService is assigned, startExperiment won't create training service again
    const manager = nniManager as any;
    manager.trainingService = new MockedTrainingService('create_stage');
    // making the trial status polling more frequent to reduce testing time, i.e., to 1 second
    manager.pollInterval = 1;
    const expId: string = await nniManager.startExperiment(experimentParams);
    assert.strictEqual(expId, 'unittest');

    // Sleep here because the start of tuner takes a while.
    // Also, wait for that some trials are submitted, waiting for at most 3 seconds.
    for (let i = 0; i < 5; i++) {
        await timersPromises.setTimeout(500);
        if (manager.currSubmittedTrialNum >= 2)
            break;
    }
    assert.isAtLeast(manager.currSubmittedTrialNum, 2);
}

async function cleanExperiment(): Promise<void> {
    // FIXME: more proper clean up
    const manager: any = nniManager;
    console.log('hello5');
    await killPid(manager.dispatcherPid);
    manager.dispatcherPid = 0;
    console.log('hello6');
    await manager.stopExperimentTopHalf();
    await restServer.shutdown();
    console.log('hello7');
    // cleanupUnitTest();
    console.log('hello8');
}

// async function testAddCustomizedTrialJob(): Promise<void> {
//     await nniManager.addCustomizedTrialJob('"hyperParams"').then(() => {
//     }).catch((error) => {
//         assert.fail(error);
//     })
// }

async function testListTrialJobs(): Promise<void> {
    await timersPromises.setTimeout(200);
    const trialJobDetails = await nniManager.listTrialJobs();
    assert.isAtLeast(trialJobDetails.length, 2);
}

async function testGetTrialJobValid(): Promise<void> {
    const trialJobDetail = await nniManager.getTrialJob('1234');
    assert.strictEqual(trialJobDetail.trialJobId, '1234');
}

async function testGetTrialJobWithInvalidId(): Promise<void> {
    // query a not exist id, getTrialJob returns undefined,
    // because getTrialJob queries data from db
    const trialJobDetail = await nniManager.getTrialJob('4321');
    assert.strictEqual(trialJobDetail, undefined);
}

async function testCancelTrialJobByUser(): Promise<void> {
    await nniManager.cancelTrialJobByUser('1234');
    // test datastore to verify the trial is cancelled and the event is stored in db
    // NOTE: it seems a SUCCEEDED trial can also be cancelled
    const manager = nniManager as any;
    const trialJobInfo: TrialJobInfo = await manager.dataStore.getTrialJob('1234');
    assert.strictEqual(trialJobInfo.status, 'USER_CANCELED');
}

async function testGetExperimentProfile(): Promise<void> {
    const profile = await nniManager.getExperimentProfile();
    assert.strictEqual(profile.id, 'unittest');
    assert.strictEqual(profile.logDir, path.join(os.homedir(),'nni-experiments','unittest'));
}

async function testUpdateExperimentProfileTrialConcurrency(): Promise<void> {
    let expParams = Object.assign({}, experimentParams); // skip deep copy of inner object
    expParams.trialConcurrency = 3;
    experimentProfile.params = expParams;
    await nniManager.updateExperimentProfile(experimentProfile, 'TRIAL_CONCURRENCY');
    const profile = await nniManager.getExperimentProfile();
    assert.strictEqual(profile.params.trialConcurrency, 3);
}

async function testUpdateExperimentProfileMaxExecDuration(): Promise<void> {
    let expParams = Object.assign({}, experimentParams); // skip deep copy of inner object
    expParams.maxExperimentDuration = '11s';
    experimentProfile.params = expParams;
    await nniManager.updateExperimentProfile(experimentProfile, 'MAX_EXEC_DURATION');
    const profile = await nniManager.getExperimentProfile();
    assert.strictEqual(profile.params.maxExperimentDuration, '11s');
}

async function testUpdateExperimentProfileSearchSpace(): Promise<void> {
    let expParams = Object.assign({}, experimentParams); // skip deep copy of inner object
    // The search space here should be dict, it is stringified within nnimanager's updateSearchSpace
    const newSearchSpace = {'lr': {'_type': 'choice', '_value': [0.01, 0.001, 0.002, 0.004, 0.008]}};
    expParams.searchSpace = newSearchSpace;
    experimentProfile.params = expParams;
    await nniManager.updateExperimentProfile(experimentProfile, 'SEARCH_SPACE');
    const profile = await nniManager.getExperimentProfile();
    assert.strictEqual(profile.params.searchSpace, newSearchSpace);
}

async function testUpdateExperimentProfileMaxTrialNum(): Promise<void> {
    let expParams = Object.assign({}, experimentParams); // skip deep copy of inner object
    expParams.maxTrialNumber = 4;
    experimentProfile.params = expParams;
    await nniManager.updateExperimentProfile(experimentProfile, 'MAX_TRIAL_NUM');
    const profile = await nniManager.getExperimentProfile();
    assert.strictEqual(profile.params.maxTrialNumber, 4);
}

async function testGetStatus(): Promise<void> {
    const status = nniManager.getStatus();
    assert.strictEqual(status.status, 'RUNNING');
}

async function testGetMetricDataWithTrialJobId(): Promise<void> {
    // Query an exist trialJobId
    // The metric is synthesized in the mocked training service
    await timersPromises.setTimeout(600);
    const metrics = await nniManager.getMetricData('1234');
    assert.strictEqual(metrics.length, 1);
    assert.strictEqual(metrics[0].type, 'FINAL');
    assert.strictEqual(metrics[0].data, '"0.9"');
}

async function testGetMetricDataWithInvalidTrialJobId(): Promise<void> {
    // Query an invalid trialJobId
    const metrics = await nniManager.getMetricData('4321');
    // The returned is an empty list
    assert.strictEqual(metrics.length, 0);
}

async function testGetTrialJobStatistics(): Promise<void> {
    // Waiting for 1 second to make sure SUCCEEDED status has been sent from
    // the mocked training service. There would be at least two trials has
    // SUCCEEDED status, i.e., trial '1234' and '3456', though '1234' has been
    // cancelled (by a former test) before the SUCCEEDED status.
    await timersPromises.setTimeout(1000);
    const statistics = await nniManager.getTrialJobStatistics();
    console.log(statistics);
    assert.strictEqual(statistics.length, 2);
    const oneItem: TrialJobStatistics | undefined = statistics.find(element => element.trialJobStatus === 'SUCCEEDED');
    if (oneItem)
        assert.strictEqual(oneItem.trialJobNumber, 2);
    else
        assert.fail('SUCCEEDED trial not found!');
}

// async function testAddCustomizedTrialJobReachMaxTrialNumber(): Promise<void> {
//     // test currSubmittedTrialNum reach maxTrialNumber
//     nniManager.addCustomizedTrialJob('"hyperParam"').then(() => {
//         nniManager.getTrialJobStatistics().then(function (trialJobStatistics) {
//             if (trialJobStatistics[0].trialJobStatus === 'WAITING')
//                 expect(trialJobStatistics[0].trialJobNumber).to.be.equal(2);
//             else
//                 expect(trialJobStatistics[1].trialJobNumber).to.be.equal(2);
//         })
//     }).catch((error) => {
//         assert.fail(error);
//     })
// }

async function testFinalExperimentStatus(): Promise<void> {
    await timersPromises.setTimeout(3000);
    const status = await nniManager.getStatus();
    console.log(status);
    assert.strictEqual(status.status, 'DONE');
    // assert.strictEqual('1', '1');
}


// FIXME: timeout on macOS
describe('Unit test for nnimanager hello world', function () {

    before(prepareExperiment);

    // it('test addCustomizedTrialJob', () => testAddCustomizedTrialJob());
    it('test listTrialJobs', () => testListTrialJobs());
    it('test getTrialJob valid', () => testGetTrialJobValid());
    it('test getTrialJob with invalid id', () => testGetTrialJobWithInvalidId());
    it('test cancelTrialJobByUser', () => testCancelTrialJobByUser());
    it('test getExperimentProfile', () => testGetExperimentProfile());
    it('test updateExperimentProfile TRIAL_CONCURRENCY', () => testUpdateExperimentProfileTrialConcurrency());
    it('test updateExperimentProfile MAX_EXEC_DURATION', () => testUpdateExperimentProfileMaxExecDuration());
    it('test updateExperimentProfile SEARCH_SPACE', () => testUpdateExperimentProfileSearchSpace());
    it('test updateExperimentProfile MAX_TRIAL_NUM', () => testUpdateExperimentProfileMaxTrialNum());
    it('test getStatus', () => testGetStatus());
    it('test getMetricData with trialJobId', () => testGetMetricDataWithTrialJobId());
    it('test getMetricData with invalid trialJobId', () => testGetMetricDataWithInvalidTrialJobId());
    it('test getTrialJobStatistics', () => testGetTrialJobStatistics());
    // it('test addCustomizedTrialJob reach maxTrialNumber', () => testAddCustomizedTrialJobReachMaxTrialNumber());
    it('test the final experiment status is not ERROR', () => testFinalExperimentStatus());

    after(cleanExperiment);

});