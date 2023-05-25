// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as os from 'os';
import { assert } from 'chai';

import { IocShim } from 'common/ioc_shim';
import { Database, DataStore, TrialJobInfo } from '../../common/datastore';
import { Manager, TrialJobStatistics} from '../../common/manager';
import { TrialJobDetail } from '../../common/trainingService';
import { killPid } from '../../common/utils';
import { NNIManager } from '../../core/nnimanager';
import { SqlDB } from '../../core/sqlDatabase';
import { NNIDataStore } from '../../core/nniDataStore';
import { MockedTrainingService } from '../mock/trainingService';
import { TensorboardManager } from '../../common/tensorboardManager';
import { NNITensorboardManager } from '../../extensions/nniTensorboardManager';
import * as path from 'path';
import { RestServer } from '../../rest_server';
import globals from '../../common/globals/unittest';
import { UnitTestHelpers } from '../../core/tuner_command_channel';
import * as timersPromises from 'timers/promises';

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

async function initContainer(mode: string = 'create'): Promise<void> {
    // updating the action is not necessary for the correctness of the tests.
    // keep it here as a reminder.
    if (mode === 'resume') {
        const globalAsAny = global as any;
        globalAsAny.nni.args.action = mode;
    }
    restServer = new RestServer(globals.args.port, globals.args.urlPrefix);
    await restServer.start();
    IocShim.bind(Database, SqlDB);
    IocShim.bind(DataStore, NNIDataStore);
    IocShim.bind(Manager, NNIManager);
    IocShim.bind(TensorboardManager, NNITensorboardManager);
    await IocShim.get<DataStore>(DataStore).init();
}

async function prepareExperiment(): Promise<void> {
    // globals.showLog();
    // create ~/nni-experiments/.experiment
    const expsFile = path.join(globals.args.experimentsDirectory, '.experiment');
    if (!fs.existsSync(expsFile)) {
        fs.writeFileSync(expsFile, '{}');
    }
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

    await initContainer();
    nniManager = IocShim.get(Manager);

    // if trainingService is assigned, startExperiment won't create training service again
    const manager = nniManager as any;
    manager.trainingService = new MockedTrainingService('create_stage');
    // making the trial status polling more frequent to reduce testing time, i.e., to 1 second
    manager.pollInterval = 1;
    const expId: string = await nniManager.startExperiment(experimentParams);
    assert.strictEqual(expId, 'unittest');

    // Sleep here because the start of tuner takes a while.
    // Also, wait for that some trials are submitted, waiting for at most 10 seconds.
    // NOTE: this waiting period should be long enough depending on different running environment and randomness.
    for (let i = 0; i < 10; i++) {
        await timersPromises.setTimeout(1000);
        if (manager.currSubmittedTrialNum >= 2)
            break;
    }
    assert.isAtLeast(manager.currSubmittedTrialNum, 2);
}

async function cleanExperiment(): Promise<void> {
    const manager: any = nniManager;
    await killPid(manager.dispatcherPid);
    manager.dispatcherPid = 0;
    await manager.stopExperimentTopHalf();
    await manager.stopExperimentBottomHalf();
    await restServer.shutdown();
    IocShim.clear();
}

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

async function testUpdateExperimentProfileTrialConcurrency(concurrency: number): Promise<void> {
    let expParams = Object.assign({}, experimentParams); // skip deep copy of inner object
    expParams.trialConcurrency = concurrency;
    experimentProfile.params = expParams;
    await nniManager.updateExperimentProfile(experimentProfile, 'TRIAL_CONCURRENCY');
    const profile = await nniManager.getExperimentProfile();
    assert.strictEqual(profile.params.trialConcurrency, concurrency);
}

async function testUpdateExperimentProfileMaxExecDuration(): Promise<void> {
    let expParams = Object.assign({}, experimentParams); // skip deep copy of inner object
    expParams.maxExperimentDuration = '11s';
    experimentProfile.params = expParams;
    await nniManager.updateExperimentProfile(experimentProfile, 'MAX_EXEC_DURATION');
    const profile = await nniManager.getExperimentProfile();
    assert.strictEqual(profile.params.maxExperimentDuration, '11s');
}

async function testUpdateExperimentProfileSearchSpace(space: number[]): Promise<void> {
    let expParams = Object.assign({}, experimentParams); // skip deep copy of inner object
    // The search space here should be dict, it is stringified within nnimanager's updateSearchSpace
    const newSearchSpace = {'lr': {'_type': 'choice', '_value': space}};
    expParams.searchSpace = newSearchSpace;
    experimentProfile.params = expParams;
    await nniManager.updateExperimentProfile(experimentProfile, 'SEARCH_SPACE');
    const profile = await nniManager.getExperimentProfile();
    assert.strictEqual(profile.params.searchSpace, newSearchSpace);
}

async function testUpdateExperimentProfileMaxTrialNum(maxTrialNum: number): Promise<void> {
    let expParams = Object.assign({}, experimentParams); // skip deep copy of inner object
    expParams.maxTrialNumber = maxTrialNum;
    experimentProfile.params = expParams;
    await nniManager.updateExperimentProfile(experimentProfile, 'MAX_TRIAL_NUM');
    const profile = await nniManager.getExperimentProfile();
    assert.strictEqual(profile.params.maxTrialNumber, maxTrialNum);
}

async function testGetStatus(): Promise<void> {
    const status = nniManager.getStatus();
    // it is possible that the submitted trials run too fast to reach status NO_MORE_TRIAL
    assert.include(['RUNNING', 'NO_MORE_TRIAL'], status.status);
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
    // the mocked training service. There would be at least one trials has
    // SUCCEEDED status, i.e., '3456'.
    // '1234' may be in SUCCEEDED status or USER_CANCELED status,
    // depending on the order of SUCCEEDED and USER_CANCELED events.
    // There are 4 trials, because maxTrialNumber is updated to 4.
    // Then accordingly to the mocked training service, there are two trials
    // SUCCEEDED, one trial RUNNING, and one trial WAITING.
    // NOTE: The WAITING trial is not always submitted before the running of this test.
    // An example statistics:
    // [
    // { trialJobStatus: 'SUCCEEDED', trialJobNumber: 2 },
    // { trialJobStatus: 'RUNNING', trialJobNumber: 1 },
    // { trialJobStatus: 'WAITING', trialJobNumber: 1 }
    // ]
    // or
    // [
    // { trialJobStatus: 'USER_CANCELED', trialJobNumber: 1 },
    // { trialJobStatus: 'SUCCEEDED', trialJobNumber: 1 },
    // { trialJobStatus: 'RUNNING', trialJobNumber: 1 },
    // { trialJobStatus: 'WAITING', trialJobNumber: 1 }
    // ]
    for (let i = 0; i < 5; i++) {
        await timersPromises.setTimeout(500);
        const trialJobDetails = await nniManager.listTrialJobs();
        if (trialJobDetails.length >= 4)
            break;
    }
    const statistics = await nniManager.getTrialJobStatistics();
    assert.isAtLeast(statistics.length, 2);
    const succeededTrials: TrialJobStatistics | undefined = statistics.find(element => element.trialJobStatus === 'SUCCEEDED');
    if (succeededTrials) {
        if (succeededTrials.trialJobNumber !== 2) {
            const canceledTrials: TrialJobStatistics | undefined = statistics.find(element => element.trialJobStatus === 'USER_CANCELED');
            if (canceledTrials)
                assert.strictEqual(canceledTrials.trialJobNumber, 1);
            else
                assert.fail('USER_CANCELED trial not found when succeeded trial number is not 2!');
        }
    }
    else
        assert.fail('SUCCEEDED trial not found!');
    const runningTrials: TrialJobStatistics | undefined = statistics.find(element => element.trialJobStatus === 'RUNNING');
    if (runningTrials)
        assert.strictEqual(runningTrials.trialJobNumber, 1);
    else
        assert.fail('RUNNING trial not found!');
    const waitingTrials: TrialJobStatistics | undefined = statistics.find(element => element.trialJobStatus === 'WAITING');
    if (waitingTrials)
        assert.strictEqual(waitingTrials.trialJobNumber, 1);
    else
        assert.fail('RUNNING trial not found!');
}

async function testFinalExperimentStatus(): Promise<void> {
    const status = nniManager.getStatus();
    assert.notEqual(status.status, 'ERROR');
}


describe('Unit test for nnimanager basic testing', function () {

    before(prepareExperiment);

    // it('test addCustomizedTrialJob', () => testAddCustomizedTrialJob());
    it('test listTrialJobs', () => testListTrialJobs());
    it('test getTrialJob valid', () => testGetTrialJobValid());
    it('test getTrialJob with invalid id', () => testGetTrialJobWithInvalidId());
    it('test cancelTrialJobByUser', () => testCancelTrialJobByUser());
    it('test getExperimentProfile', () => testGetExperimentProfile());
    it('test updateExperimentProfile TRIAL_CONCURRENCY', () => testUpdateExperimentProfileTrialConcurrency(4));
    it('test updateExperimentProfile MAX_EXEC_DURATION', () => testUpdateExperimentProfileMaxExecDuration());
    it('test updateExperimentProfile SEARCH_SPACE', () => testUpdateExperimentProfileSearchSpace([0.01,0.001,0.002,0.003,0.004,0.005]));
    it('test updateExperimentProfile MAX_TRIAL_NUM', () => testUpdateExperimentProfileMaxTrialNum(4));
    it('test getStatus', () => testGetStatus());
    it('test getMetricData with trialJobId', () => testGetMetricDataWithTrialJobId());
    it('test getMetricData with invalid trialJobId', () => testGetMetricDataWithInvalidTrialJobId());
    it('test getTrialJobStatistics', () => testGetTrialJobStatistics());
    // TODO: test experiment changes from Done to Running, after maxTrialNumber/maxExecutionDuration is updated.
    // FIXME: make sure experiment crash leads to the ERROR state.
    it('test the final experiment status is not ERROR', () => testFinalExperimentStatus());

    after(cleanExperiment);

});

async function resumeExperiment(): Promise<void> {
    globals.reset();
    // the following function call show nnimanager.log in console
    // globals.showLog();
    // explicitly reset the websocket channel because it is singleton, does not work when two experiments
    // (one is start and the other is resume) run in the same process.
    UnitTestHelpers.reset();
    await initContainer('resume');
    nniManager = IocShim.get(Manager);

    // if trainingService is assigned, startExperiment won't create training service again
    const manager = nniManager as any;
    manager.trainingService = new MockedTrainingService('resume_stage');
    // making the trial status polling more frequent to reduce testing time, i.e., to 1 second
    manager.pollInterval = 1;
    // as nniManager is a singleton, manually reset its member variables here.
    manager.currSubmittedTrialNum = 0;
    manager.trialConcurrencyChange = 0;
    manager.dispatcherPid = 0;
    manager.waitingTrials = [];
    manager.trialJobs = new Map<string, TrialJobDetail>();
    manager.trialDataForTuner = '';
    manager.trialDataForResume = '';
    manager.readonly = false;
    manager.status = {
        status: 'INITIALIZED',
        errors: []
    };
    await nniManager.resumeExperiment(false);
}

async function testMaxTrialNumberAfterResume(): Promise<void> {
    // testing the resumed nnimanager correctly counts (max) trial number
    // waiting 18 seconds to make trials reach maxTrialNum, waiting this long
    // because trial concurrency is set to 1 and macos CI is pretty slow.
    await timersPromises.setTimeout(18000);
    const trialJobDetails = await nniManager.listTrialJobs();
    assert.strictEqual(trialJobDetails.length, 5);
}

async function testAddCustomizedTrialJobFail(): Promise<void> {
    // will fail because the max trial number has already reached
    await nniManager.addCustomizedTrialJob('{"lr": 0.006}')
    .catch((err: Error) => {
        assert.strictEqual(err.message, 'reach maxTrialNum');
    });
}

async function testAddCustomizedTrialJob(): Promise<void> {
    // max trial number has been extended to 7, adding customized trial here will be succeeded
    const sequenceId = await nniManager.addCustomizedTrialJob('{"lr": 0.006}');
    await timersPromises.setTimeout(1000);
    const trialJobDetails = await nniManager.listTrialJobs();
    const customized = trialJobDetails.find(element =>
        element.hyperParameters !== undefined
        && element.hyperParameters[0] === '{"parameter_id":null,"parameter_source":"customized","parameters":{"lr":0.006}}');
    assert.notEqual(customized, undefined);
}

// NOTE: this describe should be executed in couple with the above describe
describe('Unit test for nnimanager resume testing', function() {

    before(resumeExperiment);

    // First update maxTrialNumber to 5 for the second test
    it('test updateExperimentProfile TRIAL_CONCURRENCY', () => testUpdateExperimentProfileTrialConcurrency(1));
    it('test updateExperimentProfile MAX_TRIAL_NUM', () => testUpdateExperimentProfileMaxTrialNum(5));
    it('test max trial number after resume', () => testMaxTrialNumberAfterResume());
    it('test add customized trial job failure', () => testAddCustomizedTrialJobFail());
    // update search to contain only one hyper config, update maxTrialNum to add additional two trial budget,
    // then a customized trial can be submitted successfully.
    // NOTE: trial concurrency should be set to 1 to avoid tuner sending too many trials before the space is updated
    it('test updateExperimentProfile SEARCH_SPACE', () => testUpdateExperimentProfileSearchSpace([0.008]));
    it('test updateExperimentProfile MAX_TRIAL_NUM', () => testUpdateExperimentProfileMaxTrialNum(7));
    it('test add customized trial job succeeded', () => testAddCustomizedTrialJob());
    it('test the final experiment status is not ERROR', () => testFinalExperimentStatus());

    after(cleanExperiment);

});
