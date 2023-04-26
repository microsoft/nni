// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import fs from 'fs/promises';
import { Server } from 'http';
import os from 'os';
import path from 'path';
import { setTimeout } from 'timers/promises';

import express from 'express';

import { DefaultMap } from 'common/default_map';
import { Deferred } from 'common/deferred';
import type { LocalConfig } from 'common/experimentConfig';
import globals from 'common/globals/unittest';
import { LocalTrainingServiceV3 } from 'training_service/local_v3';

/**
 *  This is in fact an integration test.
 *
 *  It tests following tasks:
 *
 *   1. Create two trials concurrently.
 *   2. Create a trial that will crash.
 *   3. Create a trial and kill it.
 *
 *  As an integration test, the environment is a bit complex.
 *  It requires a temporary directory to generate trial codes,
 *  and it requires an express server to serve trials' command channel.
 *
 *  The trials' output (including stderr) can be found in "nni-experiments/unittest".
 *  This is configured by "common/globals/unittest".
 **/

describe('## training_service.local_v3 ##', () => {
    before(beforeHook);

    it('start', testStart);
    it('concurrent trials', testConcurrentTrials);
    it('failed trial', testFailedTrial);
    it('stop trial', testStopTrial);
    it('stop', testStop);

    after(afterHook);
});

/* global states */

const config: LocalConfig = {
    platform: 'local',
    trialCommand: 'not used',
    trialCodeDirectory: 'not used',
    maxTrialNumberPerGpu: 1,
    reuseMode: true,
};

// The training service.
const ts = new LocalTrainingServiceV3('test-local', config);

// Event recorders.
// The key is trial ID, and the value is resovled when the corresponding callback has been invoked.
const trialStarted: DefaultMap<string, Deferred<void>> = new DefaultMap(() => new Deferred());
const trialStopped: DefaultMap<string, Deferred<void>> = new DefaultMap(() => new Deferred());
const paramSent: DefaultMap<string, Deferred<void>> = new DefaultMap(() => new Deferred());

// Each trial's exit code.
// When the default shell is powershell, all non-zero values may become 1.
const exitCodes: Record<string, number | null> = {};

// Trial parameters to be sent.
// Each trial consumes one in order.
const parameters = [ { x: 1 }, { x: 2 }, { x: 3 }, { x: 4 } ];

// Received trial metrics.
const metrics: any[] = [];

let envId: string;

/* test cases */

async function testStart() {
    await ts.init();

    ts.onTrialStart(async (trialId, _time) => {
        trialStarted.get(trialId).resolve();
    });

    ts.onTrialEnd(async (trialId, _time, code) => {
        trialStopped.get(trialId).resolve();
        exitCodes[trialId] = code;
    });

    ts.onRequestParameter(async (trialId) => {
        ts.sendParameter(trialId, formatParameter(parameters.shift()));
        paramSent.get(trialId).resolve();
    });

    ts.onMetric(async (trialId, metric) => {
        metrics.push({ trialId, metric: JSON.parse(metric) });
    });

    const envs = await ts.start();
    assert.equal(envs.length, 1);

    envId = envs[0].id;
}

/**
 *  Run two trials concurrently.
 **/
async function testConcurrentTrials() {
    const origParamLen = parameters.length;
    metrics.length = 0;

    const trialCode = `
import nni
param = nni.get_next_parameter()
nni.report_intermediate_result(param['x'] * 0.5)
nni.report_intermediate_result(param['x'])
nni.report_final_result(param['x'])
`;
    const dir1 = await writeTrialCode('dir1', 'trial.py', trialCode);
    const dir2 = await writeTrialCode('dir2', 'trial.py', trialCode);
    await ts.uploadDirectory('dir1', dir1);
    await ts.uploadDirectory('dir2', dir2);

    const [trial1, trial2] = await Promise.all([
        ts.createTrial(envId, 'python trial.py', 'dir1', 1),
        ts.createTrial(envId, 'python trial.py', 'dir2', 2),
    ]);

    // the creation should success
    assert.notEqual(trial1, null);
    assert.notEqual(trial2, null);

    // start and stop callbacks should be invoked
    await trialStopped.get(trial1!).promise;
    assert.ok(trialStarted.get(trial1!).settled);

    await trialStopped.get(trial2!).promise;
    assert.ok(trialStarted.get(trial2!).settled);

    // exit code should be 0
    assert.equal(exitCodes[trial1!], 0, 'trial #1 exit code should be 0');
    assert.equal(exitCodes[trial2!], 0, 'trial #2 exit code should be 0');

    // each trial should consume 1 parameter and yield 3 metrics
    assert.equal(parameters.length, origParamLen - 2);
    assert.equal(metrics.length, 6);

    // verify metric value
    // because the two trials are created concurrently,
    // we don't know who gets the first parameter and who gets the second
    const metrics1 = getMetrics(trial1!);
    const metrics2 = getMetrics(trial2!);
    if (metrics1[0] === 1) {
        assert.deepEqual(metrics1, [ 1, 2, 2 ]);
        assert.deepEqual(metrics2, [ 0.5, 1, 1 ]);
    } else {
        assert.deepEqual(metrics2, [ 1, 2, 2 ]);
        assert.deepEqual(metrics1, [ 0.5, 1, 1 ]);
    }
}

/**
 *  Run a trial that exits with code 1.
 **/
async function testFailedTrial() {
    const origParamLen = parameters.length;
    metrics.length = 0;

    const trialCode = `exit(1)`;
    const dir = await writeTrialCode('dir1', 'trial_fail.py', trialCode);
    await ts.uploadDirectory('code_dir', dir);

    const trial: string = (await ts.createTrial(envId, 'python trial_fail.py', 'code_dir', 3))!;

    // despite it exit immediately, the creation should be success
    assert.notEqual(trial, null);

    // the callbacks should be invoked
    await trialStopped.get(trial).promise;
    assert.ok(trialStarted.get(trial).settled);

    // exit code should be 1
    assert.equal(exitCodes[trial], 1);

    // it should not consume parameter or yield metrics
    assert.equal(parameters.length, origParamLen);
    assert.equal(metrics.length, 0);
}

/**
 *  Create a long running trial and stop it.
 **/
async function testStopTrial() {
    const origParamLen = parameters.length;
    metrics.length = 0;

    const trialCode = `
import sys
import time
import nni
param = nni.get_next_parameter()
nni.report_intermediate_result(sys.version_info.minor)  # python 3.7 behaves differently
time.sleep(60)
nni.report_intermediate_result(param['x'])
nni.report_final_result(param['x'])
`
    const dir = await writeTrialCode('dir1', 'trial_long.py', trialCode);
    await ts.uploadDirectory('code_dir', dir);

    const trial: string = (await ts.createTrial(envId, 'python trial_long.py', 'dir1', 4))!;
    assert.notEqual(trial, null);

    // wait for it to request parameter
    await paramSent.get(trial).promise;
    // wait a while for it to report first intermediate result
    // might be longer on macOS
    await setTimeout(process.platform == 'darwin' ? 1000 : 100);  // TODO: use an env var to distinguish pipeline so we can reduce the delay
    await ts.stopTrial(trial);

    // the callbacks should be invoked
    await setTimeout(1);
    assert.ok(trialStopped.get(trial).settled);
    assert.ok(trialStarted.get(trial).settled);

    // it should consume 1 parameter and yields one metric
    assert.equal(parameters.length, origParamLen - 1);
    assert.equal(getMetrics(trial).length, 1);

    // killed trials' exit code should be null for python 3.8+
    // in 3.7 there is a bug (bpo-1054041)
    if (getMetrics(trial)[0] !== 7) {
        assert.equal(exitCodes[trial], null);
    }
}

async function testStop() {
    await ts.stop();
}

/* environment */

let tmpDir: string | null = null;
let server: Server | null = null;

async function beforeHook(): Promise<void> {
    /* create tmp dir */

    const tmpRoot = path.join(os.tmpdir(), 'nni-ut');
    await fs.mkdir(tmpRoot, { recursive: true });
    tmpDir = await fs.mkdtemp(tmpRoot + path.sep);

    /* launch rest server */

    const app = express();
    app.use('/', globals.rest.getExpressRouter());
    server = app.listen(0);
    const deferred = new Deferred<void>();
    server.on('listening', () => {
        globals.args.port = (server!.address() as any).port;
        deferred.resolve();
    });
    await deferred.promise;
}

async function afterHook() {
    if (tmpDir !== null) {
        try { await fs.rm(tmpDir, { force: true, recursive: true }) } catch { };
    }

    if (server !== null) {
        const deferred = new Deferred<void>();
        server.close(() => { deferred.resolve(); });
        await deferred.promise;
    }

    globals.reset();
}

/* helpers */

async function writeTrialCode(dir: string, file: string, content: string): Promise<string> {
    await fs.mkdir(path.join(tmpDir!, dir), { recursive: true });
    await fs.writeFile(path.join(tmpDir!, dir, file), content);
    return path.join(tmpDir!, dir);
}

// FIXME: parameter / metric formatting should be more structural so it does not need helpers here

function formatParameter(param: any) {
    return JSON.stringify({
        parameter_id: param.x,
        parameters: param,
    });
}

function getMetrics(trialId: string): number[] {
    return metrics.filter(metric => (metric.trialId === trialId)).map(metric => JSON.parse(metric.metric.value)) as any;
}
