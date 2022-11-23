// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  This is in fact an integration test.
 **/

import assert from 'assert/strict';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { setTimeout } from 'timers/promises';

import { DefaultMap } from 'common/default_map';
import { Deferred } from 'common/deferred';
import type { LocalConfig } from 'common/experimentConfig';
import { LocalTrainingServiceV3 } from 'training_service/local_v3';

describe('## local v3 ##', () => {
    before(beforeHook);

    it('integration test', () => testAll());
});

const config: LocalConfig = {
    platform: 'local',
    trialCommand: 'not used',
    trialCodeDirectory: 'not used',
    maxTrialNumberPerGpu: 1,
    reuseMode: true,
};

const parameters = [ { x: 1 }, { x: 2 }, { x: 3 }, { x: 4 } ];

const metrics: any[] = [];

const trialStarted = new DefaultMap<string, Deferred<void>>(() => new Deferred());
const trialStopped = new DefaultMap<string, Deferred<void>>(() => new Deferred());
const exitCodes: Record<string, number | null> = {};

let tmpDir: string;

async function beforeHook(): Promise<void> {
    const tmpRoot = path.join(os.tmpdir(), 'nni-ut');
    await fs.mkdir(tmpRoot, { recursive: true });
    tmpDir = await fs.mkdtemp(tmpRoot + path.sep);
}

async function testAll() {
    const origParamLen = parameters.length;

    /* init */

    const ts = new LocalTrainingServiceV3('test-local', config);
    await ts.init();

    ts.onTrialStart(async (trialId, _time) => { trialStarted.get(trialId).resolve(); });
    ts.onTrialEnd(async (trialId, _time, code) => {
        trialStopped.get(trialId).resolve();
        exitCodes[trialId] = code;
    });
    ts.onRequestParameter(async (trialId) => { ts.sendParameter(trialId, JSON.stringify(parameters.shift())); });
    ts.onMetric(async (trialId, metric) => { metrics.push({ trialId, metric: JSON.parse(metric) }); });

    const envs = await ts.start();
    assert.equal(envs.length, 1);
    const envId = envs[0].id;

    /* prepare code dir */

    const normalTrialCode = `
import nni
param = nni.get_next_parameter()
nni.report_intermediate_result(param['x'] * 0.5)
nni.report_final_result(param['x'])
`;

    const failTrialCode = `
import nni
exit(1)
`

    const longTrialCode = `
import nni
param = nni.get_next_parameter()
nni.report_intermediate_result(param['x'] * 0.5)
while True:
    pass
`

    await fs.mkdir(path.join(tmpDir, 'dir1'));
    await fs.writeFile(path.join(tmpDir, 'dir1', 'trial.py'), normalTrialCode);
    await fs.writeFile(path.join(tmpDir, 'dir1', 'trial_fail.py'), failTrialCode);
    await fs.writeFile(path.join(tmpDir, 'dir1', 'trial_long.py'), longTrialCode);

    await fs.mkdir(path.join(tmpDir, 'dir2'));
    await fs.writeFile(path.join(tmpDir, 'dir2', 'trial.py'), normalTrialCode);

    await ts.uploadDirectory('dir1', path.join(tmpDir, 'dir1'));
    await ts.uploadDirectory('dir2', path.join(tmpDir, 'dir2'));

    /* two concurrent normal trials */

    const [trial1, trial2] = await Promise.all([
        ts.createTrial(envId, 'python trial.py', 'dir1'),
        ts.createTrial(envId, 'python trial.py', 'dir2'),
    ]);

    assert.notEqual(trial1, null);
    assert.notEqual(trial2, null);

    await trialStopped.get(trial1!).promise;
    await trialStopped.get(trial2!).promise;

    assert.equal(exitCodes[trial1!], 0);
    assert.equal(exitCodes[trial2!], 0);

    assert.equal(parameters.length, origParamLen - 2);
    console.log(metrics);

    /* a failed trial */

    const trial3: string = (await ts.createTrial(envId, 'python trial_fail.py', 'dir1'))!;
    assert.notEqual(trial3, null);
    await trialStopped.get(trial3).promise;
    assert.equal(parameters.length, origParamLen - 2);
    //assert.equal(metrics.length, 2);
    assert.equal(exitCodes[trial3], 1);

    /* kill a long trial */

    const trial4: string = (await ts.createTrial(envId, 'python trial_long.py', 'dir2'))!;
    assert.notEqual(trial4, null);
    await trialStarted.get(trial4).promise;
    await setTimeout(100);
    assert.equal(parameters.length, origParamLen - 3);
    // metrics
    await ts.stopTrial(trial4);
    await trialStopped.get(trial4).promise;
}
