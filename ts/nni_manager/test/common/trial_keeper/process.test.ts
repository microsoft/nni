// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { setTimeout } from 'timers/promises';

import { Deferred } from 'common/deferred';
import { TrialProcess, TrialProcessOptions } from 'common/trial_keeper/process';

describe('## trial keeper : trial process ##', () => {
    before(beforeHook);

    it('normal', () => testNormal());
    it('graceful kill', () => testGracefulKill());
    it('force kill', () => testForceKill());

    //it('kill process tree', () => testKillTree());

    after(afterHook);
});

/**
 *  Run a simple trial and check that:
 *
 *    - The trial is successfully launched;
 *    - The trial can read correct environment variable;
 *    - Trial keeper can read the trial's output;
 *    - Trial keeper can receive the trial's exit code;
 *    - The callbacks are invoked in correct order.
 **/
async function testNormal() {
    options.command = `python -c "import os ; print(os.environ['NNI_TRIAL_JOB_ID']) ; exit(2)"`;

    const started = new Deferred<void>();
    let reportedStartTime: any = null;
    let realStartTime: any = null;

    const stopped = new Deferred<void>();
    let reportedStopTime: any = null;
    let realStopTime: any = null;

    let exitCode: any = null;

    const proc = new TrialProcess('test_id1');
    proc.onStart(time => {
        started.resolve();
        reportedStartTime = time;
        realStartTime = Date.now();
    });
    proc.onStop((time, code, _signal) => {
        stopped.resolve();
        reportedStopTime = time;
        realStopTime = Date.now();
        exitCode = code;
    });

    const success = await proc.spawn(options);
    assert.ok(success);

    await started.promise;
    await stopped.promise;

    if (process.platform === 'win32') {
        assert.notEqual(exitCode, 0);
    } else {
        assert.equal(exitCode, 2);
    }

    assert.ok(reportedStartTime < reportedStopTime);
    assert.ok(realStartTime < realStopTime);
    assert.ok(realStartTime - reportedStartTime < 1000);
    assert.ok(realStopTime - reportedStopTime < 1000);

    // FIXME:
    // In fact the output files are not guaranteed to be ready now.
    // Maybe it needs an `await proc.done()` API for log collection.

    const stdout = await fs.readFile(path.join(tmpDir, 'trial.stdout'), { encoding: 'utf8' });
    const stderr = await fs.readFile(path.join(tmpDir, 'trial.stderr'), { encoding: 'utf8' });
    assert.equal(stdout.trim(), 'test_id1');
    assert.equal(stderr.trim(), '');
}

async function testGracefulKill(): Promise<void> {
    const script = `
import sys
try:
    while True:
        pass
except KeyboardInterrupt:
    print('graceful exit', file=sys.stderr, flush=True)
`
    const stderr = (process.platform === 'win32' ? undefined : 'graceful exit');
    await testKillHelper('test_grace', script, '', stderr);
}

async function testForceKill(): Promise<void> {
    // this script will never exit on SIGINT
    const script = `
import signal
signal.signal(signal.SIGINT, lambda *args: print('handler invoked', flush=True))
while True:
    pass
`
    const stdout = (process.platform === 'win32' ? undefined : 'handler invoked');
    await testKillHelper('test_force', script, stdout, '');
}

async function testKillHelper(trialId: string, script: string, stdout?: string, stderr?: string): Promise<void> {
    const timeout = (process.platform === 'darwin' ? 200 : 50);

    await fs.writeFile(path.join(tmpDir, `trial_${trialId}.py`), script);
    options.command = `python trial_${trialId}.py`;

    const started = new Deferred<void>();
    const stopped = new Deferred<void>();

    const proc = new TrialProcess(trialId);
    proc.onStart(_time => { started.resolve(); });
    proc.onStop((_time, _code, _signal) => { stopped.resolve(); });

    const success = await proc.spawn(options);
    assert.ok(success);

    await started.promise;
    await setTimeout(timeout);  // the python script needs some time to reach "while True"
    assert.ok(!stopped.settled);

    await proc.kill(timeout);
    await stopped.promise;
    assert.ok(stopped.settled);

    const out = await fs.readFile(path.join(tmpDir, 'trial.stdout'), { encoding: 'utf8' });
    const err = await fs.readFile(path.join(tmpDir, 'trial.stderr'), { encoding: 'utf8' });
    if (stdout !== undefined) {
        assert.equal(out.trim(), stdout.trim());
    }
    if (stderr !== undefined) {
        assert.equal(err.trim(), stderr.trim());
    }
}

async function testKillTree(): Promise<void> {
    const script = `
from subprocess import Popen
proc = Popen(['python', '-c', 'import time ; time.sleep(600)'])
print(proc.pid, flush=True)
while True:
    pass
`

    await fs.writeFile(path.join(tmpDir, `trial_test_tree.py`), script);
    options.command = `python trial_test_tree.py`;

    const proc = new TrialProcess('test_tree');
    await proc.spawn(options);

    await setTimeout(100);
    const stdout = await fs.readFile(path.join(tmpDir, 'trial.stdout'), { encoding: 'utf8' });
    const pid = Number(stdout.trim());

    assert.ok(isAlive(pid));

    await proc.kill(100);
    const stopped = new Deferred<void>();
    proc.onStop((_time, _code, _signal) => { stopped.resolve(); });
    await stopped.promise;

    await setTimeout(100);
    assert.ok(!isAlive(pid));
}

function isAlive(pid: number): boolean {
    try {
        process.kill(pid, 0);
        return true;
    } catch (_e) {
        return false;
    }
}

/* environment */

let tmpDir: string;

const options: TrialProcessOptions = {
    command: '',
    codeDirectory: '',
    outputDirectory: '',
    commandChannelUrl: '',
    platform: 'ut',
    environmentVariables: {},
};

async function beforeHook(): Promise<void> {
    const tmpRoot = path.join(os.tmpdir(), 'nni-ut');
    await fs.mkdir(tmpRoot, { recursive: true });
    tmpDir = await fs.mkdtemp(tmpRoot + path.sep);
    options.codeDirectory = tmpDir;
    options.outputDirectory = tmpDir;
}

async function afterHook(): Promise<void> {
    fs.rm(tmpDir, { force: true, recursive: true });
}
