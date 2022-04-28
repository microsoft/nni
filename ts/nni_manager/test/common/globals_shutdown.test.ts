// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import { setTimeout } from 'timers/promises';

import { Deferred } from 'ts-deferred';

import 'common/globals/unittest';
import { ShutdownManager, UnitTestHelpers } from 'common/globals/shutdown';

/* environment */

UnitTestHelpers.setShutdownTimeout(10);

let shutdown: ShutdownManager = new ShutdownManager();
let callbackCount: number[] = [ 0, 0 ];
let exitCode: number | null = null;

/* test cases */

// Test a normal shutdown.
// Each callback should be invoked once.
async function testShutdown(): Promise<void> {
    shutdown.register('ModuleA', async () => { callbackCount[0] += 1; });
    shutdown.register('ModuleB', async () => { callbackCount[1] += 1; });

    shutdown.initiate('unittest');

    await setTimeout(10);
    assert.deepEqual(callbackCount, [ 1, 1 ]);
    assert.equal(exitCode, 0);
}

// Test a shutdown caused by critical error.
// The faulty module's callback will not be invoked by design.
async function testError(): Promise<void> {
    shutdown.notifyInitializeComplete();
    shutdown.register('ModuleA', async () => { callbackCount[0] += 1; });
    shutdown.register('ModuleB', async () => { callbackCount[1] += 1; });

    shutdown.criticalError('ModuleA', new Error('test critical error'));

    await setTimeout(10);
    assert.deepEqual(callbackCount, [ 0, 1 ]);
    assert.equal(exitCode, 1);
}

// Test a shutdown caused by critical error in initializing phase.
// Current implementation does not invoke callbacks in this case, so the timeout is 0.
// If you have modified shutdown logic and this case failed, check the timeout.
async function testInitError(): Promise<void> {
    shutdown.register('ModuleA', async () => { callbackCount[0] += 1; });

    shutdown.criticalError('ModuleA', new Error('test init error'));

    await setTimeout();
    assert.equal(exitCode, 1);
}

// Simulate an error inside shutdown callback.
async function testCallbackError(): Promise<void> {
    shutdown.notifyInitializeComplete();
    shutdown.register('ModuleA', async () => { callbackCount[0] += 1; });
    shutdown.register('ModuleB', async () => {
        callbackCount[1] += 1;
        throw new Error('Module B callback error');
    });

    shutdown.initiate('unittest');

    await setTimeout(10);
    assert.deepEqual(callbackCount, [ 1, 1 ]);
    assert.equal(exitCode, 1);
}

// Simulate unresponsive shutdown callback.
// Pay attention that timeout handler does not explicitly cancel shutdown callback
// because in real world it terminates the process.
// But in mocked environment process.exit() is overwritten so the callback will eventually finish,
// and it can cause another process.exit().
// Make sure not to recover mocked process.exit() before the callback finish.
async function testTimeout(): Promise<void> {
    const deferred = new Deferred<void>();

    shutdown.register('ModuleA', async () => { callbackCount[0] += 1; });
    shutdown.register('ModuleB', async () => {
        await setTimeout(30);  // we have set timeout to 10 ms so this times out
        callbackCount[1] += 1;
        deferred.resolve();
    });

    shutdown.initiate('unittest');

    await setTimeout(20);
    assert.deepEqual(callbackCount, [ 1, 0 ]);
    assert.equal(exitCode, 1);

    // if we don't await, process.exit() will be recovered and it will terminate testing.
    await deferred.promise;
}

/* register */

describe('## globals.shutdown ##', () => {
    before(beforeHook);
    beforeEach(beforeEachHook);

    it('normal', testShutdown);
    it('on error', testError);
    it('on init fail', testInitError);
    it('callback raise error', testCallbackError);
    it('timeout', testTimeout);

    after(afterHook);
});

/* hooks */

const origProcessExit = process.exit;

function beforeHook() {
    process.exit = ((code: number) => { exitCode = code; }) as any;
}

function beforeEachHook() {
    shutdown = new ShutdownManager();
    callbackCount = [ 0, 0 ];
    exitCode = null;
}

function afterHook() {
    process.exit = origProcessExit;
}
