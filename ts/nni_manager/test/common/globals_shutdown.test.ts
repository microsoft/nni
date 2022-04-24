// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert/strict';
import { setTimeout } from 'timers/promises';

import { Deferred } from 'ts-deferred';

import 'common/globals/unittest';
import { ShutdownManager, UnitTestHelpers } from 'common/globals/shutdown';

UnitTestHelpers.setShutdownTimeout(10);

let shutdown: ShutdownManager = new ShutdownManager();
let callbackCount: number[] = [ 0, 0 ];
let exitCode: number | null = null;

async function testShutdown(): Promise<void> {
    shutdown.register('ModuleA', async () => { callbackCount[0] += 1; });
    shutdown.register('ModuleB', async () => { callbackCount[1] += 1; });

    shutdown.initiate('unittest');

    await setTimeout(10);
    assert.deepEqual(callbackCount, [ 1, 1 ]);
    assert.equal(exitCode, 0);
}

async function testError(): Promise<void> {
    shutdown.notifyInitializeComplete();
    shutdown.register('ModuleA', async () => { callbackCount[0] += 1; });
    shutdown.register('ModuleB', async () => { callbackCount[1] += 1; });

    shutdown.criticalError('ModuleA', new Error('test critical error'));

    await setTimeout(10);
    assert.deepEqual(callbackCount, [ 0, 1 ]);
    assert.equal(exitCode, 1);
}

async function testInitError(): Promise<void> {
    shutdown.register('ModuleA', async () => { callbackCount[0] += 1; });

    shutdown.criticalError('ModuleA', new Error('test init error'));

    await setTimeout();
    assert.equal(exitCode, 1);
}

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

    await deferred.promise;
}

describe('## globals.shutdown ##', () => {
    before(beforeHook);
    afterEach(afterEachHook);

    it('normal', testShutdown);
    it('on error', testError);
    it('on init fail', testInitError);
    it('callback raise error', testCallbackError);
    it('timeout', testTimeout);

    after(afterHook);
});

const origProcessExit = process.exit;

function beforeHook() {
    process.exit = ((code: number) => { exitCode = code; }) as any;
}

function afterEachHook() {
    shutdown = new ShutdownManager();
    callbackCount = [ 0, 0 ];
    exitCode = null;
}

function afterHook() {
    process.exit = ((code?: number) => {
        console.error('@@ process.exit');
        console.error(new Error().stack);
        origProcessExit(code);
    }) as any;
}
