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

import * as assert from 'assert';
import { ChildProcess, spawn, StdioOptions } from 'child_process';
import { Deferred } from 'ts-deferred';
import { cleanupUnitTest, prepareUnitTest, getMsgDispatcherCommand, getTunerProc } from '../../common/utils';
import * as CommandType from '../commands';
import { createDispatcherInterface, IpcInterface } from '../ipcInterface';

let dispatcher: IpcInterface | undefined;
let procExit: boolean = false;
let procError: boolean = false;

function startProcess(): void {
    // create fake assessor process
    const stdio: StdioOptions = ['ignore', 'pipe', process.stderr, 'pipe', 'pipe'];

    const dispatcherCmd: string = getMsgDispatcherCommand(
        // Mock tuner config
        {
            className: 'DummyTuner',
            codeDir: './',
            classFileName: 'dummy_tuner.py'
        },
        // Mock assessor config
        {
            className: 'DummyAssessor',
            codeDir: './',
            classFileName: 'dummy_assessor.py'
        },
        // advisor
        undefined
    );
    const proc: ChildProcess = getTunerProc(dispatcherCmd, stdio,  'core/test', process.env);
    proc.on('error', (error: Error): void => {
        procExit = true;
        procError = true;
    });
    proc.on('exit', (code: number): void => {
        procExit = true;
        procError = (code !== 0);
    });

    // create IPC interface
    dispatcher = createDispatcherInterface(proc);
    (<IpcInterface>dispatcher).onCommand((commandType: string, content: string): void => {
        console.log(commandType, content);  // tslint:disable-line:no-console
    });
}

describe('core/ipcInterface.terminate', (): void => {
    before(() => {
        prepareUnitTest();
        startProcess();
    });

    after(() => {
        cleanupUnitTest();
    });

    it('normal', () => {
        (<IpcInterface>dispatcher).sendCommand(
            CommandType.REPORT_METRIC_DATA,
            '{"trial_job_id":"A","type":"PERIODICAL","value":1,"sequence":123}');

        const deferred: Deferred<void> = new Deferred<void>();
        setTimeout(
            () => {
                assert.ok(!procExit);
                assert.ok(!procError);
                deferred.resolve();
            },
            1000);

        return deferred.promise;
    });

    it('terminate', () => {
        (<IpcInterface>dispatcher).sendCommand(CommandType.TERMINATE);

        const deferred: Deferred<void> = new Deferred<void>();
        setTimeout(
            () => {
                assert.ok(procExit);
                assert.ok(!procError);
                deferred.resolve();
            },
            5000);

        return deferred.promise;
    });
});
