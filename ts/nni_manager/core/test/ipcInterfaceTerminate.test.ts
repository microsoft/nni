// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

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
            experimentName: 'exp1',
            maxExecDuration: 3600,
            searchSpace: '',
            trainingServicePlatform: 'local',
            authorName: '',
            trialConcurrency: 1,
            maxTrialNum: 5,
            tuner: {
                className: 'DummyTuner',
                codeDir: './',
                classFileName: 'dummy_tuner.py',
                checkpointDir: './'
            },
            assessor: {
                className: 'DummyAssessor',
                codeDir: './',
                classFileName: 'dummy_assessor.py',
                checkpointDir: './'
            }
        }
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
        console.log(commandType, content);
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
            10000);

        return deferred.promise;
    });
});
