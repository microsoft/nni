// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import { ChildProcess, spawn, StdioOptions } from 'child_process';
import { Deferred } from 'ts-deferred';
import { cleanupUnitTest, prepareUnitTest, getTunerProc, getCmdPy } from '../../common/utils';
import * as CommandType from '../commands';
import { createDispatcherInterface, IpcInterface } from '../ipcInterface';
import { NNIError } from '../../common/errors';

let sentCommands: { [key: string]: string }[] = [];
const receivedCommands: { [key: string]: string }[] = [];

let rejectCommandType: Error | undefined;

function runProcess(): Promise<Error | null> {
    // the process is intended to throw error, do not reject
    const deferred: Deferred<Error | null> = new Deferred<Error | null>();

    // create fake assessor process
    const stdio: StdioOptions = ['ignore', 'pipe', process.stderr, 'pipe', 'pipe'];
    const command: string = getCmdPy() + ' assessor.py';
    const proc: ChildProcess = getTunerProc(command, stdio,  'core/test', process.env);
    // record its sent/received commands on exit
    proc.on('error', (error: Error): void => { deferred.resolve(error); });
    proc.on('exit', (code: number): void => {
        if (code !== 0) {
            deferred.resolve(new Error(`return code: ${code}`));
        } else {
            let str = proc.stdout.read().toString();
            if(str.search("\r\n")!=-1){
                sentCommands = str.split("\r\n");
            }
            else{
                sentCommands = str.split('\n');
            }
            deferred.resolve(null);
        }
    });

    // create IPC interface
    const dispatcher: IpcInterface = createDispatcherInterface(proc);
    dispatcher.onCommand((commandType: string, content: string): void => {
        receivedCommands.push({ commandType, content });
    });

    // Command #1: ok
    dispatcher.sendCommand('IN');

    // Command #2: ok
    dispatcher.sendCommand('ME', '123');

    // Command #3: FE is not tuner/assessor command, test the exception type of send non-valid command
    try {
        dispatcher.sendCommand('FE', '1');
    } catch (error) {
        rejectCommandType = error;
    }

    return deferred.promise;
}

describe('core/protocol', (): void => {

    before(async () => {
        prepareUnitTest();
        await runProcess();
    });

    after(() => {
        cleanupUnitTest();
    });

    it('should have sent 2 successful commands', (): void => {
        assert.equal(sentCommands.length, 3);
        assert.equal(sentCommands[2], '');
    });

    it('sendCommand() should work without content', (): void => {
        assert.equal(sentCommands[0], "('IN', '')");
    });

    it('sendCommand() should work with content', (): void => {
        assert.equal(sentCommands[1], "('ME', '123')");
    });

    it('sendCommand() should throw on wrong command type', (): void => {
        assert.equal((<Error>rejectCommandType).name.split(' ')[0], 'AssertionError');
    });

    it('should have received 3 commands', (): void => {
        assert.equal(receivedCommands.length, 3);
    });

    it('onCommand() should work without content', (): void => {
        assert.deepStrictEqual(receivedCommands[0], {
            commandType: 'KI',
            content: ''
        });
    });

    it('onCommand() should work with content', (): void => {
        assert.deepStrictEqual(receivedCommands[1], {
            commandType: 'KI',
            content: 'hello'
        });
    });

    it('onCommand() should work with Unicode content', (): void => {
        assert.deepStrictEqual(receivedCommands[2], {
            commandType: 'KI',
            content: '世界'
        });
    });

});
