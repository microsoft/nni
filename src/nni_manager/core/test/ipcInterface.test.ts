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
import { cleanupUnitTest, prepareUnitTest } from '../../common/utils';
import * as CommandType from '../commands';
import { createDispatcherInterface, IpcInterface } from '../ipcInterface';
import { NNIError } from '../../common/errors';

let sentCommands: { [key: string]: string }[] = [];
const receivedCommands: { [key: string]: string }[] = [];

let commandTooLong: Error | undefined;
let rejectCommandType: Error | undefined;

function runProcess(): Promise<Error | null> {
    // the process is intended to throw error, do not reject
    const deferred: Deferred<Error | null> = new Deferred<Error | null>();

    // create fake assessor process
    const stdio: StdioOptions = ['ignore', 'pipe', process.stderr, 'pipe', 'pipe'];
    const proc: ChildProcess = spawn('python3 assessor.py', [], { stdio, cwd: 'core/test', shell: true });

    // record its sent/received commands on exit
    proc.on('error', (error: Error): void => { deferred.resolve(error); });
    proc.on('exit', (code: number): void => {
        if (code !== 0) {
            deferred.resolve(new Error(`return code: ${code}`));
        } else {
            sentCommands = proc.stdout.read().toString().split('\n');
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

    // Command #3: too long
    try {
        dispatcher.sendCommand('ME', 'x'.repeat(1_000_000));
    } catch (error) {
        commandTooLong = error;
    }

    // Command #4: FE is not tuner/assessor command, test the exception type of send non-valid command 
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
        assert.equal(sentCommands[0], '(\'IN\', \'\')');
    });

    it('sendCommand() should work with content', (): void => {
        assert.equal(sentCommands[1], '(\'ME\', \'123\')');
    });

    it('sendCommand() should throw on too long command', (): void => {
        if (commandTooLong === undefined) {
            assert.fail('Should throw error')
        } else {
            const err: Error | undefined = (<NNIError>commandTooLong).cause;
            assert(err && err.name === 'RangeError');
            assert(err && err.message === 'Command too long');
        }
    });

    it('sendCommand() should throw on wrong command type', (): void => {
        assert.equal((<Error>rejectCommandType).name, 'AssertionError [ERR_ASSERTION]');
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
