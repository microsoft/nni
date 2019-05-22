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
import { ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import { Readable, Writable } from 'stream';
import { NNIError } from '../common/errors';
import { getLogger, Logger } from '../common/log';
import { getLogDir } from '../common/utils';
import * as CommandType from './commands';

const ipcOutgoingFd: number = 3;
const ipcIncomingFd: number = 4;

/**
 * Encode a command
 * @param commandType a command type defined in 'core/commands'
 * @param content payload of the command
 * @returns binary command data
 */
function encodeCommand(commandType: string, content: string): Buffer {
    const contentBuffer: Buffer = Buffer.from(content);
    if (contentBuffer.length >= 1_000_000) {
        throw new RangeError('Command too long');
    }
    const contentLengthBuffer: Buffer = Buffer.from(contentBuffer.length.toString().padStart(6, '0'));

    return Buffer.concat([Buffer.from(commandType), contentLengthBuffer, contentBuffer]);
}

/**
 * Decode a command
 * @param Buffer binary incoming data
 * @returns a tuple of (success, commandType, content, remain)
 *          success: true if the buffer contains at least one complete command; otherwise false
 *          remain: remaining data after the first command
 */
function decodeCommand(data: Buffer): [boolean, string, string, Buffer] {
    if (data.length < 8) {
        return [false, '', '', data];
    }
    const commandType: string = data.slice(0, 2).toString();
    const contentLength: number = parseInt(data.slice(2, 8).toString(), 10);
    if (data.length < contentLength + 8) {
        return [false, '', '', data];
    }
    const content: string = data.slice(8, contentLength + 8).toString();
    const remain: Buffer = data.slice(contentLength + 8);

    return [true, commandType, content, remain];
}

class IpcInterface {
    private acceptCommandTypes: Set<string>;
    private outgoingStream: Writable;
    private incomingStream: Readable;
    private eventEmitter: EventEmitter;
    private readBuffer: Buffer;
    private logger: Logger = getLogger();

    /**
     * Construct a IPC proxy
     * @param proc the process to wrap
     * @param acceptCommandTypes set of accepted commands for this process
     */
    constructor(proc: ChildProcess, acceptCommandTypes: Set<string>) {
        this.acceptCommandTypes = acceptCommandTypes;
        this.outgoingStream = <Writable>proc.stdio[ipcOutgoingFd];
        this.incomingStream = <Readable>proc.stdio[ipcIncomingFd];
        this.eventEmitter = new EventEmitter();
        this.readBuffer = Buffer.alloc(0);

        this.incomingStream.on('data', (data: Buffer) => { this.receive(data); });
    }

    /**
     * Send a command to process
     * @param commandType: a command type defined in 'core/commands'
     * @param content: payload of command
     */
    public sendCommand(commandType: string, content: string = ''): void {
        this.logger.debug(`ipcInterface command type: [${commandType}], content:[${content}]`);
        assert.ok(this.acceptCommandTypes.has(commandType));

        try {
            const data: Buffer = encodeCommand(commandType, content);
            if (!this.outgoingStream.write(data)) {
                this.logger.warning('Commands jammed in buffer!');
            }
        } catch (err) {
            throw NNIError.FromError(
                err,
                `Dispatcher Error, please check this dispatcher log file for more detailed information: ${getLogDir()}/dispatcher.log . `
            );
        }
    }

    /**
     * Add a command listener
     * @param listener the listener callback
     */
    public onCommand(listener: (commandType: string, content: string) => void): void {
        this.eventEmitter.on('command', listener);
    }

    /**
     * Deal with incoming data from process
     * Invoke listeners for each complete command received, save incomplete command to buffer
     * @param data binary incoming data
     */
    private receive(data: Buffer): void {
        this.readBuffer = Buffer.concat([this.readBuffer, data]);
        while (this.readBuffer.length > 0) {
            const [success, commandType, content, remain] = decodeCommand(this.readBuffer);
            if (!success) {
                break;
            }
            assert.ok(this.acceptCommandTypes.has(commandType));
            this.eventEmitter.emit('command', commandType, content);
            this.readBuffer = remain;
        }
    }
}

/**
 * Create IPC proxy for tuner process
 * @param process_ the tuner process
 */
function createDispatcherInterface(process: ChildProcess): IpcInterface {
    return new IpcInterface(process, new Set([...CommandType.TUNER_COMMANDS, ...CommandType.ASSESSOR_COMMANDS]));
}

export { IpcInterface, createDispatcherInterface };
