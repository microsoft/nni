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
/* tslint:disable:no-any */

import * as fs from 'fs';
import * as path from 'path';
import { Writable } from 'stream';
import { WritableStreamBuffer } from 'stream-buffers';
import { format } from 'util';
import * as component from '../common/component';
import { getLogDir } from './utils';

const CRITICAL: number = 1;
const ERROR: number = 2;
const WARNING: number = 3;
const INFO: number = 4;
const DEBUG: number = 5;

class BufferSerialEmitter {
    private buffer: Buffer;
    private emitting: boolean;
    private writable: Writable;

    constructor(writable: Writable) {
        this.buffer = Buffer.alloc(0);
        this.emitting = false;
        this.writable = writable;
    }

    public feed(buffer: Buffer): void {
        this.buffer = Buffer.concat([this.buffer, buffer]);
        if (!this.emitting) {
            this.emit();
        }
    }

    private emit(): void {
        this.emitting = true;
        this.writable.write(this.buffer, () => {
            if (this.buffer.length === 0) {
                this.emitting = false;
            } else {
                this.emit();
            }
        });
        this.buffer = Buffer.alloc(0);
    }
}

@component.Singleton
class Logger {
    private DEFAULT_LOGFILE: string = path.join(getLogDir(), 'nnimanager.log');
    private level: number = DEBUG;
    private bufferSerialEmitter: BufferSerialEmitter;
    private writable: Writable;

    constructor(fileName?: string) {
        let logFile: string | undefined = fileName;
        if (logFile === undefined) {
            logFile = this.DEFAULT_LOGFILE;
        }
        this.writable = fs.createWriteStream(logFile, {
            flags: 'a+',
            encoding: 'utf8',
            autoClose: true
        });
        this.bufferSerialEmitter = new BufferSerialEmitter(this.writable);
    }

    public close() {
        this.writable.destroy();
    }

    public debug(...param: any[]): void {
        if (this.level >= DEBUG) {
            this.log('DEBUG', param);
        }
    }

    public info(...param: any[]): void {
        if (this.level >= INFO) {
            this.log('INFO', param);
        }
    }

    public warning(...param: any[]): void {
        if (this.level >= WARNING) {
            this.log('WARNING', param);
        }
    }

    public error(...param: any[]): void {
        if (this.level >= ERROR) {
            this.log('ERROR', param);
        }
    }

    public critical(...param: any[]): void {
        this.log('CRITICAL', param);
    }

    private log(level: string, param: any[]): void {
        const buffer: WritableStreamBuffer = new WritableStreamBuffer();
        buffer.write(`[${(new Date()).toISOString()}] ${level} `);
        buffer.write(format(null, param));
        buffer.write('\n');
        buffer.end();
        this.bufferSerialEmitter.feed(buffer.getContents());
    }
}

function getLogger(fileName?: string): Logger {
    component.Container.bind(Logger).provider({
        get: (): Logger => new Logger(fileName)
    });

    return component.get(Logger);
}

export { Logger, getLogger };
