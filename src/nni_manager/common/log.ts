// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';
/* tslint:disable:no-any */

import * as fs from 'fs';
import * as path from 'path';
import { Writable } from 'stream';
import { WritableStreamBuffer } from 'stream-buffers';
import { format } from 'util';
import * as component from '../common/component';
import { getExperimentStartupInfo, isReadonly } from './experimentStartupInfo';
import { getLogDir } from './utils';

const FATAL = 1;
const ERROR = 2;
const WARNING = 3;
const INFO = 4;
const DEBUG = 5;
const TRACE = 6;

const logLevelNameMap: Map<string, number> = new Map([['fatal', FATAL],
    ['error', ERROR], ['warning', WARNING], ['info', INFO], ['debug', DEBUG], ['trace', TRACE]]);

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
    private level: number = INFO;
    private bufferSerialEmitter: BufferSerialEmitter;
    private writable: Writable;
    private readonly = false;

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

        const logLevelName: string = getExperimentStartupInfo()
                                    .getLogLevel();
        const logLevel: number | undefined = logLevelNameMap.get(logLevelName);
        if (logLevel !== undefined) {
            this.level = logLevel;
        }

        this.readonly = isReadonly();
    }

    public close() {
        this.writable.destroy();
    }

    public trace(...param: any[]): void {
        if (this.level >= TRACE) {
            this.log('TRACE', param);
        }
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

    public fatal(...param: any[]): void {
        this.log('FATAL', param);
    }
    
    /**
     * if the experiment is not in readonly mode, write log content to stream
     * @param level log level
     * @param param the params to be written
     */
    private log(level: string, param: any[]): void {
        if (!this.readonly) {
            const buffer: WritableStreamBuffer = new WritableStreamBuffer();
            buffer.write(`[${(new Date()).toLocaleString()}] ${level} `);
            buffer.write(format(param));
            buffer.write('\n');
            buffer.end();
            this.bufferSerialEmitter.feed(buffer.getContents());
        }
    }
}

function getLogger(): Logger {
    return component.get(Logger);
}

export { Logger, getLogger, logLevelNameMap };
