// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  The underlying IO stream of loggers.
 *
 *  Normal modules should not use this directly. Use "common/log.ts" instead.
 **/

import fs from 'fs';
import { setTimeout } from 'timers/promises';
import util from 'util';

import type { NniManagerArgs } from './arguments';
import type { NniPaths } from './paths';

export interface LogStream {
    writeLine(line: string): void;
    writeLineSync(line: string): void;
    close(): Promise<void>;
}

const writePromise = util.promisify(fs.write);

class LogStreamImpl implements LogStream {
    private buffer: string[] = [];
    private flushing: boolean = false;
    private logFileFd: number;
    private toConsole: boolean;

    constructor(logFile: string, toConsole: boolean) {
        this.logFileFd = fs.openSync(logFile, 'a');
        this.toConsole = toConsole;
    }

    public writeLine(line: string): void {
        this.buffer.push(line);
        this.flush();
    }

    public writeLineSync(line: string): void {
        if (this.toConsole) {
            console.log(line);
        }
        fs.writeSync(this.logFileFd, line + '\n');
    }

    public async close(): Promise<void> {
        while (this.flushing) {
            await setTimeout();
        }
        fs.closeSync(this.logFileFd);
        this.logFileFd = 2;  // stderr
        this.toConsole = false;
    }

    private async flush(): Promise<void> {
        if (this.flushing) {
            return;
        }
        this.flushing = true;
        while (this.buffer.length > 0) {
            const lines = this.buffer.join('\n');
            this.buffer.length = 0;
            if (this.toConsole) {
                console.log(lines);
            }
            await writePromise(this.logFileFd, lines + '\n');
        }
        this.flushing = false;
    }
}

export function initLogStream(args: NniManagerArgs, paths: NniPaths): LogStream {
    return new LogStreamImpl(paths.nniManagerLog, args.foreground);
}

export function initLogStreamCustom(args: NniManagerArgs, path: string): LogStream {
    return new LogStreamImpl(path, args.foreground);
}
