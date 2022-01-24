// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  The underlying stream of loggers.
 *
 *  This module should only be used by "common/log.ts".
 **/

import fs from 'fs';

import type { NniManagerArgs, NniPaths } from './index';

export class LogStream {
    private isOpen: boolean = false;
    private logFileFd!: number;
    private logFilePath: string;
    private toConsole: boolean;

    constructor(args: NniManagerArgs, paths: NniPaths) {
        this.logFilePath = paths.nniManagerLog;
        this.toConsole = args.foreground;
        this.open();
    }

    public writeLine(line: string): void {
        if (this.isOpen) {
            // use writeSync() because the doc says it is unsafe to write() without waiting resolved
            // hopefully this will not cause performance issue, or we will need to do buffer ourself
            // createWriteStream() is buffered and cannot manually flush, so not good for logging
            fs.writeSync(this.logFileFd, line + '\n');
            if (this.toConsole) {
                console.log(line);
            }
        }
    }

    public open(): void {
        if (!this.isOpen) {
            this.logFileFd = fs.openSync(this.logFilePath, 'a');
            this.isOpen = true;
        }
    }

    public close(): void {
        if (this.isOpen) {
            fs.closeSync(this.logFileFd);
            this.isOpen = false;
        }
    }
}

// many old test cases indirectly import `globals`
// provide a dummy stream or they will fail
// new test cases should import `globals/unittest` instead
export const dummyStream = <LogStream>{
    writeLine: (_line: string) => { /* empty */ },
    open: () => { /* empty */ },
    close: () => { /* empty */ }
};
