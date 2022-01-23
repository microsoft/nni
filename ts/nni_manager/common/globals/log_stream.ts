// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import fs from 'fs';

import type { NniManagerArgs, NniPaths } from './index';

export class LogStream {
    private isOpen: boolean = false;
    private logFilePath: string;
    private logFileStream!: fs.WriteStream;
    private toConsole: boolean;

    constructor(args: NniManagerArgs, paths: NniPaths) {
        this.logFilePath = paths.nniManagerLog;
        this.toConsole = args.foreground;
        this.open();
    }

    public writeLine(line: string): void {
        if (this.isOpen) {
            this.logFileStream.write(line + '\n');
            if (this.toConsole) {
                console.log(line);
            }
        }
    }

    public open(): void {
        if (!this.isOpen) {
            this.logFileStream = fs.createWriteStream(this.logFilePath, { flags: 'a' });
            this.isOpen = true;
        }
    }

    public close(): void {
        if (this.isOpen) {
            this.logFileStream.close();
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
