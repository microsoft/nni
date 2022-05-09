// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Unit test helper.
 *  It should be inside "test", but must be here for compatibility, until we refactor all test cases.
 *
 *  Use this module to replace NNI globals with mocked values:
 *
 *      import 'common/globals/unittest';
 *
 *  Or:
 *
 *      import globals from 'common/globals/unittest';
 *
 *  You can then edit these mocked globals and the injection will be visible to all modules.
 *  Remember to invoke `resetGlobals()` in "after()" hook if you do so.
 *
 *  Attention: TypeScript will remove "unused" import statements. Use the first format when "globals" is never used.
 **/

import os from 'os';
import path from 'path';

import type { NniManagerArgs } from './arguments';
import { NniPaths, createPaths } from './paths';
import type { LogStream } from './log_stream';

// copied from https://www.typescriptlang.org/docs/handbook/2/mapped-types.html
type Mutable<Type> = {
    -readonly [Property in keyof Type]: Type[Property];
};

export interface MutableGlobals {
    args: Mutable<NniManagerArgs>;
    paths: Mutable<NniPaths>;
    logStream: LogStream;

    reset(): void;
}

export function resetGlobals(): void {
    const args: NniManagerArgs = {
        port: 8080,
        experimentId: 'unittest',
        action: 'create',
        experimentsDirectory: path.join(os.homedir(), 'nni-experiments'),
        logLevel: 'info',
        foreground: false,
        urlPrefix: '',
        tunerCommandChannel: null,
        pythonInterpreter: 'python',
        mode: 'unittest'
    };
    const paths = createPaths(args);
    const logStream = {
        writeLine: (_line: string): void => { /* dummy */ },
        writeLineSync: (_line: string): void => { /* dummy */ },
        close: async (): Promise<void> => { /* dummy */ }
    };
    const shutdown = {
        register: (..._: any): void => { /* dummy */ },
    };

    const globalAsAny = global as any;
    const utGlobals = { args, paths, logStream, shutdown, reset: resetGlobals };
    if (globalAsAny.nni === undefined) {
        globalAsAny.nni = utGlobals;
    } else {
        Object.assign(globalAsAny.nni, utGlobals);
    }
}

function isUnitTest(): boolean {
    const event = process.env['npm_lifecycle_event'] ?? '';
    return event.startsWith('test') || event === 'mocha' || event === 'nyc';
}

if (isUnitTest()) {
    resetGlobals();
}

const globals: MutableGlobals = (global as any).nni;
export default globals;
