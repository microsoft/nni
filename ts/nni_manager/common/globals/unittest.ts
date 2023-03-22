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
import { LogStream } from './log_stream';
import { NniPaths, createPaths } from './paths';
import { RestManager } from './rest';

// Enforce ts-node to compile `shutdown.ts`.
// Without this line it might complain "log_1.getRobustLogger is not a function".
// "Magic. Do not touch."
import './shutdown';

// copied from https://www.typescriptlang.org/docs/handbook/2/mapped-types.html
type Mutable<Type> = {
    -readonly [Property in keyof Type]: Type[Property];
};

export interface MutableGlobals {
    args: Mutable<NniManagerArgs>;
    paths: Mutable<NniPaths>;
    logStream: LogStream;
    rest: RestManager;

    reset(): void;
    showLog(): void;
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
    const rest = new RestManager();
    const shutdown = {
        register: (..._: any): void => { /* dummy */ },
    };

    const globalAsAny = global as any;
    const utGlobals = { args, paths, logStream, rest, shutdown, reset: resetGlobals, showLog };
    if (globalAsAny.nni === undefined) {
        globalAsAny.nni = utGlobals;
    } else {
        Object.assign(globalAsAny.nni, utGlobals);
    }
}

function showLog(): void {
    globals.args.logLevel = 'trace';
    globals.logStream.writeLine = (line): void => { console.debug(line); };
    globals.logStream.writeLineSync = (line): void => { console.debug(line); };
}

function isUnitTest(): boolean {
    if ((global as any).nniUnitTest) {
        return true;
    }
    const event = process.env['npm_lifecycle_event'] ?? '';
    return event.startsWith('test') || event === 'mocha' || event === 'nyc';
}

if (isUnitTest()) {
    resetGlobals();
}

export const globals: MutableGlobals = (global as any).nni;
export default globals;
