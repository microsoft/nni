// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Unit test helper.
 *
 *  Use this module to replace NNI globals with mocked values:
 *
 *      import globals from 'common/globals/unittest';
 *
 *  You can then edit these mocked globals and the injection will be visible to all modules.
 *  Remember to invoke `resetGlobals()` in after hook if you do so.
 **/

import os from 'os';
import path from 'path';

import type { NniManagerArgs } from './arguments';
import { NniPaths, createPaths } from './paths';

// copied from https://www.typescriptlang.org/docs/handbook/2/mapped-types.html
type Mutable<Type> = {
    -readonly [Property in keyof Type]: Type[Property];
};

export interface MutableGlobals {
    args: Mutable<NniManagerArgs>;
    paths: Mutable<NniPaths>;
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
        mode: 'unittest',
        dispatcherPipe: undefined
    };

    const paths = createPaths(args);

    const globals = { args, paths };
    if (global.nni === undefined) {
        global.nni = globals;
    } else {
        Object.assign(global.nni, globals);
    }
}

function isUnitTest(): boolean {
    const event = process.env['npm_lifecycle_event'] ?? '';
    return event.startsWith('test') || event === 'mocha' || event === 'nyc';
}

if (isUnitTest()) {
    resetGlobals();
}

const globals: MutableGlobals = global.nni;
export default globals;
