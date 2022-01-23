// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Unit test helper.
 *
 *  Import this module before index.ts will replace NNI globals with empty objects.
 *  You can then edit these mocked globals and the injection will be visible to all modules.
 **/

import type { NniManagerArgs, NniPaths } from './index';
import { LogStream, dummyStream } from './log_stream';
import { initPaths } from './paths';

// copied from https://www.typescriptlang.org/docs/handbook/2/mapped-types.html
type Mutable<Type> = {
    -readonly [Property in keyof Type]: Type[Property];
};

export class MutableGlobals {
    args: Mutable<NniManagerArgs> = <any>{};
    paths: Mutable<NniPaths> = <any>{};

    logStream: LogStream = dummyStream;

    init(args: NniManagerArgs): void {
        this.args = args;
        this.paths = initPaths(this.args);
    }
}

if (global.nni === undefined) {
    global.nni = <any>new MutableGlobals();
} else {
    const mutableGlobals = new MutableGlobals();
    Object.assign(global.nni, mutableGlobals);
    (<MutableGlobals>global.nni).init = mutableGlobals.init.bind(global.nni);
}

const globals = <MutableGlobals>global.nni;
export default globals;
