// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Collection of global objects.
 *
 *  Although global is anti-pattern in OOP, there are two scenarios NNI uses globals.
 *
 *   1. Some constant configs (like command line args) are accessed here and there with util functions.
 *      It is possible to pass parameters instead, but not worthy the refactor.
 *
 *   2. Some singletons (like root logger) are indeed global.
 *      The singletons need to be registered in `global` to support 3rd-party training services,
 *      because they are compiled outside NNI manager and therefore module scope singletons will not work.
 **/

import assert from 'assert/strict';

import { NniManagerArgs, parseArgs } from './arguments';
import { LogStream, initLogStream, initLogStreamCustom } from './log_stream';
import { NniPaths, createPaths } from './paths';
import { RestManager } from './rest';
import { ShutdownManager } from './shutdown';

export { NniManagerArgs, NniPaths };

/**
 *  Collection of global objects.
 *
 *  It can be obtained with `import globals from 'common/globals'` or `global.nni`.
 *  The former is preferred because it exposes less underlying implementations.
 **/
export interface NniGlobals {
    readonly args: NniManagerArgs;
    readonly paths: NniPaths;
    readonly rest: RestManager;
    readonly shutdown: ShutdownManager;

    readonly logStream: LogStream;
}

// give type hint to `global.nni` (copied from SO, dunno how it works)
declare global {
    var nni: NniGlobals;  // eslint-disable-line
}

// prepare the namespace object and export it
if (global.nni === undefined) {
    global.nni = {} as NniGlobals;
}
export const globals: NniGlobals = global.nni;
export default globals;

/**
 *  Initialize globals.
 *  Must and must only be invoked once in "main.ts".
 **/
export function initGlobals(): void {
    assert.deepEqual(global.nni, {});

    const args = parseArgs(process.argv.slice(2));
    const paths = createPaths(args);
    const logStream = initLogStream(args, paths);
    const rest = new RestManager();
    const shutdown = new ShutdownManager();

    const globals: NniGlobals = { args, paths, logStream, rest, shutdown };
    Object.assign(global.nni, globals);
}

export function initGlobalsCustom(args: NniManagerArgs, logPath: string): void {
    assert.deepEqual(global.nni, {});

    const paths = createPaths(args);
    const logStream = initLogStreamCustom(args, logPath);
    const rest = new RestManager();
    const shutdown = new ShutdownManager();

    const globals: NniGlobals = { args, paths, logStream, rest, shutdown };
    Object.assign(global.nni, globals);
}
