// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Collection of global objects.
 *
 *  Though global is anit-pattern in OOP, there are two scenarios NNI uses globals.
 *
 *   1. Some constant configs (like command line args) are accessed here and there with util functions.
 *      It is possible to pass parameters instead, but not worthy the refactor.
 *
 *   2. Some singletons (like root logger) are semantically global.
 *      The singletons need to be registered in `global` to support 3rd-party training services,
 *      because they are compiled outside NNI manager and module scope singletons will not work.
 **/

import { parseArgs } from './arguments';
import { LogStream, dummyStream } from './log_stream';
import { initPaths } from './paths';

/**
 *  Collection of all global objects.
 *
 *  It can be obtained with `import globals from '...'` or `global.nni`.
 *  The former is preferred because it exposes less implementation details.
 **/
export interface NniGlobals {
    // Type-1 globals, public API
    readonly args: NniManagerArgs;
    readonly paths: NniPaths;

    // Type-2 globals, should not be used by normal modules
    readonly logStream: LogStream;
}

/**
 *  Command line arguments provided by "nni/experiment/launcher.py".
 *
 *  Hyphen-separated words are automatically converted to camelCases by yargs lib, but snake_cases are not.
 *  So it supports "--log-level" but does not support "--log_level".
 **/
export interface NniManagerArgs {
    readonly port: number;
    readonly experimentId: string;
    readonly action: 'create' | 'resume' | 'view';
    readonly experimentsDirectory: string;  // must be absolute
    readonly logLevel: 'critical' | 'error' | 'warning' | 'info' | 'debug';
    readonly foreground: boolean;
    readonly urlPrefix: string;  // leading and trailing slashes are all stripped (python side's responsibility)

    // these are planned to be removed
    readonly mode: string | undefined;
    readonly dispatcherPipe: string | undefined;
}

export interface NniPaths {
    readonly experimentsList: string;  // the json file containing basic info of all experiments
    readonly logDirectory: string;  // directory of nni manager log and dispatcher log. trial logs are not here
    readonly nniManagerLog: string;

    // these are planned to be removed? (if they are never directly used anywhere)
    readonly experimentRoot: string;
    readonly checkpointDirectory: string;
    readonly databaseDirectory: string;
}

class Globals implements NniGlobals {
    args!: NniManagerArgs;
    paths!: NniPaths;
    logStream: LogStream = dummyStream;

    init(): void {
        this.args = parseArgs(process.argv.slice(2));
        this.paths = initPaths(this.args);
        this.logStream = new LogStream(this.args, this.paths);
    }
}

// give type hint to `global.nni` (copied from SO, dunno how it works)
declare global {
    var nni: NniGlobals;  // eslint-disable-line
}

if (global.nni === undefined) {
    global.nni = new Globals();
}  // otherwise this is inside a 3rd-party training service and globals are already initialized
const globals: NniGlobals = global.nni;
export default globals;

// Must and must only be invoked in "main.ts".
export function initGlobals(): void {
    (<Globals>globals).init();
}
