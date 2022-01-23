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
import { initPaths } from './paths';

export interface NniManagerArgs {
    readonly port: number;
    readonly experimentId: string;
    readonly action: 'create' | 'resume' | 'view';
    readonly experimentsDirectory: string;
    readonly logLevel: 'critical' | 'error' | 'warning' | 'info' | 'debug';
    readonly foreground: boolean;
    readonly urlPrefix: string;

    // these are scheduled to be removed
    readonly mode: string | undefined;
    readonly dispatcherPipe: string | undefined;
}

export interface NniPaths {
    readonly experimentsList: string;
    readonly logDirectory: string;
    readonly nniManagerLog: string;

    // these are scheduled to be removed
    readonly experimentRoot: string;
    readonly checkpointDirectory: string;
    readonly databaseDirectory: string;
}

export interface NniGlobals {
    readonly args: NniManagerArgs;
    readonly paths: NniPaths;
}

class Globals implements NniGlobals {
    args: NniManagerArgs;
    paths: NniPaths;

    constructor() {
        this.args = parseArgs(process.argv.slice(2));
        this.paths = initPaths(this.args);
    }
}

// give type hint to `global.nni` (copied from SO, dunno how it works)
declare global {
    var nni: NniGlobals;  // eslint-disable-line
}

if (global.nni === undefined) {
    global.nni = new Globals();
}

const nniGlobals: NniGlobals = global.nni;

export default nniGlobals;
