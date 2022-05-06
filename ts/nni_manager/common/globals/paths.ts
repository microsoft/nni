// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Manage experiment paths.
 *
 *  Ideally all path constants should be put here so other modules (especially training services)
 *  do not need to know file hierarchy of nni-experiments folder, which is an implicit undocumented protocol.
 **/

import assert from 'assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';

import type { NniManagerArgs } from './arguments';

export interface NniPaths {
    readonly experimentRoot: string;
    readonly experimentsDirectory: string;
    readonly experimentsList: string;
    readonly logDirectory: string;  // contains nni manager and dispatcher log; trial logs are not here
    readonly nniManagerLog: string;
}

export function createPaths(args: NniManagerArgs): NniPaths {
    assert(
        path.isAbsolute(args.experimentsDirectory),
        `Command line arg --experiments-directory "${args.experimentsDirectory}" is not absoulte`
    );
    const experimentRoot = path.join(args.experimentsDirectory, args.experimentId);

    const logDirectory = path.join(experimentRoot, 'log');

    // TODO: move all `mkdir`s here
    fs.mkdirSync(logDirectory, { recursive: true });

    const nniManagerLog = path.join(logDirectory, 'nnimanager.log');

    // TODO: this should follow experiments directory config
    const experimentsList = path.join(os.homedir(), 'nni-experiments', '.experiment');

    return {
        experimentRoot,
        experimentsDirectory: args.experimentsDirectory,
        experimentsList,
        logDirectory,
        nniManagerLog,
    };
}
