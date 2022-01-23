// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Manage experiment paths.
 **/

import assert from 'assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';

import type { NniManagerArgs, NniPaths } from './index';

export function initPaths(args: NniManagerArgs): NniPaths {
    assert(path.isAbsolute(args.experimentsDirectory), 'Command line arg --experiments-directory is not absolute');

    const experimentRoot = path.join(args.experimentsDirectory, args.experimentId);

    const checkpointDirectory = path.join(experimentRoot, 'checkpoint');
    const databaseDirectory = path.join(experimentRoot, 'db');
    const logDirectory = path.join(experimentRoot, 'log');

    // TODO: move all mkdir here
    fs.mkdirSync(logDirectory, { recursive: true });

    const nniManagerLog = path.join(logDirectory, 'nnimanager.log');

    const experimentsList = path.join(os.homedir(), 'nni-experiments', '.experiment');

    return {
        experimentsList,
        logDirectory,
        nniManagerLog,

        experimentRoot,
        checkpointDirectory,
        databaseDirectory,
    };
}
