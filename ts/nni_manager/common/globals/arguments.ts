// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Parse NNI manager's command line arguments.
 **/

import assert from 'assert/strict';

import yargs from 'yargs/yargs';

import type { NniManagerArgs } from './index';

export function parseArgs(rawArgs: string[]): NniManagerArgs {
    if (rawArgs === undefined) {
        rawArgs = process.argv.slice(2);
    }

    const parser = yargs(rawArgs).options(yargsOptions).strict().fail((_msg, err, _yargs) => { throw err; });
    const parsedArgs: NniManagerArgs = parser.parseSync();

    // strip yargs leftovers
    const argsAsAny: any = {};
    for (const key in yargsOptions) {
        argsAsAny[key] = (<any>parsedArgs)[key];
        assert(!Number.isNaN(argsAsAny[key]), `Command line arg --${key} is not a number`);
    }
    if (argsAsAny.mode === '') {
        argsAsAny.mode = undefined;
    }
    if (argsAsAny.dispatcherPipe === '') {
        argsAsAny.dispatcherPipe = undefined;
    }
    const args: NniManagerArgs = argsAsAny;

    const prefixErrMsg = `Command line arg --url-prefix "${args.urlPrefix}" is not stripped`;
    assert(!args.urlPrefix.startsWith('/') && !args.urlPrefix.endsWith('/'), prefixErrMsg);

    return args;
}

const yargsOptions = {
    port: {
        demandOption: true,
        type: 'number'
    },
    experimentId: {
        demandOption: true,
        type: 'string'
    },
    action: {
        choices: [ 'create', 'resume', 'view' ] as const,
        demandOption: true
    },
    experimentsDirectory: {
        demandOption: true,
        type: 'string'
    },
    logLevel: {
        choices: [ 'critical', 'error', 'warning', 'info', 'debug' ] as const,
        demandOption: true
    },
    foreground: {
        default: false,
        type: 'boolean'
    },
    urlPrefix: {
        default: '',
        type: 'string'
    },

    mode: {
        default: '',
        type: 'string'
    },
    dispatcherPipe: {
        default: '',
        type: 'string'
    }
} as const;
