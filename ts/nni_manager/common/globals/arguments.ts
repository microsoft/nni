// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Parse NNI manager's command line arguments.
 **/

import assert from 'assert/strict';

import yargs from 'yargs/yargs';

/**
 *  Command line arguments provided by "nni/experiment/launcher.py".
 *
 *  Hyphen-separated words are automatically converted to camelCases by yargs lib, but snake_cases are not.
 *  So it supports "--log-level" but does not support "--log_level".
 *
 *  Unfortunately I misunderstood "experiment_working_directory" config field when deciding the name.
 *  It defaults to "~/nni-experiments" rather than "~/nni-experiments/<experiment-id>",
 *  and further more the working directory is "site-packages/nni_node", not either.
 *  For compatibility concern we cannot change the public API, so there is an inconsistency here.
 **/
export interface NniManagerArgs {
    readonly port: number;
    readonly experimentId: string;
    readonly action: 'create' | 'resume' | 'view';
    readonly experimentsDirectory: string;  // renamed "config.experiment_working_directory", must be absolute
    readonly logLevel: 'critical' | 'error' | 'warning' | 'info' | 'debug' | 'trace';
    readonly foreground: boolean;
    readonly urlPrefix: string;  // leading and trailing "/" must be stripped
    readonly tunerCommandChannel: string | null;
    readonly pythonInterpreter: string;

    // these are planned to be removed
    readonly mode: string;
}

export function parseArgs(rawArgs: string[]): NniManagerArgs {
    const parser = yargs(rawArgs).options(yargsOptions).strict().fail(false);
    const parsedArgs: NniManagerArgs = parser.parseSync();

    // strip yargs leftovers
    const argsAsAny: any = {};
    for (const key in yargsOptions) {
        argsAsAny[key] = (parsedArgs as any)[key];
        assert(!Number.isNaN(argsAsAny[key]), `Command line arg --${key} is not a number`);
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
        choices: [ 'critical', 'error', 'warning', 'info', 'debug', 'trace' ] as const,
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
    tunerCommandChannel: {
        default: null,
        type: 'string'
    },
    pythonInterpreter: {
        demandOption: true,
        type: 'string'
    },

    mode: {
        default: '',
        type: 'string'
    }
} as const;
