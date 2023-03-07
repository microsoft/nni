import 'app-module-path/cwd';

import fs from 'node:fs';
import path from 'node:path';

import yargs from 'yargs/yargs';

import { NniManagerArgs, globals, initGlobalsCustom } from 'common/globals';
import { Logger, getRobustLogger } from 'common/log';

const logger: Logger = getRobustLogger('TrialKeeper.main');

function main() {
    process.on('SIGTERM', () => { globals.shutdown.initiate('SIGTERM'); });
    process.on('SIGBREAK', () => { globals.shutdown.initiate('SIGBREAK'); });
    process.on('SIGINT', () => { globals.shutdown.initiate('SIGINT'); });

    const args = parseArgs();

    const logPath = path.join(
        args.experimentsDirectory,
        args.experimentId,
        'env',
        args.environmentId,
        'trialkeeper.log'
    )
    fs.mkdirSync(path.dirname(logPath), { recursive: true });

    initGlobalsCustom(args, logPath);

    logger.info('Trial keeper start');
    logger.debug('Arguments:', args);

    //const daemon = new TrialKeeperDaemon(args.config);
    //globals.shutdown.notifyInitializeComplete();
    //keeper.start();
}

interface TrialKeeperArgs {
    readonly experimentId: string;
    readonly experimentsDirectory: string;
    readonly logLevel: 'critical' | 'error' | 'warning' | 'info' | 'debug' | 'trace';
    readonly pythonInterpreter: string;
    readonly platform: string;
    readonly environmentId: string;
    readonly managerCommandChannel: string;
}

interface MergedArgs extends NniManagerArgs, TrialKeeperArgs { }

function parseArgs(): MergedArgs {
    const rawArgs = process.argv.slice(2);
    const parser = yargs(rawArgs).options(yargsOptions).strict().fail(false);
    const args: TrialKeeperArgs = parser.parseSync();
    return {
        // shared args
        experimentId: args.experimentId,
        experimentsDirectory: args.experimentsDirectory,
        logLevel: args.logLevel,
        pythonInterpreter: args.pythonInterpreter,

        // trial keeper args
        platform: args.platform,
        environmentId: args.environmentId,
        managerCommandChannel: args.managerCommandChannel,

        // unused nni manager args
        port: 0,
        action: 'create',
        foreground: false,
        urlPrefix: '',
        tunerCommandChannel: null,
        mode: '',
    };
}

const yargsOptions = {
    experimentId: {
        demandOption: true,
        type: 'string'
    },
    experimentsDirectory: {
        demandOption: true,
        type: 'string'
    },
    logLevel: {
        choices: [ 'critical', 'error', 'warning', 'info', 'debug' ] as const,
        demandOption: true
    },
    pythonInterpreter: {
        demandOption: true,
        type: 'string'
    },

    platform: {
        demandOption: true,
        type: 'string'
    },
    environmentId: {
        demandOption: true,
        type: 'string'
    },
    managerCommandChannel: {
        demandOption: true,
        type: 'string'
    }
} as const;

main();
