import 'app-module-path/cwd';

import fs from 'node:fs/promises';
import path from 'node:path';

import yargs from 'yargs/yargs';

import { NniManagerArgs, globals, initGlobalsCustom } from 'common/globals';
import { Logger, getRobustLogger } from 'common/log';

import { WsChannelClient } from 'common/command_channel/websocket/client';
import { RemoteTrialKeeper, registerForChannel } from './rpc';

import { RestServerCore } from 'rest_server/core';

const logger: Logger = getRobustLogger('TrialKeeper.main');

interface TrialKeeperArgs {
    readonly experimentId: string;
    readonly experimentsDirectory: string;
    readonly logLevel: 'critical' | 'error' | 'warning' | 'info' | 'debug' | 'trace';
    readonly pythonInterpreter: string;
    readonly platform: string;
    readonly environmentId: string;
    readonly managerCommandChannel: string;
}

async function main(): Promise<void> {
    console.log('Start trial keeper:', process.argv);

    process.on('SIGTERM', () => { globals.shutdown.initiate('SIGTERM'); });
    process.on('SIGBREAK', () => { globals.shutdown.initiate('SIGBREAK'); });
    process.on('SIGINT', () => { globals.shutdown.initiate('SIGINT'); });

    const args = await parseArgs();

    const envDir = path.join(
        args.experimentsDirectory,
        args.experimentId,
        'environments',
        args.environmentId
    )
    //await fs.mkdir(envDir, { recursive: true });

    // SFTP requires the upload dir to exist
    await fs.mkdir(path.join(envDir, 'upload'), { recursive: true });

    const logPath = path.join(envDir, 'log.txt')
    initGlobalsCustom(args, logPath);
    logger.info('Start');
    logger.info('    args:', process.argv);
    logger.info('    config:', args);

    const restServer = new RestServerCore(8081);
    await restServer.start();

    const client = new WsChannelClient(args.managerCommandChannel);
    registerForChannel(client);
    await client.connect();

    logger.info('Initialized');
    globals.shutdown.notifyInitializeComplete();
}

interface MergedArgs extends NniManagerArgs, TrialKeeperArgs { }

async function parseArgs(): Promise<MergedArgs> {
    const configPath = process.argv[2];
    const configJson = await fs.readFile(configPath, { encoding: 'utf8' });
    const args: TrialKeeperArgs = JSON.parse(configJson);
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
        port: 8081,
        action: 'create',
        foreground: false,
        urlPrefix: '',
        tunerCommandChannel: null,
        mode: '',
    };
}

main();
