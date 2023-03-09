// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import 'app-module-path/cwd';

import fs from 'node:fs/promises';
import path from 'node:path';
import util from 'node:util';

// FIXME: reordering imports may cause circular dependency (imported objects become undefined)

import { globals, initGlobalsCustom } from 'common/globals';
import { Logger, getRobustLogger } from 'common/log';
import { WsChannelClient } from 'common/command_channel/websocket/client';
import { RestServerCore } from 'rest_server/core';
import { registerOnChannel } from './rpc';

const logger: Logger = getRobustLogger('TrialKeeper.main');

interface TrialKeeperConfig {
    readonly experimentId: string;
    readonly experimentsDirectory: string;
    readonly logLevel: 'critical' | 'error' | 'warning' | 'info' | 'debug' | 'trace';
    readonly pythonInterpreter: string;
    readonly platform: string;
    readonly environmentId: string;
    readonly managerCommandChannel: string;
}

async function main(): Promise<void> {
    process.on('SIGTERM', () => { globals.shutdown.initiate('SIGTERM'); });
    process.on('SIGBREAK', () => { globals.shutdown.initiate('SIGBREAK'); });
    process.on('SIGINT', () => { globals.shutdown.initiate('SIGINT'); });

    const workDir = process.argv[2];

    const configPath = path.join(workDir, 'trial_keeper_config.json');
    const config: TrialKeeperConfig = JSON.parse(await fs.readFile(configPath, { encoding: 'utf8' }));
    const args = {
        // shared args
        experimentId: config.experimentId,
        experimentsDirectory: config.experimentsDirectory,
        logLevel: config.logLevel,
        pythonInterpreter: config.pythonInterpreter,

        // trial keeper args
        platform: config.platform,
        environmentId: config.environmentId,
        managerCommandChannel: config.managerCommandChannel,

        // unused nni manager args
        port: 0,
        action: 'create',
        foreground: false,
        urlPrefix: '',
        tunerCommandChannel: null,
        mode: '',
    } as const;

    const logPath = path.join(workDir, 'trial_keeper.log');

    initGlobalsCustom(args, logPath);

    logger.info('Trial keeper start');
    logger.debug('command:', process.argv);
    logger.debug('config:', config);

    const client = new WsChannelClient(args.managerCommandChannel);
    registerOnChannel(client);
    await client.connect();

    const restServer = new RestServerCore();
    await restServer.start();
    logger.info('Running on port', globals.args.port);

    logger.info('Initialized');
    globals.shutdown.notifyInitializeComplete();

    // notify launcher
    await fs.writeFile(path.join(workDir, 'success.flag'), 'ok');
}

if (!process.argv[1].endsWith('mocha')) {  // the unit test imports all scripts and will reach here
    main().catch(error => {
        logger.critical(error);
        console.error(util.inspect(error));
        process.exit(1);
    });
}
