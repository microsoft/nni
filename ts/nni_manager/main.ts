// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Entry point of NNI manager.
 *
 *  NNI manager is normally started by "nni/experiment/launcher.py".
 *  It requires command line arguments defined as NniManagerArgs in "common/globals/arguments.ts".
 *
 *  Example usage:
 *
 *      node main.js \
 *          --port 8080 \
 *          --experiment-id ID \
 *          --action create \
 *          --experiments-directory /home/USER/nni-experiments \
 *          --log-level info \
 *          --foreground false \  (optional)
 *          --mode local  (required for now, will be removed later)
 **/

import 'app-module-path/register';  // so we can use absolute path to import

import fs from 'fs';

import { Container, Scope } from 'typescript-ioc';

import * as component from 'common/component';
import { Database, DataStore } from 'common/datastore';
import { ExperimentManager } from 'common/experimentManager';
import globals, { initGlobals } from 'common/globals';
import { Logger, getLogger } from 'common/log';
import { Manager } from 'common/manager';
import { TensorboardManager } from 'common/tensorboardManager';
import { NNIDataStore } from 'core/nniDataStore';
import { NNIManager } from 'core/nnimanager';
import { SqlDB } from 'core/sqlDatabase';
import { NNIExperimentsManager } from 'extensions/experiments_manager';
import { NNITensorboardManager } from 'extensions/nniTensorboardManager';
import { RestServer } from 'rest_server';

import path from 'path';

const logger: Logger = getLogger('main');

async function start(): Promise<void> {
    logger.info('Start NNI manager');

    Container.bind(Manager).to(NNIManager).scope(Scope.Singleton);
    Container.bind(Database).to(SqlDB).scope(Scope.Singleton);
    Container.bind(DataStore).to(NNIDataStore).scope(Scope.Singleton);
    Container.bind(ExperimentManager).to(NNIExperimentsManager).scope(Scope.Singleton);
    Container.bind(TensorboardManager).to(NNITensorboardManager).scope(Scope.Singleton);

    const ds: DataStore = component.get(DataStore);
    await ds.init();

    const restServer = new RestServer(globals.args.port, globals.args.urlPrefix);
    await restServer.start();

    globals.shutdown.notifyInitializeComplete();
}

// Register callbacks to free training service resources on unexpected shutdown.
// A graceful stop should use REST API,
// because interrupts can cause strange behaviors in children processes.
process.on('SIGTERM', () => { globals.shutdown.initiate('SIGTERM'); });
process.on('SIGBREAK', () => { globals.shutdown.initiate('SIGBREAK'); });
process.on('SIGINT', () => { globals.shutdown.initiate('SIGINT'); });

/* main */

initGlobals();

start().then(() => {
    logger.debug('start() returned.');
}).catch((error) => {
    try {
        logger.error('Failed to start:', error);
    } catch (loggerError) {
        console.error('Failed to start:', error);
        console.error('Seems logger is faulty:', loggerError);
    }
    process.exit(1);
});

// Node.js exits when there is no active handler,
// and we have registered a lot of handlers which are never cleaned up.
// So it runs forever until NNIManager calls `process.exit()`.
