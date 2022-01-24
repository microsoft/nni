// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Entry point of NNI manager.
 *
 *  NNI manager is normally started by "nni/experiment/launcher.py".
 *  It requires command line arguments defined as NniManagerArgs in "common/globals/index.ts".
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
import { getLogger, setLogLevel, startLogging } from 'common/log';
import { Manager } from 'common/manager';
import { TensorboardManager } from 'common/tensorboardManager';
import { NNIDataStore } from 'core/nniDataStore';
import { NNIExperimentsManager } from 'core/nniExperimentsManager';
import { NNITensorboardManager } from 'core/nniTensorboardManager';
import { NNIManager } from 'core/nnimanager';
import { SqlDB } from 'core/sqlDatabase';
import { RestServer } from 'rest_server';

async function start(): Promise<void> {
    getLogger('main').info('Start NNI manager');

    Container.bind(Manager).to(NNIManager).scope(Scope.Singleton);
    Container.bind(Database).to(SqlDB).scope(Scope.Singleton);
    Container.bind(DataStore).to(NNIDataStore).scope(Scope.Singleton);
    Container.bind(ExperimentManager).to(NNIExperimentsManager).scope(Scope.Singleton);
    Container.bind(TensorboardManager).to(NNITensorboardManager).scope(Scope.Singleton);

    const ds: DataStore = component.get(DataStore);
    await ds.init();

    const restServer = new RestServer(globals.args.port, globals.args.urlPrefix);
    await restServer.start();
}

function shutdown(): void {
    (component.get(Manager) as Manager).stopExperiment();
}

// Free training service resources on unexpected shutdown.
// A graceful stop should use REST API.
// Both call the same function, but interrupts can cause strange behaviors in children processes.
process.on('SIGBREAK', shutdown);
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

/* main */

initGlobals();

start().then(() => {
    getLogger('main').debug('start() returned.');
}).catch((err: any) => {
    try {
        getLogger('main').error('Failed to start:', err);
    } catch (loggerError: Error) {
        console.log('Failed to start:', err);
        console.log('Seems logger is faulty:', loggerError);
    }
    process.exit(1);
});

// Node.js exits when there is no active handler,
// and we have registered a lot of handlers which are never cleaned up.
// So it runs forever until NNIManager calls `process.exit()`.
