// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import 'app-module-path/register';
import { Container, Scope } from 'typescript-ioc';

import * as fs from 'fs';
import * as path from 'path';
import * as component from './common/component';
import { Database, DataStore } from './common/datastore';
import { setExperimentStartupInfo } from './common/experimentStartupInfo';
import { getLogger, setLogLevel, startLogging } from './common/log';
import { Manager, ExperimentStartUpMode } from './common/manager';
import { ExperimentManager } from './common/experimentManager';
import { TensorboardManager } from './common/tensorboardManager';
import { getLogDir, mkDirP, parseArg } from './common/utils';
import { NNIDataStore } from './core/nniDataStore';
import { NNIManager } from './core/nnimanager';
import { SqlDB } from './core/sqlDatabase';
import { NNIExperimentsManager } from './core/nniExperimentsManager';
import { NNITensorboardManager } from './core/nniTensorboardManager';
import { RestServer } from './rest_server';
import { parseArgs } from 'common/globals/arguments';

const args = parseArgs(process.argv.slice(2));

async function initContainer(): Promise<void> {
    Container.bind(Manager)
        .to(NNIManager)
        .scope(Scope.Singleton);
    Container.bind(Database)
        .to(SqlDB)
        .scope(Scope.Singleton);
    Container.bind(DataStore)
        .to(NNIDataStore)
        .scope(Scope.Singleton);
    Container.bind(ExperimentManager)
        .to(NNIExperimentsManager)
        .scope(Scope.Singleton);
    Container.bind(TensorboardManager)
        .to(NNITensorboardManager)
        .scope(Scope.Singleton);
    const DEFAULT_LOGFILE: string = path.join(getLogDir(), 'nnimanager.log');
    if (!args.foreground) {
        startLogging(DEFAULT_LOGFILE);
    }
    // eslint-disable-next-line @typescript-eslint/no-use-before-define
    setLogLevel(args.logLevel);
    const ds: DataStore = component.get(DataStore);

    await ds.init();
}

setExperimentStartupInfo(
    args.action === 'create',
    args.experimentId,
    args.port,
    args.mode,
    args.experimentsDirectory,
    args.logLevel,
    args.action === 'view',
    args.dispatcherPipe ?? '',
    args.urlPrefix
);

mkDirP(getLogDir())
    .then(async () => {
        try {
            await initContainer();
            const restServer: RestServer = component.get(RestServer);
            await restServer.start();
        } catch (err) {
            getLogger('main').error(`${err.stack}`);
            throw err;
        }
    })
    .catch((err: Error) => {
        console.error(`Failed to create log dir: ${err.stack}`);
    });

function cleanUp(): void {
    (component.get(Manager) as Manager).stopExperiment();
}

process.on('SIGTERM', cleanUp);
process.on('SIGBREAK', cleanUp);
process.on('SIGINT', cleanUp);
