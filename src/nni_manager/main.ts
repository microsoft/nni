/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';

import { Container, Scope } from 'typescript-ioc';

import * as component from './common/component';
import * as fs from 'fs';
import { Database, DataStore } from './common/datastore';
import { setExperimentStartupInfo } from './common/experimentStartupInfo';
import { getLogger, Logger, logLevelNameMap } from './common/log';
import { Manager } from './common/manager';
import { TrainingService } from './common/trainingService';
import { parseArg, uniqueString, mkDirP, getLogDir } from './common/utils';
import { NNIDataStore } from './core/nniDataStore';
import { NNIManager } from './core/nnimanager';
import { SqlDB } from './core/sqlDatabase';
import { NNIRestServer } from './rest_server/nniRestServer';
import { LocalTrainingServiceForGPU } from './training_service/local/localTrainingServiceForGPU';
import {
    RemoteMachineTrainingService
} from './training_service/remote_machine/remoteMachineTrainingService';
import { PAITrainingService } from './training_service/pai/paiTrainingService';
import { KubeflowTrainingService } from './training_service/kubernetes/kubeflow/kubeflowTrainingService';
import { FrameworkControllerTrainingService } from './training_service/kubernetes/frameworkcontroller/frameworkcontrollerTrainingService';

function initStartupInfo(startExpMode: string, resumeExperimentId: string, basePort: number, logDirectory: string, experimentLogLevel: string) {
    const createNew: boolean = (startExpMode === 'new');
    const expId: string = createNew ? uniqueString(8) : resumeExperimentId;
    setExperimentStartupInfo(createNew, expId, basePort, logDirectory, experimentLogLevel);
}

async function initContainer(platformMode: string): Promise<void> {
    if (platformMode === 'local') {
        Container.bind(TrainingService).to(LocalTrainingServiceForGPU).scope(Scope.Singleton);
    } else if (platformMode === 'remote') {
        Container.bind(TrainingService).to(RemoteMachineTrainingService).scope(Scope.Singleton);
    } else if (platformMode === 'pai') {
        Container.bind(TrainingService).to(PAITrainingService).scope(Scope.Singleton);
    } else if (platformMode === 'kubeflow') {
        Container.bind(TrainingService).to(KubeflowTrainingService).scope(Scope.Singleton);
    } else if (platformMode === 'frameworkcontroller') {
        Container.bind(TrainingService).to(FrameworkControllerTrainingService).scope(Scope.Singleton);
    }
    else {
        throw new Error(`Error: unsupported mode: ${mode}`);
    }
    Container.bind(Manager).to(NNIManager).scope(Scope.Singleton);
    Container.bind(Database).to(SqlDB).scope(Scope.Singleton);
    Container.bind(DataStore).to(NNIDataStore).scope(Scope.Singleton);
    const ds: DataStore = component.get(DataStore);

    await ds.init();
}

function usage(): void {
    console.info('usage: node main.js --port <port> --mode <local/remote/pai/kubeflow/frameworkcontroller> --start_mode <new/resume> --experiment_id <id>');
}

const strPort: string = parseArg(['--port', '-p']);
if (!strPort || strPort.length === 0) {
    usage();
    process.exit(1);
}

const port: number = parseInt(strPort, 10);

const mode: string = parseArg(['--mode', '-m']);
if (!['local', 'remote', 'pai', 'kubeflow', 'frameworkcontroller'].includes(mode)) {
    console.log(`FATAL: unknown mode: ${mode}`);
    usage();
    process.exit(1);
}

const startMode: string = parseArg(['--start_mode', '-s']);
if (!['new', 'resume'].includes(startMode)) {
    console.log(`FATAL: unknown start_mode: ${startMode}`);
    usage();
    process.exit(1);
}

const experimentId: string = parseArg(['--experiment_id', '-id']);
if (startMode === 'resume' && experimentId.trim().length < 1) {
    console.log(`FATAL: cannot resume experiment, invalid experiment_id: ${experimentId}`);
    usage();
    process.exit(1);
}

const logDir: string = parseArg(['--log_dir', '-ld']);
if (logDir.length > 0) {
    if (!fs.existsSync(logDir)) {
        console.log(`FATAL: log_dir ${logDir} does not exist`);
    }
}

const logLevel: string = parseArg(['--log_level', '-ll']);
if (logLevel.length > 0 && !logLevelNameMap.has(logLevel)) {
    console.log(`FATAL: invalid log_level: ${logLevel}`);
}

initStartupInfo(startMode, experimentId, port, logDir, logLevel);

mkDirP(getLogDir()).then(async () => {
    const log: Logger = getLogger();
    try {
        await initContainer(mode);
        const restServer: NNIRestServer = component.get(NNIRestServer);
        await restServer.start();
        log.info(`Rest server listening on: ${restServer.endPoint}`);
    } catch (err) {
        log.error(`${err.stack}`);
        throw err;
    }
}).catch((err: Error) => {
    console.error(`Failed to create log dir: ${err.stack}`);
});

process.on('SIGTERM', async () => {
    const log: Logger = getLogger();
    let hasError: boolean = false;
    try{
        const nniManager: Manager = component.get(Manager);
        await nniManager.stopExperiment();
        const ds: DataStore = component.get(DataStore);
        await ds.close();
        const restServer: NNIRestServer = component.get(NNIRestServer);
        await restServer.stop();
    }catch(err){
        hasError = true;
        log.error(`${err.stack}`);
    }finally{
        await log.close();
        process.exit(hasError?1:0);
    }
})