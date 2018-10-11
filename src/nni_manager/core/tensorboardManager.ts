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

import {
    BoardManager
} from '../common/manager';
import { delay, getExperimentRootDir, uniqueString } from '../common/utils';
import * as component from '../common/component';
import { DataStore, TrialJobInfo } from '../common/datastore';
import { NNIErrorNames } from '../common/errors';
import { getLogger, Logger } from '../common/log';
import * as cpp from 'child-process-promise';
import * as cp from 'child_process';
import { HostJobApplicationForm, TrainingService, TrialJobStatus } from '../common/trainingService';


/**
 * TensorboardManager
 */
class TensorboardManager implements BoardManager {
    
    private DEFAULT_PORT: number = 6006;
    private TENSORBOARD_COMMAND: string = 'PATH=$PATH:~/.local/bin:/usr/local/bin tensorboard';
    private tbPortTrialMap: Map<number, string>;
    private trialTbPidMap: Map<string, number>;
    private trainingService: TrainingService;
    private dataStore: DataStore;
    private log: Logger = getLogger();

    constructor() {
        this.tbPortTrialMap = new Map();
        this.trialTbPidMap = new Map();
        this.trainingService = component.get(TrainingService);
        this.dataStore = component.get(DataStore);
    }

    public async startTensorBoard(trialJobIds: string[], tbCmd?: string, port?: number): Promise<string> {
        const trialJobId = trialJobIds[0];
        console.log('-------------------------------57----------------')
        let tensorBoardPort: number = this.DEFAULT_PORT;
        if (port !== undefined) {
            tensorBoardPort = port;
        }
        const tbEndpoint: string = `http://localhost:${tensorBoardPort}`;
        console.log('-------------------------------63----------------')
        this.tbPortTrialMap.set(tensorBoardPort, trialJobId);
        try {
            if (await this.isTensorBoardRunning(tensorBoardPort)) {
                await this.stopTensorBoard(tensorBoardPort);
            }
        } catch (error) {
            if (error.name !== NNIErrorNames.NOT_FOUND) {
                throw error;
            } else {
                this.tbPortTrialMap.delete(tensorBoardPort);
            }
        }
        const logDirs: string[] = [];
        console.log('-------------------------------77----------------')
        logDirs.push(await this.getLogDir(trialJobId));
        
        let tensorBoardCmd: string = this.TENSORBOARD_COMMAND;
        if (tbCmd !== undefined && tbCmd.trim().length > 0) {
            tensorBoardCmd = tbCmd;
        }
        console.log('-------------------------------84----------------')
        const cmd: string = `${tensorBoardCmd} --logdir ${logDirs.join(':')} --port ${tensorBoardPort}`;
        const pid: number = await this.runTensorboardProcess(cmd);
        console.log('-------------------------------87----------------')
        this.trialTbPidMap.set(trialJobId, pid);

        return tbEndpoint;
    }
    
    public async runTensorboardProcess(cmd: string): Promise<number>{
        console.log('-------------------------------94----------------')
        const process: cp.ChildProcess = cp.exec(cmd);
        return Promise.resolve(process.pid);
    }
    
    public async cleanUp(): Promise<void> {
        
    }
    
    public async stopTensorBoard(tensorBoardPort: number): Promise<void> {
    
    }
    
    private async isTensorBoardRunning(port: number): Promise<boolean> {
        return Promise.resolve(false);
    }
    
    private async getLogDir(trialJobId: string): Promise<string> {
        const jobInfo: TrialJobInfo = await this.dataStore.getTrialJob(trialJobId);
        const logPath: string | undefined = jobInfo.logPath;
        if (logPath === undefined) {
            throw new Error(`Failed to find job logPath: ${jobInfo.id}`);
        }
        return logPath.split('://')[1].split(':')[1]; //TODO use url parse
    }
}

export { TensorboardManager };
