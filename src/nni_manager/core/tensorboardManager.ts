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
import { HostJobApplicationForm, TrainingService, TrialJobStatus, ITensorBoardManager } from '../common/trainingService';
import { PAITrainingService } from '../training_service/pai/paiTrainingService'
import { RemoteMachineTrainingService } from '../training_service/remote_machine/remoteMachineTrainingService'
import { MethodNotImplementedError, NNIError} from '../common/errors';

/**
 * TensorboardManager
 */
class TensorboardManager implements BoardManager {
    
    private DEFAULT_PORT: number = 6006;
    private TENSORBOARD_COMMAND: string = 'PATH=$PATH:~/.local/bin:/usr/local/bin tensorboard';
    private tbPid?: number;
    private isRunning: boolean;
    private tensorBoardManagerInstance: ITensorBoardManager;
    private log: Logger;

    constructor() {
        this.log = getLogger();
        this.isRunning = false;
        let trainingService: any = component.get(TrainingService);
        this.tensorBoardManagerInstance = trainingService as ITensorBoardManager;
    }
    
    public async Run(trialJobId: string): Promise<void>{
        while(this.isRunning){
            this.tensorBoardManagerInstance.addCopyDataTask(trialJobId);
            await delay(60000);
        }
    }

    public async startTensorBoard(trialJobIds: string[], tbCmd?: string, port?: number): Promise<string> {
        
        let tensorBoardPort: number = this.DEFAULT_PORT;
        if (port !== undefined) {
            tensorBoardPort = port;
        }
        const tbEndpoint: string = `http://localhost:${tensorBoardPort}`;
        
        if(this.tbPid !== undefined){
            await this.stopTensorBoard();
        }

        const logDirs: string[] = [];
        for(let trialJobId of trialJobIds){
            const localDir = this.tensorBoardManagerInstance.getLocalDirectory(trialJobId);
            logDirs.push(localDir);
        }

        let tensorBoardCmd: string = this.TENSORBOARD_COMMAND;
        if (tbCmd !== undefined && tbCmd.trim().length > 0) {
            tensorBoardCmd = tbCmd;
        }
        const cmd: string = `${tensorBoardCmd} --logdir ${logDirs.join(':')} --port ${tensorBoardPort}`;
        this.tbPid = await this.runTensorBoardProcess(cmd);
        
        for(let trialJobId of trialJobIds){
            this.Run(trialJobId).catch((error)=>{
                 this.log.error(`Run copy data error: ${error}`);
            });  
        }
   
        return tbEndpoint;
    }
    
    public async runTensorBoardProcess(cmd: string): Promise<number>{
        const process: cp.ChildProcess = cp.exec(cmd);
        this.isRunning = true;
        return Promise.resolve(process.pid);
    }
    
    public async stopTensorBoard(): Promise<void> {
        if(this.tbPid !== undefined){
            await cpp.exec(`pkill -9 -P ${this.tbPid}`);
            this.tbPid = undefined;
            this.isRunning = false;
        }
        Promise.resolve();
    }
}

export { TensorboardManager };
