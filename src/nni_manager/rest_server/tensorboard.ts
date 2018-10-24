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

import * as component from '../common/component';
import { DataStore, TrialJobInfo } from '../common/datastore';
import { NNIErrorNames } from '../common/errors';
import { getLogger, Logger } from '../common/log';
import { HostJobApplicationForm, TrainingService, TrialJobStatus } from '../common/trainingService';

export class TensorBoard {
    private DEFAULT_PORT: number = 6006;
    private TENSORBOARD_COMMAND: string = 'PATH=$PATH:~/.local/bin:/usr/local/bin tensorboard';
    private tbJobMap: Map<string, string>;
    private trainingService: TrainingService;
    private dataStore: DataStore;
    private log: Logger = getLogger();

    constructor() {
        this.tbJobMap = new Map();
        this.trainingService = component.get(TrainingService);
        this.dataStore = component.get(DataStore);
    }

    public async startTensorBoard(trialJobIds: string[], tbCmd?: string, port?: number): Promise<string> {
        let tensorBoardPort: number = this.DEFAULT_PORT;
        if (port !== undefined) {
            tensorBoardPort = port;
        }
        const host: string = await this.getJobHost(trialJobIds);
        const tbEndpoint: string = `http://${host}:${tensorBoardPort}`;

        try {
            if (await this.isTensorBoardRunningOnHost(host)) {
                await this.stopHostTensorBoard(host);
            }
        } catch (error) {
            if (error.name !== NNIErrorNames.NOT_FOUND) {
                throw error;
            } else {
                this.tbJobMap.delete(host);
            }
        }
        const logDirs: string[] = [];

        for (const id of trialJobIds) {
            logDirs.push(await this.getLogDir(id));
        }
        let tensorBoardCmd: string = this.TENSORBOARD_COMMAND;
        if (tbCmd !== undefined && tbCmd.trim().length > 0) {
            tensorBoardCmd = tbCmd;
        }
        const cmd: string = `${tensorBoardCmd} --logdir ${logDirs.join(':')} --port ${tensorBoardPort}`;

        const form: HostJobApplicationForm = {
            jobType: 'HOST',
            host: host,
            cmd: cmd
        };

        const jobId: string = (await this.trainingService.submitTrialJob(form)).id;
        this.tbJobMap.set(host, jobId);

        return tbEndpoint;
    }

    public async cleanUp(): Promise<void> {
        const stopTensorBoardTasks: Promise<void>[] = [];
        this.tbJobMap.forEach((jobId: string, host: string) => {
            stopTensorBoardTasks.push(this.stopHostTensorBoard(host).catch((err: Error) => {
                this.log.error(`Error occurred stopping tensorboard service: ${err.message}`);
            }));
        });
        await Promise.all(stopTensorBoardTasks);
    }

    public stopTensorBoard(endPoint: string): Promise<void> {
        const host: string = this.getEndPointHost(endPoint);

        return this.stopHostTensorBoard(host);
    }

    private stopHostTensorBoard(host: string): Promise<void> {
        const jobId: string | undefined = this.tbJobMap.get(host);
        if (jobId === undefined) {
            return Promise.resolve();
        }

        return this.trainingService.cancelTrialJob(jobId);
    }

    private async isTensorBoardRunningOnHost(host: string): Promise<boolean> {
        const jobId: string | undefined = this.tbJobMap.get(host);
        if (jobId === undefined) {
            return false;
        }

        const status: TrialJobStatus = (await this.trainingService.getTrialJob(jobId)).status;

        return ['RUNNING', 'WAITING'].includes(status);
    }

    private async getJobHost(trialJobIds: string[]): Promise<string> {
        if (trialJobIds === undefined || trialJobIds.length < 1) {
            throw new Error('No trail job specified.');
        }
        const jobInfo: TrialJobInfo = await this.dataStore.getTrialJob(trialJobIds[0]);
        const logPath: string | undefined = jobInfo.logPath;
        if (logPath === undefined) {
            throw new Error(`Failed to find job logPath: ${jobInfo.id}`);
        }

        return logPath.split('://')[1].split(':')[0]; //TODO use url parse
    }

    private async getLogDir(trialJobId: string): Promise<string> {
        const jobInfo: TrialJobInfo = await this.dataStore.getTrialJob(trialJobId);
        const logPath: string | undefined = jobInfo.logPath;
        if (logPath === undefined) {
            throw new Error(`Failed to find job logPath: ${jobInfo.id}`);
        }

        return logPath.split('://')[1].split(':')[1]; //TODO use url parse
    }

    private getEndPointHost(endPoint: string): string {
        const parts = endPoint.match(/.*:\/\/(.*):(.*)/);
        if (parts !== null) {
            return parts[1];
        } else {
            throw new Error(`Invalid endPoint: ${endPoint}`);
        }
    }
}
