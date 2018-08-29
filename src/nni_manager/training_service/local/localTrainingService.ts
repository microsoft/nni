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

import * as cpp from 'child-process-promise';
import * as cp from 'child_process';
import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as path from 'path';
import * as ts from 'tail-stream';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { getLogger, Logger } from '../../common/log';
import { TrialConfig } from '../common/trialConfig';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import {
    HostJobApplicationForm, JobApplicationForm, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric, TrialJobStatus
} from '../../common/trainingService';
import { delay, getExperimentRootDir, uniqueString } from '../../common/utils';

const tkill = require('tree-kill');

/**
 * Decode a command
 * @param Buffer binary incoming data
 * @returns a tuple of (success, commandType, content, remain)
 *          success: true if the buffer contains at least one complete command; otherwise false
 *          remain: remaining data after the first command
 */
function decodeCommand(data: Buffer): [boolean, string, string, Buffer] {
    if (data.length < 8) {
        return [false, '', '', data];
    }
    const commandType: string = data.slice(0, 2).toString();
    const contentLength: number = parseInt(data.slice(2, 8).toString(), 10);
    if (data.length < contentLength + 8) {
        return [false, '', '', data];
    }
    const content: string = data.slice(8, contentLength + 8).toString();
    const remain: Buffer = data.slice(contentLength + 8);

    return [true, commandType, content, remain];
}

/**
 * LocalTrialJobDetail
 */
class LocalTrialJobDetail implements TrialJobDetail {
    public id: string;
    public status: TrialJobStatus;
    public submitTime: Date;
    public startTime?: Date;
    public endTime?: Date;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: JobApplicationForm;
    public pid?: number;

    constructor(id: string, status: TrialJobStatus, submitTime: Date, workingDirectory: string, form: JobApplicationForm) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.url = `file://localhost:${workingDirectory}`;
    }
}

/**
 * Local training service
 */
class LocalTrainingService implements TrainingService {
    private eventEmitter: EventEmitter;
    private jobMap: Map<string, LocalTrialJobDetail>;
    private jobQueue: string[];
    private initialized: boolean;
    private stopping: boolean;
    private rootDir!: string;
    protected log: Logger;
    protected localTrailConfig?: TrialConfig;

    constructor() {
        this.eventEmitter = new EventEmitter();
        this.jobMap = new Map<string, LocalTrialJobDetail>();
        this.jobQueue = [];
        this.initialized = false;
        this.stopping = false;
        this.log = getLogger();
    }

    public async run(): Promise<void> {
        while (!this.stopping) {
            while (this.jobQueue.length !== 0) {
                const trialJobId: string = this.jobQueue[0];
                const [success, resource] = this.tryGetAvailableResource();
                if (!success) {
                    break;
                }
                this.occupyResource(resource);
                this.jobQueue.shift();
                await this.runTrialJob(trialJobId, resource);
            }
            await delay(5000);
        }
    }

    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        const jobs: TrialJobDetail[] = [];
        for (const key of this.jobMap.keys()) {
            const trialJob: TrialJobDetail = await this.getTrialJob(key);
            if (trialJob.form.jobType === 'TRIAL') {
                jobs.push(trialJob);
            }
        }

        return jobs;
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        const trialJob: LocalTrialJobDetail | undefined = this.jobMap.get(trialJobId);
        if (trialJob === undefined) {
            throw new NNIError(NNIErrorNames.NOT_FOUND, 'Trial job not found');
        }
        if (trialJob.form.jobType === 'HOST') {
            return this.getHostJob(trialJobId);
        }
        if (trialJob.status === 'RUNNING') {
            let alive: boolean = false;
            try {
                await cpp.exec(`kill -0 ${trialJob.pid}`);
                alive = true;
            } catch (error) {
                //ignore
            }

            if (!alive) {
                trialJob.endTime = new Date();
                this.setTrialJobStatus(trialJob, 'FAILED');
                try {
                    const state: string = await fs.promises.readFile(path.join(trialJob.workingDirectory, '.nni', 'state'), 'utf8');
                    const match: RegExpMatchArray | null = state.trim().match(/^(\d+)\s+(\d+)$/);
                    if (match !== null) {
                        const { 1: code, 2: timestamp } = match;
                        if (parseInt(code, 10) === 0) {
                            this.setTrialJobStatus(trialJob, 'SUCCEEDED');
                        }
                        trialJob.endTime = new Date(parseInt(timestamp, 10));
                    }
                } catch (error) {
                    //ignore
                }
                this.log.info(`trailJob status update: ${trialJobId}, ${trialJob.status}`);
            }
        }

        return trialJob;
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.eventEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.eventEmitter.off('metric', listener);
    }

    public submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        this.log.info(`submitTrialJob: form: ${JSON.stringify(form)}`);
        if (form.jobType === 'HOST') {
            return this.runHostJob(<HostJobApplicationForm>form);
        } else if (form.jobType === 'TRIAL') {
            const trialJobId: string = uniqueString(5);
            const trialJobDetail: LocalTrialJobDetail = new LocalTrialJobDetail(
                trialJobId,
                'WAITING',
                new Date(),
                path.join(this.rootDir, 'trials', trialJobId),
                form);
            this.jobQueue.push(trialJobId);
            this.jobMap.set(trialJobId, trialJobDetail);

            this.log.debug(`submitTrialJob: return: ${JSON.stringify(trialJobDetail)} `);

            return Promise.resolve(trialJobDetail);
        } else {
            return Promise.reject(new Error(`Job form not supported: ${JSON.stringify(form)}`));
        }
    }

    public async cancelTrialJob(trialJobId: string): Promise<void> {
        this.log.info(`cancelTrialJob: ${trialJobId}`);
        const trialJob: LocalTrialJobDetail | undefined = this.jobMap.get(trialJobId);
        if (trialJob === undefined) {
            throw new NNIError(NNIErrorNames.NOT_FOUND, 'Trial job not found');
        }
        if (trialJob.form.jobType === 'TRIAL') {
            await tkill(trialJob.pid, 'SIGKILL');
        } else if (trialJob.form.jobType === 'HOST') {
            await cpp.exec(`pkill -9 -P ${trialJob.pid}`);
        } else {
            throw new Error(`Job type not supported: ${trialJob.form.jobType}`);
        }
        this.setTrialJobStatus(trialJob, 'USER_CANCELED');
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        if (!this.initialized) {
            this.rootDir = getExperimentRootDir();
            await cpp.exec(`mkdir -p ${this.rootDir}`);
            this.initialized = true;
        }
        switch (key) {
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                this.localTrailConfig = <TrialConfig>JSON.parse(value);
                // Parse trial config failed, throw Error
                if (!this.localTrailConfig) {
                    throw new Error('trial config parsed failed');
                }
                break;
            default:
        }
    }

    public getClusterMetadata(key: string): Promise<string> {
        switch (key) {
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                let getResult : Promise<string>;
                if(!this.localTrailConfig) {
                    getResult = Promise.reject(new NNIError(NNIErrorNames.NOT_FOUND, `${key} is never set yet`));
                } else {
                    getResult = Promise.resolve(!this.localTrailConfig? '' : JSON.stringify(this.localTrailConfig));
                }
                return getResult;     
            default:
                return Promise.reject(new NNIError(NNIErrorNames.NOT_FOUND, 'Key not found'));
        }
    }

    public cleanUp(): Promise<void> {
        this.stopping = true;

        return Promise.resolve();
    }

    protected onTrialJobStatusChanged(trialJob: TrialJobDetail, oldStatus: TrialJobStatus): void {
        //abstract
    }

    protected getEnvironmentVariables(trialJobDetail: TrialJobDetail, _: {}): { key: string; value: string }[] {
        return [
            { key: 'NNI_PLATFORM', value: 'local' },
            { key: 'NNI_SYS_DIR', value: trialJobDetail.workingDirectory },
            { key: 'NNI_TRIAL_JOB_ID', value: trialJobDetail.id },
            { key: 'NNI_OUTPUT_DIR', value: trialJobDetail.workingDirectory }
        ];
    }

    protected setExtraProperties(trialJobDetail: TrialJobDetail, resource: {}): void {
        //abstract
    }

    protected tryGetAvailableResource(): [boolean, {}] {
        return [true, {}];
    }

    protected occupyResource(_: {}): void {
        //abstract
    }

    private setTrialJobStatus(trialJob: LocalTrialJobDetail, newStatus: TrialJobStatus): void {
        if (trialJob.status !== newStatus) {
            const oldStatus: TrialJobStatus = trialJob.status;
            trialJob.status = newStatus;
            this.onTrialJobStatusChanged(trialJob, oldStatus);
        }
    }

    private async runTrialJob(trialJobId: string, resource: {}): Promise<void> {
        const trialJobDetail: LocalTrialJobDetail = <LocalTrialJobDetail>this.jobMap.get(trialJobId);
        const variables: { key: string; value: string }[] = this.getEnvironmentVariables(trialJobDetail, resource);

        const runScriptLines: string[] = [];

        if (!this.localTrailConfig) {
            throw new Error('trial config is not initialized');
        }
        runScriptLines.push(
            '#!/bin/bash',
            `cd ${this.localTrailConfig.codeDir}`);
        for (const variable of variables) {
            runScriptLines.push(`export ${variable.key}=${variable.value}`);
        }
        runScriptLines.push(
            `eval ${this.localTrailConfig.command} 2>${path.join(trialJobDetail.workingDirectory, '.nni', 'stderr')}`,
            `echo $? \`date +%s%3N\` >${path.join(trialJobDetail.workingDirectory, '.nni', 'state')}`);

        await cpp.exec(`mkdir -p ${trialJobDetail.workingDirectory}`);
        await cpp.exec(`mkdir -p ${path.join(trialJobDetail.workingDirectory, '.nni')}`);
        await cpp.exec(`touch ${path.join(trialJobDetail.workingDirectory, '.nni', 'metrics')}`);
        await fs.promises.writeFile(path.join(trialJobDetail.workingDirectory, 'run.sh'), runScriptLines.join('\n'), { encoding: 'utf8' });
        await fs.promises.writeFile(
            path.join(trialJobDetail.workingDirectory, 'parameter.cfg'),
            (<TrialJobApplicationForm>trialJobDetail.form).hyperParameters,
            { encoding: 'utf8' });
        const process: cp.ChildProcess = cp.exec(`bash ${path.join(trialJobDetail.workingDirectory, 'run.sh')}`);

        this.setTrialJobStatus(trialJobDetail, 'RUNNING');
        trialJobDetail.startTime = new Date();
        trialJobDetail.pid = process.pid;
        this.setExtraProperties(trialJobDetail, resource);

        let buffer: Buffer = Buffer.alloc(0);
        const stream: ts.Stream = ts.createReadStream(path.join(trialJobDetail.workingDirectory, '.nni', 'metrics'));
        stream.on('data', (data: Buffer) => {
            buffer = Buffer.concat([buffer, data]);
            while (buffer.length > 0) {
                const [success, , content, remain] = decodeCommand(buffer);
                if (!success) {
                    break;
                }
                this.eventEmitter.emit('metric', {
                    id: trialJobDetail.id,
                    data: content
                });
                this.log.debug(`Sending metrics, job id: ${trialJobDetail.id}, metrics: ${content}`);
                buffer = remain;
            }
        });
    }

    private async runHostJob(form: HostJobApplicationForm): Promise<TrialJobDetail> {
        const jobId: string = uniqueString(5);
        const workDir: string = path.join(this.rootDir, 'hostjobs', jobId);
        await cpp.exec(`mkdir -p ${workDir}`);
        const wrappedCmd: string = `cd ${workDir} && ${form.cmd}>stdout 2>stderr`;
        this.log.debug(`runHostJob: command: ${wrappedCmd}`);
        const process: cp.ChildProcess = cp.exec(wrappedCmd);
        const jobDetail: LocalTrialJobDetail = {
            id: jobId,
            status: 'RUNNING',
            submitTime: new Date(),
            workingDirectory: workDir,
            form: form,
            pid: process.pid
        };
        this.jobMap.set(jobId, jobDetail);
        this.log.debug(`runHostJob: return: ${JSON.stringify(jobDetail)} `);

        return jobDetail;
    }

    private async getHostJob(jobId: string): Promise<TrialJobDetail> {
        const jobDetail: LocalTrialJobDetail | undefined = this.jobMap.get(jobId);
        if (jobDetail === undefined) {
            throw new NNIError(NNIErrorNames.NOT_FOUND, `Host Job not found: ${jobId}`);
        }
        try {
            await cpp.exec(`kill -0 ${jobDetail.pid}`);

            return jobDetail;
        } catch (error) {
            if (error instanceof Error) {
                this.log.debug(`getHostJob: error: ${error.message}`);
                this.jobMap.delete(jobId);
                throw new NNIError(NNIErrorNames.NOT_FOUND, `Host Job not found: ${error.message}`);
            } else {
                throw error;
            }
        }
    }
}

export { LocalTrainingService };
