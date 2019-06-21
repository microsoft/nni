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
import * as tkill from 'tree-kill';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { getInitTrialSequenceId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import {
    HostJobApplicationForm, HyperParameters, JobApplicationForm, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric, TrialJobStatus
} from '../../common/trainingService';
import {
    delay, generateParamFileName, getExperimentRootDir, getJobCancelStatus, getNewLine, isAlive, uniqueString
} from '../../common/utils';
import { TrialConfig } from '../common/trialConfig';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { execMkdir, execNewFile, getScriptName, runScript, setEnvironmentVariable } from '../common/util';
import { GPUScheduler } from './gpuScheduler';

/**
 * Decode a command
 * @param Buffer binary incoming data
 * @returns a tuple of (success, commandType, content, remain)
 *          success: true if the buffer contains at least one complete command; otherwise false
 *          remain: remaining data after the first command
 */
// tslint:disable:newline-per-chained-call informative-docs
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
// tslint:enable:newline-per-chained-call informative-docs

/**
 * LocalTrialJobDetail
 */
class LocalTrialJobDetail implements TrialJobDetail {
    public id: string;
    public status: TrialJobStatus;
    public submitTime: number;
    public startTime?: number;
    public endTime?: number;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: JobApplicationForm;
    public sequenceId: number;
    public pid?: number;
    public gpuIndices?: number[];

    constructor(
        id: string, status: TrialJobStatus, submitTime: number,
        workingDirectory: string, form: JobApplicationForm, sequenceId: number) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.url = `file://localhost:${workingDirectory}`;
        this.sequenceId = sequenceId;
        this.gpuIndices = [];
    }
}

/**
 * Local training service config
 */
class LocalConfig {
    public maxTrialNumPerGpu?: number;
    public gpuIndices?: string;
    public useActiveGpu?: boolean;
    constructor(gpuIndices?: string, maxTrialNumPerGpu?: number, useActiveGpu?: boolean) {
        if (gpuIndices !== undefined) {
            this.gpuIndices = gpuIndices;
        }
        if (maxTrialNumPerGpu !== undefined) {
            this.maxTrialNumPerGpu = maxTrialNumPerGpu;
        }
        if (useActiveGpu !== undefined) {
            this.useActiveGpu = useActiveGpu;
        }
    }
}

/**
 * Local machine training service
 */
class LocalTrainingService implements TrainingService {
    private readonly eventEmitter: EventEmitter;
    private readonly jobMap: Map<string, LocalTrialJobDetail>;
    private readonly jobQueue: string[];
    private initialized: boolean;
    private stopping: boolean;
    private rootDir!: string;
    private trialSequenceId: number;
    private gpuScheduler!: GPUScheduler;
    private readonly occupiedGpuIndexNumMap: Map<number, number>;
    private designatedGpuIndices!: Set<number>;
    private readonly log: Logger;
    private localTrailConfig?: TrialConfig;
    private localConfig?: LocalConfig;
    private isMultiPhase: boolean;
    private readonly jobStreamMap: Map<string, ts.Stream>;
    private maxTrialNumPerGpu: number;
    private useActiveGpu: boolean;

    constructor() {
        this.eventEmitter = new EventEmitter();
        this.jobMap = new Map<string, LocalTrialJobDetail>();
        this.jobQueue = [];
        this.initialized = false;
        this.stopping = false;
        this.log = getLogger();
        this.trialSequenceId = -1;
        this.jobStreamMap = new Map<string, ts.Stream>();
        this.log.info('Construct local machine training service.');
        this.occupiedGpuIndexNumMap = new Map<number, number>();
        this.maxTrialNumPerGpu = 1;
        this.useActiveGpu = false;
        this.isMultiPhase = false;
    }

    public async run(): Promise<void> {
        this.log.info('Run local machine training service.');
        const longRunningTasks: Promise<void>[] = [this.runJobLoop()];
        if (this.gpuScheduler !== undefined) {
            longRunningTasks.push(this.gpuScheduler.run());
        }
        await Promise.all(longRunningTasks);
        this.log.info('Local machine training service exit.');
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
            const alive: boolean = await isAlive(trialJob.pid);
            if (!alive) {
                trialJob.endTime = Date.now();
                this.setTrialJobStatus(trialJob, 'FAILED');
                try {
                    const state: string = await fs.promises.readFile(path.join(trialJob.workingDirectory, '.nni', 'state'), 'utf8');
                    const match: RegExpMatchArray | null = state.trim()
                        .match(/^(\d+)\s+(\d+)/);
                    if (match !== null) {
                        const { 1: code, 2: timestamp } = match;
                        if (parseInt(code, 10) === 0) {
                            this.setTrialJobStatus(trialJob, 'SUCCEEDED');
                        }
                        trialJob.endTime = parseInt(timestamp, 10);
                    }
                } catch (error) {
                    //ignore
                }
                this.log.debug(`trailJob status update: ${trialJobId}, ${trialJob.status}`);
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
        if (form.jobType === 'HOST') {
            return this.runHostJob(<HostJobApplicationForm>form);
        } else if (form.jobType === 'TRIAL') {
            const trialJobId: string = uniqueString(5);
            const trialJobDetail: LocalTrialJobDetail = new LocalTrialJobDetail(
                trialJobId,
                'WAITING',
                Date.now(),
                path.join(this.rootDir, 'trials', trialJobId),
                form,
                this.generateSequenceId()
            );
            this.jobQueue.push(trialJobId);
            this.jobMap.set(trialJobId, trialJobDetail);

            this.log.debug(`submitTrialJob: return: ${JSON.stringify(trialJobDetail)} `);

            return Promise.resolve(trialJobDetail);
        } else {
            return Promise.reject(new Error(`Job form not supported: ${JSON.stringify(form)}`));
        }
    }

    /**
     * Update trial job for multi-phase
     * @param trialJobId trial job id
     * @param form job application form
     */
    public async updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<TrialJobDetail> {
        const trialJobDetail: undefined | TrialJobDetail = this.jobMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }
        if (form.jobType === 'TRIAL') {
            await this.writeParameterFile(trialJobDetail.workingDirectory, (<TrialJobApplicationForm>form).hyperParameters);
        } else {
            throw new Error(`updateTrialJob failed: jobType ${form.jobType} not supported.`);
        }

        return trialJobDetail;
    }

    /**
     * Is multiphase job supported in current training service
     */
    public get isMultiPhaseJobSupported(): boolean {
        return true;
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trialJob: LocalTrialJobDetail | undefined = this.jobMap.get(trialJobId);
        if (trialJob === undefined) {
            throw new NNIError(NNIErrorNames.NOT_FOUND, 'Trial job not found');
        }
        if (trialJob.pid === undefined) {
            this.setTrialJobStatus(trialJob, 'USER_CANCELED');

            return Promise.resolve();
        }
        if (trialJob.form.jobType === 'TRIAL') {
            tkill(trialJob.pid, 'SIGKILL');
        } else if (trialJob.form.jobType === 'HOST') {
            await cpp.exec(`pkill -9 -P ${trialJob.pid}`);
        } else {
            throw new Error(`Job type not supported: ${trialJob.form.jobType}`);
        }
        this.setTrialJobStatus(trialJob, getJobCancelStatus(isEarlyStopped));

        return Promise.resolve();
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        if (!this.initialized) {
            this.rootDir = getExperimentRootDir();
            // tslint:disable-next-line:non-literal-fs-path
            if (!fs.existsSync(this.rootDir)) {
                await cpp.exec(`powershell.exe mkdir ${this.rootDir}`);
            }
            this.initialized = true;
        }
        switch (key) {
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                this.localTrailConfig = <TrialConfig>JSON.parse(value);
                // Parse trial config failed, throw Error
                if (this.localTrailConfig === undefined) {
                    throw new Error('trial config parsed failed');
                }
                this.log.info(`required GPU number is ${this.localTrailConfig.gpuNum}`);
                if (this.gpuScheduler === undefined && this.localTrailConfig.gpuNum > 0) {
                    this.gpuScheduler = new GPUScheduler();
                }
                break;
            case TrialConfigMetadataKey.LOCAL_CONFIG:
                this.localConfig = <LocalConfig>JSON.parse(value);
                this.log.info(`Specified GPU indices: ${this.localConfig.gpuIndices}`);
                if (this.localConfig.gpuIndices !== undefined) {
                    this.designatedGpuIndices = new Set(this.localConfig.gpuIndices.split(',')
                            .map((x: string) => parseInt(x, 10)));
                    if (this.designatedGpuIndices.size === 0) {
                        throw new Error('gpuIndices can not be empty if specified.');
                    }
                }
                if (this.localConfig.maxTrialNumPerGpu !== undefined) {
                    this.maxTrialNumPerGpu = this.localConfig.maxTrialNumPerGpu;
                }

                if (this.localConfig.useActiveGpu !== undefined) {
                    this.useActiveGpu = this.localConfig.useActiveGpu;
                }
                break;
            case TrialConfigMetadataKey.MULTI_PHASE:
                this.isMultiPhase = (value === 'true' || value === 'True');
                break;
            default:
        }
    }

    public getClusterMetadata(key: string): Promise<string> {
        switch (key) {
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                let getResult: Promise<string>;
                if (this.localTrailConfig === undefined) {
                    getResult = Promise.reject(new NNIError(NNIErrorNames.NOT_FOUND, `${key} is never set yet`));
                } else {
                    getResult = Promise.resolve(JSON.stringify(this.localTrailConfig));
                }

                return getResult;
            default:
                return Promise.reject(new NNIError(NNIErrorNames.NOT_FOUND, 'Key not found'));
        }
    }

    public async cleanUp(): Promise<void> {
        this.log.info('Stopping local machine training service...');
        this.stopping = true;
        for (const stream of this.jobStreamMap.values()) {
            stream.destroy();
        }
        if (this.gpuScheduler !== undefined) {
            await this.gpuScheduler.stop();
        }

        return Promise.resolve();
    }

    private onTrialJobStatusChanged(trialJob: LocalTrialJobDetail, oldStatus: TrialJobStatus): void {
        //if job is not running, destory job stream
        if (['SUCCEEDED', 'FAILED', 'USER_CANCELED', 'SYS_CANCELED', 'EARLY_STOPPED'].includes(trialJob.status)) {
            if (this.jobStreamMap.has(trialJob.id)) {
                const stream: ts.Stream | undefined = this.jobStreamMap.get(trialJob.id);
                if (stream === undefined) {
                    throw new Error(`Could not find stream in trial ${trialJob.id}`);
                }
                stream.destroy();
                this.jobStreamMap.delete(trialJob.id);
            }
        }
        if (trialJob.gpuIndices !== undefined && trialJob.gpuIndices.length > 0 && this.gpuScheduler !== undefined) {
            if (oldStatus === 'RUNNING' && trialJob.status !== 'RUNNING') {
                for (const index of trialJob.gpuIndices) {
                    const num: number | undefined = this.occupiedGpuIndexNumMap.get(index);
                    if (num === undefined) {
                        throw new Error(`gpu resource schedule error`);
                    } else if (num === 1) {
                        this.occupiedGpuIndexNumMap.delete(index);
                    } else {
                        this.occupiedGpuIndexNumMap.set(index, num - 1);
                    }
                }
            }
        }
    }

    private getEnvironmentVariables(
        trialJobDetail: TrialJobDetail,
        resource: { gpuIndices: number[] }): { key: string; value: string }[] {
        const envVariables: { key: string; value: string }[] = [
            { key: 'NNI_PLATFORM', value: 'local' },
            { key: 'NNI_SYS_DIR', value: trialJobDetail.workingDirectory },
            { key: 'NNI_TRIAL_JOB_ID', value: trialJobDetail.id },
            { key: 'NNI_OUTPUT_DIR', value: trialJobDetail.workingDirectory },
            { key: 'NNI_TRIAL_SEQ_ID', value: trialJobDetail.sequenceId.toString() },
            { key: 'MULTI_PHASE', value: this.isMultiPhase.toString() }
        ];

        envVariables.push({
            key: 'CUDA_VISIBLE_DEVICES',
            value: this.gpuScheduler === undefined ? '-1' : resource.gpuIndices.join(',')
        });

        return envVariables;
    }

    private setExtraProperties(trialJobDetail: LocalTrialJobDetail, resource: { gpuIndices: number[] }): void {
        trialJobDetail.gpuIndices = resource.gpuIndices;
    }

    private tryGetAvailableResource(): [boolean, { gpuIndices: number[]}] {
        if (this.localTrailConfig === undefined) {
            throw new Error('localTrailConfig is not initialized!');
        }

        const resource: { gpuIndices: number[] } = { gpuIndices: [] };
        if (this.gpuScheduler === undefined) {
            return [true, resource];
        }

        let selectedGPUIndices: number[] = [];
        const availableGpuIndices: number[] = this.gpuScheduler.getAvailableGPUIndices(this.useActiveGpu, this.occupiedGpuIndexNumMap);
        for (const index of availableGpuIndices) {
            const num: number | undefined = this.occupiedGpuIndexNumMap.get(index);
            if (num === undefined || num < this.maxTrialNumPerGpu) {
                selectedGPUIndices.push(index);
            }
        }

        if (this.designatedGpuIndices !== undefined) {
            this.checkSpecifiedGpuIndices();
            selectedGPUIndices = selectedGPUIndices.filter((index: number) => this.designatedGpuIndices.has(index));
        }

        if (selectedGPUIndices.length < this.localTrailConfig.gpuNum) {
            return [false, resource];
        }

        selectedGPUIndices.splice(this.localTrailConfig.gpuNum);
        Object.assign(resource, { gpuIndices: selectedGPUIndices });

        return [true, resource];
    }

    private checkSpecifiedGpuIndices(): void {
        const gpuCount: number = this.gpuScheduler.getSystemGpuCount();
        if (this.designatedGpuIndices !== undefined) {
            for (const index of this.designatedGpuIndices) {
                if (index >= gpuCount) {
                    throw new Error(`Specified GPU index not found: ${index}`);
                }
            }
        }
    }

    private occupyResource(resource: {gpuIndices: number[]}): void {
        if (this.gpuScheduler !== undefined) {
            for (const index of resource.gpuIndices) {
                const num: number | undefined = this.occupiedGpuIndexNumMap.get(index);
                if (num === undefined) {
                    this.occupiedGpuIndexNumMap.set(index, 1);
                } else {
                    this.occupiedGpuIndexNumMap.set(index, num + 1);
                }
            }
        }
    }

    private async runJobLoop(): Promise<void> {
        while (!this.stopping) {
            while (!this.stopping && this.jobQueue.length !== 0) {
                const trialJobId: string = this.jobQueue[0];
                const trialJobDeatil: LocalTrialJobDetail | undefined = this.jobMap.get(trialJobId);
                if (trialJobDeatil !== undefined && trialJobDeatil.status === 'WAITING') {
                    const [success, resource] = this.tryGetAvailableResource();
                    if (!success) {
                        break;
                    }
                    this.occupyResource(resource);
                    await this.runTrialJob(trialJobId, resource);
                }
                this.jobQueue.shift();
            }
            await delay(5000);
        }
    }

    private setTrialJobStatus(trialJob: LocalTrialJobDetail, newStatus: TrialJobStatus): void {
        if (trialJob.status !== newStatus) {
            const oldStatus: TrialJobStatus = trialJob.status;
            trialJob.status = newStatus;
            this.onTrialJobStatusChanged(trialJob, oldStatus);
        }
    }

    private getScript(localTrailConfig: TrialConfig, workingDirectory: string): string[] {
        const script: string[] = [];
        if (process.platform === 'win32') {
            script.push(
                `cmd /c ${localTrailConfig.command} 2>${path.join(workingDirectory, 'stderr')}`,
                `$NOW_DATE = [int64](([datetime]::UtcNow)-(get-date "1/1/1970")).TotalSeconds`,
                `$NOW_DATE = "$NOW_DATE" + "000"`,
                `Write $LASTEXITCODE " " $NOW_DATE  | Out-File ${path.join(workingDirectory, '.nni', 'state')} -NoNewline -encoding utf8`);
        } else {
            script.push(
                `eval ${localTrailConfig.command} 2>${path.join(workingDirectory, 'stderr')}`,
                `echo $? \`date +%s000\` >${path.join(workingDirectory, '.nni', 'state')}`);
        }

        return script;
    }

    private async runTrialJob(trialJobId: string, resource: {gpuIndices: number[]}): Promise<void> {
        const trialJobDetail: LocalTrialJobDetail = <LocalTrialJobDetail>this.jobMap.get(trialJobId);
        const variables: { key: string; value: string }[] = this.getEnvironmentVariables(trialJobDetail, resource);

        if (this.localTrailConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        const runScriptContent: string[] = [];
        if (process.platform !== 'win32') {
            runScriptContent.push('#!/bin/bash');
        }
        runScriptContent.push(`cd ${this.localTrailConfig.codeDir}`);
        for (const variable of variables) {
            runScriptContent.push(setEnvironmentVariable(variable));
        }
        const scripts: string[] = this.getScript(this.localTrailConfig, trialJobDetail.workingDirectory);
        scripts.forEach((script: string) => {
            runScriptContent.push(script);
        });
        await execMkdir(trialJobDetail.workingDirectory);
        await execMkdir(path.join(trialJobDetail.workingDirectory, '.nni'));
        await execNewFile(path.join(trialJobDetail.workingDirectory, '.nni', 'metrics'));
        const scriptName: string = getScriptName('run');
        await fs.promises.writeFile(path.join(trialJobDetail.workingDirectory, scriptName),
                                    runScriptContent.join(getNewLine()), { encoding: 'utf8', mode: 0o777 });
        await this.writeParameterFile(trialJobDetail.workingDirectory, (<TrialJobApplicationForm>trialJobDetail.form).hyperParameters);
        const trialJobProcess: cp.ChildProcess = runScript(path.join(trialJobDetail.workingDirectory, scriptName));
        this.setTrialJobStatus(trialJobDetail, 'RUNNING');
        trialJobDetail.startTime = Date.now();
        trialJobDetail.pid = trialJobProcess.pid;
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

        this.jobStreamMap.set(trialJobDetail.id, stream);
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
            submitTime: Date.now(),
            workingDirectory: workDir,
            form: form,
            sequenceId: this.generateSequenceId(),
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

    private async writeParameterFile(directory: string, hyperParameters: HyperParameters): Promise<void> {
        const filepath: string = path.join(directory, generateParamFileName(hyperParameters));
        await fs.promises.writeFile(filepath, hyperParameters.value, { encoding: 'utf8' });
    }

    private generateSequenceId(): number {
        if (this.trialSequenceId === -1) {
            this.trialSequenceId = getInitTrialSequenceId();
        }

        return this.trialSequenceId++;
    }
}

export { LocalTrainingService };
