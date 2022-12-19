// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import cp from 'child_process';
import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import ts from 'tail-stream';
import tkill from 'tree-kill';
import { NNIError, NNIErrorNames } from 'common/errors';
import { getExperimentId } from 'common/experimentStartupInfo';
import { getLogger, Logger } from 'common/log';
import { powershellString, shellString, createScriptFile } from 'common/shellUtils';
import {
    HyperParameters, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric, TrialJobStatus
} from 'common/trainingService';
import {
    delay, generateParamFileName, getExperimentRootDir, getJobCancelStatus, getNewLine, isAlive, uniqueString
} from 'common/utils';
import { LocalConfig } from 'common/experimentConfig';
import { execMkdir, execNewFile, getScriptName, runScript, setEnvironmentVariable } from '../common/util';
import { GPUScheduler } from './gpuScheduler';

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
    public submitTime: number;
    public startTime?: number;
    public endTime?: number;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: TrialJobApplicationForm;
    public pid?: number;
    public gpuIndices?: number[];

    constructor(
        id: string, status: TrialJobStatus, submitTime: number,
        workingDirectory: string, form: TrialJobApplicationForm) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.url = `file://localhost:${workingDirectory}`;
        this.gpuIndices = [];
    }
}

/**
 * Local machine training service
 */
class LocalTrainingService implements TrainingService {
    private readonly config: LocalConfig;
    private readonly eventEmitter: EventEmitter;
    private readonly jobMap: Map<string, LocalTrialJobDetail>;
    private readonly jobQueue: string[];
    private initialized: boolean;
    private stopping: boolean;
    private rootDir!: string;
    private readonly experimentId!: string;
    private gpuScheduler!: GPUScheduler;
    private readonly occupiedGpuIndexNumMap: Map<number, number>;
    private readonly log: Logger;
    private readonly jobStreamMap: Map<string, ts.Stream>;

    constructor(config: LocalConfig) {
        this.config = config;
        this.eventEmitter = new EventEmitter();
        this.jobMap = new Map<string, LocalTrialJobDetail>();
        this.jobQueue = [];
        this.stopping = false;
        this.log = getLogger('LocalTrainingService');
        this.experimentId = getExperimentId();
        this.jobStreamMap = new Map<string, ts.Stream>();
        this.log.info('Construct local machine training service.');
        this.occupiedGpuIndexNumMap = new Map<number, number>();

        if (this.config.trialGpuNumber !== undefined && this.config.trialGpuNumber > 0) {
            this.gpuScheduler = new GPUScheduler();
        }

        if (this.config.gpuIndices && this.config.gpuIndices.length === 0) {
            throw new Error('gpuIndices cannot be empty when specified.');
        }

        this.rootDir = getExperimentRootDir();
        if (!fs.existsSync(this.rootDir)) {
            throw new Error('root dir not created');
        }
        this.initialized = true;
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
            jobs.push(trialJob);
        }

        return jobs;
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        const trialJob: LocalTrialJobDetail | undefined = this.jobMap.get(trialJobId);
        if (trialJob === undefined) {
            throw new NNIError(NNIErrorNames.NOT_FOUND, 'Trial job not found');
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
                this.log.debug(`trialJob status update: ${trialJobId}, ${trialJob.status}`);
            }
        }

        return trialJob;
    }

    public async getTrialFile(trialJobId: string, fileName: string): Promise<string | Buffer> {
        // check filename here for security
        if (!['trial.log', 'stderr', 'model.onnx', 'stdout'].includes(fileName)) {
            throw new Error(`File unaccessible: ${fileName}`);
        }
        let encoding: string | null = null;
        if (!fileName.includes('.') || fileName.match(/.*\.(txt|log)/g)) {
            encoding = 'utf8';
        }
        const logPath = path.join(this.rootDir, 'trials', trialJobId, fileName);
        if (!fs.existsSync(logPath)) {
            throw new Error(`File not found: ${logPath}`);
        }
        return fs.promises.readFile(logPath, {encoding: encoding as any});
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.eventEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.eventEmitter.off('metric', listener);
    }

    public submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const trialJobId: string = form.id === undefined ? uniqueString(5) : form.id;
        const trialJobDetail: LocalTrialJobDetail = new LocalTrialJobDetail(
            trialJobId,
            'WAITING',
            Date.now(),
            path.join(this.rootDir, 'trials', trialJobId),
            form
        );
        this.jobQueue.push(trialJobId);
        this.jobMap.set(trialJobId, trialJobDetail);

        this.log.debug('submitTrialJob: return:',  trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    /**
     * Update trial job for multi-phase
     * @param trialJobId trial job id
     * @param form job application form
     */
    public async updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const trialJobDetail: undefined | TrialJobDetail = this.jobMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }
        await this.writeParameterFile(trialJobDetail.workingDirectory, form.hyperParameters);

        return trialJobDetail;
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
        tkill(trialJob.pid, 'SIGTERM');
        this.setTrialJobStatus(trialJob, getJobCancelStatus(isEarlyStopped));

        const startTime = Date.now();
        while(await isAlive(trialJob.pid)) {
            if (Date.now() - startTime > 4999) {
                tkill(trialJob.pid, 'SIGKILL', (err) => {
                    if (err) {
                        this.log.error(`kill trial job error: ${err}`);
                    }
                });
                break;
            }
            await delay(500);
        }

        return Promise.resolve();
    }

    public async setClusterMetadata(_key: string, _value: string): Promise<void> { return; }
    public async getClusterMetadata(_key: string): Promise<string> { return ''; }

    public async cleanUp(): Promise<void> {
        this.log.info('Stopping local machine training service...');
        this.stopping = true;
        for (const stream of this.jobStreamMap.values()) {
            stream.end(0);
            stream.emit('end');
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
                //Refer https://github.com/Juul/tail-stream/issues/20
                setTimeout(() => {
                    stream.end(0);
                    stream.emit('end');
                    this.jobStreamMap.delete(trialJob.id);
                }, 5000);
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
        resource: { gpuIndices: number[] },
        gpuNum: number | undefined): { key: string; value: string }[] {
            const envVariables: { key: string; value: string }[] = [
                { key: 'NNI_PLATFORM', value: 'local' },
                { key: 'NNI_EXP_ID', value: this.experimentId },
                { key: 'NNI_SYS_DIR', value: trialJobDetail.workingDirectory },
                { key: 'NNI_TRIAL_JOB_ID', value: trialJobDetail.id },
                { key: 'NNI_OUTPUT_DIR', value: trialJobDetail.workingDirectory },
                { key: 'NNI_TRIAL_SEQ_ID', value: trialJobDetail.form.sequenceId.toString() },
                { key: 'NNI_CODE_DIR', value: this.config.trialCodeDirectory}
            ];
            if (gpuNum !== undefined) {
                envVariables.push({
                    key: 'CUDA_VISIBLE_DEVICES',
                    value: this.gpuScheduler === undefined ? '-1' : resource.gpuIndices.join(',')
                });
            }

        return envVariables;
    }

    private setExtraProperties(trialJobDetail: LocalTrialJobDetail, resource: { gpuIndices: number[] }): void {
        trialJobDetail.gpuIndices = resource.gpuIndices;
    }

    private tryGetAvailableResource(): [boolean, { gpuIndices: number[]}] {
        const resource: { gpuIndices: number[] } = { gpuIndices: [] };
        if (this.gpuScheduler === undefined) {
            return [true, resource];
        }

        let selectedGPUIndices: number[] = [];
        const availableGpuIndices: number[] = this.gpuScheduler.getAvailableGPUIndices(this.config.useActiveGpu, this.occupiedGpuIndexNumMap);
        for (const index of availableGpuIndices) {
            const num: number | undefined = this.occupiedGpuIndexNumMap.get(index);
            if (num === undefined || num < this.config.maxTrialNumberPerGpu) {
                selectedGPUIndices.push(index);
            }
        }

        if (this.config.gpuIndices !== undefined) {
            this.checkSpecifiedGpuIndices();
            selectedGPUIndices = selectedGPUIndices.filter((index: number) => this.config.gpuIndices!.includes(index));
        }

        if (selectedGPUIndices.length < this.config.trialGpuNumber!) {
            return [false, resource];
        }

        selectedGPUIndices.splice(this.config.trialGpuNumber!);
        Object.assign(resource, { gpuIndices: selectedGPUIndices });

        return [true, resource];
    }

    private checkSpecifiedGpuIndices(): void {
        const gpuCount: number | undefined = this.gpuScheduler.getSystemGpuCount();
        if (this.config.gpuIndices !== undefined && gpuCount !== undefined) {
            for (const index of this.config.gpuIndices) {
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
                const trialJobDetail: LocalTrialJobDetail | undefined = this.jobMap.get(trialJobId);
                if (trialJobDetail !== undefined && trialJobDetail.status === 'WAITING') {
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

    private getScript(workingDirectory: string): string[] {
        const script: string[] = [];
        const escapedCommand = shellString(this.config.trialCommand);
        if (process.platform === 'win32') {
            script.push(`$PSDefaultParameterValues = @{'Out-File:Encoding' = 'utf8'}`);
            script.push(`cd $env:NNI_CODE_DIR`);
            script.push(
                `cmd.exe /c ${escapedCommand} 1>${path.join(workingDirectory, 'stdout')} 2>${path.join(workingDirectory, 'stderr')}`,
                `$NOW_DATE = [int64](([datetime]::UtcNow)-(get-date "1/1/1970")).TotalSeconds`,
                `$NOW_DATE = "$NOW_DATE" + (Get-Date -Format fff).ToString()`,
                `Write $LASTEXITCODE " " $NOW_DATE  | Out-File "${path.join(workingDirectory, '.nni', 'state')}" -NoNewline -encoding utf8`);
        } else {
            script.push(`cd $NNI_CODE_DIR`);
            script.push(`eval ${escapedCommand} 1>${path.join(workingDirectory, 'stdout')} 2>${path.join(workingDirectory, 'stderr')}`);
            if (process.platform === 'darwin') {
                // https://superuser.com/questions/599072/how-to-get-bash-execution-time-in-milliseconds-under-mac-os-x
                // Considering the worst case, write 999 to avoid negative duration
                script.push(`echo $? \`date +%s999\` >'${path.join(workingDirectory, '.nni', 'state')}'`);
            } else {
                script.push(`echo $? \`date +%s%3N\` >'${path.join(workingDirectory, '.nni', 'state')}'`);
            }
        }

        return script;
    }

    private async runTrialJob(trialJobId: string, resource: {gpuIndices: number[]}): Promise<void> {
        const trialJobDetail: LocalTrialJobDetail = <LocalTrialJobDetail>this.jobMap.get(trialJobId);
        const variables: { key: string; value: string }[] = this.getEnvironmentVariables(trialJobDetail, resource, this.config.trialGpuNumber);

        const runScriptContent: string[] = [];
        if (process.platform !== 'win32') {
            runScriptContent.push('#!/bin/bash');
        } else {
            runScriptContent.push(`$env:PATH=${powershellString(process.env['path']!)}`)
        }
        for (const variable of variables) {
            runScriptContent.push(setEnvironmentVariable(variable));
        }
        const scripts: string[] = this.getScript(trialJobDetail.workingDirectory);
        scripts.forEach((script: string) => {
            runScriptContent.push(script);
        });
        await execMkdir(trialJobDetail.workingDirectory);
        await execMkdir(path.join(trialJobDetail.workingDirectory, '.nni'));
        await execNewFile(path.join(trialJobDetail.workingDirectory, '.nni', 'metrics'));
        const scriptName: string = getScriptName('run');
        await createScriptFile(path.join(trialJobDetail.workingDirectory, scriptName),
                runScriptContent.join(getNewLine()));
        await this.writeParameterFile(trialJobDetail.workingDirectory, trialJobDetail.form.hyperParameters);
        const trialJobProcess: cp.ChildProcess = runScript(path.join(trialJobDetail.workingDirectory, scriptName));
        this.setTrialJobStatus(trialJobDetail, 'RUNNING');
        trialJobDetail.startTime = Date.now(); // eslint-disable-line require-atomic-updates
        trialJobDetail.pid = trialJobProcess.pid; // eslint-disable-line require-atomic-updates
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

    private async writeParameterFile(directory: string, hyperParameters: HyperParameters): Promise<void> {
        const filepath: string = path.join(directory, generateParamFileName(hyperParameters));
        await fs.promises.writeFile(filepath, hyperParameters.value, { encoding: 'utf8' });
    }

    public async getTrialOutputLocalPath(trialJobId: string): Promise<string> {
        return Promise.resolve(path.join(this.rootDir, 'trials', trialJobId));
    }

    public async fetchTrialOutput(trialJobId: string, subpath: string): Promise<void> {
        let trialLocalPath = await this.getTrialOutputLocalPath(trialJobId);
        if (subpath !== undefined) {
            trialLocalPath = path.join(trialLocalPath, subpath);
        }
        if (fs.existsSync(trialLocalPath)) {
            return Promise.resolve();
        } else {
            return Promise.reject(new Error('Trial local path not exist.'));
        }
    }
}

export { LocalTrainingService };
