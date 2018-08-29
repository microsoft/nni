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
import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Client } from 'ssh2';
import { Deferred } from 'ts-deferred';
import { String } from 'typescript-string-operations';
import * as component from '../../common/component';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import { ObservableTimer } from '../../common/observableTimer';
import {
    HostJobApplicationForm, JobApplicationForm, TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric
} from '../../common/trainingService';
import { delay, getExperimentRootDir, uniqueString } from '../../common/utils';
import { GPUSummary } from '../common/gpuData';
import { TrialConfig } from '../common/trialConfig';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { GPUScheduler } from './gpuScheduler';
import { MetricsCollector } from './metricsCollector';
import {
    HOSTJOBSHELLFORMAT, RemoteCommandResult, RemoteMachineMeta,
    REMOTEMACHINERUNSHELLFORMAT, RemoteMachineScheduleInfo, RemoteMachineScheduleResult,
    RemoteMachineTrialJobDetail, ScheduleResultType
} from './remoteMachineData';
import { SSHClientUtility } from './sshClientUtility';

/**
 * Training Service implementation for Remote Machine (Linux)
 */
class RemoteMachineTrainingService implements TrainingService {
    private machineSSHClientMap: Map<RemoteMachineMeta, Client>;
    private trialJobsMap: Map<string, RemoteMachineTrialJobDetail>;
    private experimentId: string | undefined;
    // Experiment root directory
    private expRootDir: string;
    private remoteExpRootDir: string;
    private trialConfig: TrialConfig | undefined;
    private gpuScheduler: GPUScheduler;
    private jobQueue: string[];
    private timer: ObservableTimer;
    private stopping: boolean = false;
    private metricsEmitter: EventEmitter;
    private log: Logger;

    constructor(@component.Inject timer: ObservableTimer) {
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, RemoteMachineTrialJobDetail>();
        this.machineSSHClientMap = new Map<RemoteMachineMeta, Client>();
        this.gpuScheduler = new GPUScheduler(this.machineSSHClientMap);
        this.jobQueue = [];
        this.experimentId = getExperimentId();
        this.expRootDir = getExperimentRootDir();
        this.remoteExpRootDir = this.getRemoteModeExperimentRootDir();
        this.timer = timer;
        this.log = getLogger();
    }

    /**
     * Loop to launch trial jobs and collect trial metrics
     */
    public async run(): Promise<void> {
        while (!this.stopping) {
            while (this.jobQueue.length > 0) {
                const trialJobId: string = this.jobQueue[0];
                const prepareResult : boolean = await this.prepareTrialJob(trialJobId);
                if (prepareResult) {
                    // Remove trial job with trialJobId from job queue
                    this.jobQueue.shift();
                } else {
                    // Break the while loop since no GPU resource is available right now,
                    // Wait to schedule job in next time iteration
                    break;
                }
            }
            const metricsCollector: MetricsCollector = new MetricsCollector(
                this.machineSSHClientMap, this.trialJobsMap, this.remoteExpRootDir, this.metricsEmitter);
            await metricsCollector.collectMetrics();
            await delay(3000);
        }
    }

    /**
     * List submitted trial jobs
     */
    public listTrialJobs(): Promise<TrialJobDetail[]> {
        const jobs: TrialJobDetail[] = [];
        const deferred: Deferred<TrialJobDetail[]> = new Deferred<TrialJobDetail[]>();

        this.trialJobsMap.forEach(async (value: RemoteMachineTrialJobDetail, key: string) => {
            if (value.form.jobType === 'TRIAL') {
                jobs.push(await this.getTrialJob(key));
            }
        });
        deferred.resolve(jobs);

        return deferred.promise;
    }

    /**
     * Get trial job detail information
     * @param trialJobId ID of trial job
     */
    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        const trialJob: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (!trialJob) {
            throw new NNIError(NNIErrorNames.NOT_FOUND, `trial job id ${trialJobId} not found`);
        }
        //TO DO: add another job status, and design new job status change logic
        if (trialJob.status === 'RUNNING' || trialJob.status === 'UNKNOWN') {
            // Get ssh client where the job is running
            if (trialJob.rmMeta === undefined) {
                throw new Error(`rmMeta not set for submitted job ${trialJobId}`);
            }
            const sshClient: Client | undefined = this.machineSSHClientMap.get(trialJob.rmMeta);
            if (!sshClient) {
                throw new Error(`Invalid job id: ${trialJobId}, cannot find ssh client`);
            }

            return this.updateTrialJobStatus(trialJob, sshClient);
        } else {
            return trialJob;
        }
    }

    /**
     * Add job metrics listener
     * @param listener callback listener
     */
    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.on('metric', listener);
    }

    /**
     * Remove job metrics listener
     * @param listener callback listener
     */
    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.off('metric', listener);
    }

    /**
     * Submit trial job
     * @param form trial job description form
     */
    public submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        this.log.info(`submitTrialJob: form: ${JSON.stringify(form)}`);

        if (!this.trialConfig) {
            throw new Error('trial config is not initialized');
        }

        if (form.jobType === 'HOST') {
            return this.runHostJob(<HostJobApplicationForm>form);
        } else if (form.jobType === 'TRIAL') {
            // Generate trial job id(random)
            const trialJobId: string = uniqueString(5);
            const trialWorkingFolder: string = path.join(this.remoteExpRootDir, 'trials', trialJobId);

            const trialJobDetail: RemoteMachineTrialJobDetail = new RemoteMachineTrialJobDetail(
                trialJobId,
                'WAITING',
                new Date(),
                trialWorkingFolder,
                form);
            this.jobQueue.push(trialJobId);
            this.trialJobsMap.set(trialJobId, trialJobDetail);

            return Promise.resolve(trialJobDetail);
        } else {
            return Promise.reject(new Error(`Job form not supported: ${JSON.stringify(form)}, jobType should be HOST or TRIAL.`));
        }
    }

    /**
     * Cancel trial job
     * @param trialJobId ID of trial job
     */
    public async cancelTrialJob(trialJobId: string): Promise<void> {
        this.log.info(`cancelTrialJob: jobId: ${trialJobId}`);
        const deferred: Deferred<void> = new Deferred<void>();
        const trialJob: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (!trialJob) {
            deferred.reject();
            throw new Error(`trial job id ${trialJobId} not found`);
        }

        // Remove the job with trialJobId from job queue
        const index : number = this.jobQueue.indexOf(trialJobId);
        if (index >= 0) {
            this.jobQueue.splice(index, 1);
        }

        // Get ssh client where the job is running
        if (trialJob.rmMeta !== undefined) {
            // If the trial job is already scheduled, check its status and kill the trial process in remote machine
            const sshClient: Client | undefined = this.machineSSHClientMap.get(trialJob.rmMeta);
            if (!sshClient) {
                deferred.reject();
                throw new Error(`Invalid job id ${trialJobId}, cannot find ssh client`);
            }

            const jobpidPath: string = this.getJobPidPath(trialJob.id);
            try {
                await SSHClientUtility.remoteExeCommand(`pkill -P \`cat ${jobpidPath}\``, sshClient);
                trialJob.status = 'USER_CANCELED';
            } catch (error) {
                // Not handle the error since pkill failed will not impact trial job's current status
                this.log.error(`remoteTrainingService.cancelTrialJob: ${error.message}`);
            }
        } else {
            // Job is not scheduled yet, set status to 'USER_CANCELLED' directly
            trialJob.status = 'USER_CANCELED';
        }
    }

    /**
     * Set culster metadata
     * @param key metadata key
     * //1. MACHINE_LIST -- create ssh client connect of machine list
     * //2. TRIAL_CONFIG -- trial configuration
     * @param value metadata value
     */
    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.MACHINE_LIST:
                await this.setupConnections(value);
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                const remoteMachineTrailConfig: TrialConfig = <TrialConfig>JSON.parse(value);
                // Parse trial config failed, throw Error
                if (!remoteMachineTrailConfig) {
                    throw new Error('trial config parsed failed');
                }
                // codeDir is not a valid directory, throw Error
                if (!fs.lstatSync(remoteMachineTrailConfig.codeDir).isDirectory()) {
                    throw new Error(`codeDir ${remoteMachineTrailConfig.codeDir} is not a directory`);
                }
                this.trialConfig = remoteMachineTrailConfig;
                break;
            default:
                //Reject for unknown keys
                throw new Error(`Uknown key: ${key}`);
        }
    }

    /**
     * Get culster metadata
     * @param key metadata key
     */
    public getClusterMetadata(key: string): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();

        return deferred.promise;
    }

    public cleanUp(): Promise<void> {
        this.stopping = true;

        return Promise.resolve();
    }

    private async setupConnections(machineList: string): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();
        //TO DO: verify if value's format is wrong, and json parse failed, how to handle error
        const rmMetaList: RemoteMachineMeta[] = <RemoteMachineMeta[]>JSON.parse(machineList);
        let connectedRMNum: number = 0;
        rmMetaList.forEach((rmMeta: RemoteMachineMeta) => {
            const conn: Client = new Client();
            this.machineSSHClientMap.set(rmMeta, conn);
            conn.on('ready', async () => {
                await this.initRemoteMachineOnConnected(rmMeta, conn);
                if (++connectedRMNum === rmMetaList.length) {
                    deferred.resolve();
                }
            }).on('error', (err: Error) => {
                // SSH connection error, reject with error message
                deferred.reject(new Error(err.message));
            }).connect({
                host: rmMeta.ip,
                port: rmMeta.port,
                username: rmMeta.username,
                password: rmMeta.passwd
            });
        });

        return deferred.promise;
    }

    private async initRemoteMachineOnConnected(rmMeta: RemoteMachineMeta, conn: Client): Promise<void> {
        // Create root working directory after ssh connection is ready
        //TO DO: Should we mk experiments rootDir here?
        const nniRootDir: string = '/tmp/nni';
        await SSHClientUtility.remoteExeCommand(`mkdir -p ${this.remoteExpRootDir}`, conn);

        // Copy NNI scripts to remote expeirment working directory
        const remoteScriptsDir: string = this.getRemoteScriptsPath();
        await SSHClientUtility.remoteExeCommand(`mkdir -p ${remoteScriptsDir}`, conn);
        await SSHClientUtility.copyDirectoryToRemote('./scripts', remoteScriptsDir, conn);
        await SSHClientUtility.remoteExeCommand(`chmod 777 ${nniRootDir} ${nniRootDir}/* ${nniRootDir}/scripts/*`, conn);

        //Begin to execute gpu_metrics_collection scripts
        SSHClientUtility.remoteExeCommand(`cd ${remoteScriptsDir} && python3 gpu_metrics_collector.py`, conn);

        this.timer.subscribe(
            async (tick: number) => {
                const cmdresult: RemoteCommandResult = await SSHClientUtility.remoteExeCommand(
                    `tail -n 1 ${path.join(remoteScriptsDir, 'gpu_metrics')}`, conn);
                if (cmdresult && cmdresult.stdout) {
                    rmMeta.gpuSummary = <GPUSummary>JSON.parse(cmdresult.stdout);
                }
            }
        );
    }

    private async prepareTrialJob(trialJobId: string): Promise<boolean> {
        const deferred : Deferred<boolean> = new Deferred<boolean>();

        if (!this.trialConfig) {
            throw new Error('trial config is not initialized');
        }
        const trialJobDetail: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new NNIError(NNIErrorNames.INVALID_JOB_DETAIL, `Invalid job detail information for trial job ${trialJobId}`);
        }

        // get an ssh client from scheduler
        const rmScheduleResult: RemoteMachineScheduleResult = this.gpuScheduler.scheduleMachine(this.trialConfig.gpuNum, trialJobId);
        if (rmScheduleResult.resultType === ScheduleResultType.REQUIRE_EXCEED_TOTAL) {
            const errorMessage : string = `Required GPU number ${this.trialConfig.gpuNum} is too large, no machine can meet`;
            this.log.error(errorMessage);
            deferred.reject();
            throw new NNIError(NNIErrorNames.RESOURCE_NOT_AVAILABLE, errorMessage);
        } else if (rmScheduleResult.resultType === ScheduleResultType.SUCCEED
            && rmScheduleResult.scheduleInfo !== undefined) {
            const rmScheduleInfo : RemoteMachineScheduleInfo = rmScheduleResult.scheduleInfo;
            const trialWorkingFolder: string = path.join(this.remoteExpRootDir, 'trials', trialJobId);
            await this.launchTrialOnScheduledMachine(
                trialJobId, trialWorkingFolder, <TrialJobApplicationForm>trialJobDetail.form, rmScheduleInfo);

            trialJobDetail.status = 'RUNNING';
            trialJobDetail.url = `file://${rmScheduleInfo.rmMeta.ip}:${trialWorkingFolder}`;
            trialJobDetail.startTime = new Date();
            trialJobDetail.rmMeta = rmScheduleInfo.rmMeta;

            deferred.resolve(true);
        } else if (rmScheduleResult.resultType === ScheduleResultType.TMP_NO_AVAILABLE_GPU) {
            this.log.info(`Right now no available GPU can be allocated for trial ${trialJobId}, will try to schedule later`);
            deferred.resolve(false);
        } else {
            deferred.reject(`Invalid schedule resutl type: ${rmScheduleResult.resultType}`);
        }

        return deferred.promise;
    }

    private async launchTrialOnScheduledMachine(trialJobId: string, trialWorkingFolder: string, form: TrialJobApplicationForm,
                                                rmScheduleInfo: RemoteMachineScheduleInfo): Promise<void> {
        if (!this.trialConfig) {
            throw new Error('trial config is not initialized');
        }
        const cuda_visible_device: string = rmScheduleInfo.cuda_visible_device;
        const sshClient: Client = rmScheduleInfo.client;
        const trialLocalTempFolder: string = path.join(this.expRootDir, 'trials-local', trialJobId);

        await SSHClientUtility.remoteExeCommand(`mkdir -p ${trialWorkingFolder}`, sshClient);
        await SSHClientUtility.remoteExeCommand(`mkdir -p ${path.join(trialWorkingFolder, '.nni')}`, sshClient);

        await SSHClientUtility.remoteExeCommand(`touch ${path.join(trialWorkingFolder, '.nni', 'metrics')}`, sshClient);
        // RemoteMachineRunShellFormat is the run shell format string,
        // See definition in remoteMachineData.ts
        const runScriptContent: string = String.Format(
            REMOTEMACHINERUNSHELLFORMAT,
            trialWorkingFolder,
            trialJobId,
            path.join(trialWorkingFolder, '.nni', 'jobpid'),
            // Set CUDA_VISIBLE_DEVICES environment variable based on cuda_visible_device
            // If no valid cuda_visible_device is defined, set CUDA_VISIBLE_DEVICES to empty string to hide GPU device
            (typeof cuda_visible_device === 'string' && cuda_visible_device.length > 0) ?
                `CUDA_VISIBLE_DEVICES=${cuda_visible_device} ` : `CUDA_VISIBLE_DEVICES=" " `,
            this.trialConfig.command,
            path.join(trialWorkingFolder, '.nni', 'stderr'),
            path.join(trialWorkingFolder, '.nni', 'code'));

        //create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${trialLocalTempFolder}`);

        // Write file content ( run.sh and parameter.cfg ) to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run.sh'), runScriptContent, { encoding: 'utf8' });
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'parameter.cfg'), form.hyperParameters, { encoding: 'utf8' });

        // Copy local tmp files to remote machine
        await SSHClientUtility.copyFileToRemote(
            path.join(trialLocalTempFolder, 'run.sh'), path.join(trialWorkingFolder, 'run.sh'), sshClient);
        await SSHClientUtility.copyFileToRemote(
            path.join(trialLocalTempFolder, 'parameter.cfg'), path.join(trialWorkingFolder, 'parameter.cfg'), sshClient);

        // Copy files in codeDir to remote working directory
        await SSHClientUtility.copyDirectoryToRemote(this.trialConfig.codeDir, trialWorkingFolder, sshClient);
        // Execute command in remote machine
        SSHClientUtility.remoteExeCommand(`bash ${path.join(trialWorkingFolder, 'run.sh')}`, sshClient);
    }

    private async runHostJob(form: HostJobApplicationForm): Promise<TrialJobDetail> {
        const rmMeta: RemoteMachineMeta = this.getRmMetaByHost(form.host);
        const sshClient: Client | undefined = this.machineSSHClientMap.get(rmMeta);
        if (sshClient === undefined) {
            throw new Error('sshClient not found.');
        }
        const jobId: string = uniqueString(5);
        const localDir: string = path.join(this.expRootDir, 'hostjobs-local', jobId);
        const remoteDir: string = this.getHostJobRemoteDir(jobId);
        await cpp.exec(`mkdir -p ${localDir}`);
        await SSHClientUtility.remoteExeCommand(`mkdir -p ${remoteDir}`, sshClient);
        const runScriptContent: string = String.Format(
            HOSTJOBSHELLFORMAT, remoteDir, path.join(remoteDir, 'jobpid'), form.cmd, path.join(remoteDir, 'code')
        );
        await fs.promises.writeFile(path.join(localDir, 'run.sh'), runScriptContent, { encoding: 'utf8' });
        await SSHClientUtility.copyFileToRemote(
            path.join(localDir, 'run.sh'), path.join(remoteDir, 'run.sh'), sshClient);
        SSHClientUtility.remoteExeCommand(`bash ${path.join(remoteDir, 'run.sh')}`, sshClient);

        const jobDetail: RemoteMachineTrialJobDetail =  new RemoteMachineTrialJobDetail(jobId, 'RUNNING', new Date(), remoteDir, form);
        jobDetail.rmMeta = rmMeta;
        jobDetail.startTime = new Date();
        this.trialJobsMap.set(jobId, jobDetail);
        this.log.debug(`runHostJob: return: ${JSON.stringify(jobDetail)} `);

        return jobDetail;
    }

    private getRmMetaByHost(host: string): RemoteMachineMeta {
        for (const [rmMeta, client] of this.machineSSHClientMap.entries()) {
            if (rmMeta.ip === host) {
                return rmMeta;
            }
        }
        throw new Error(`Host not found: ${host}`);
    }

    private async updateTrialJobStatus(trialJob: RemoteMachineTrialJobDetail, sshClient: Client): Promise<TrialJobDetail> {
        const deferred: Deferred<TrialJobDetail> = new Deferred<TrialJobDetail>();
        const jobpidPath: string = this.getJobPidPath(trialJob.id);
        const trialReturnCodeFilePath: string = path.join(this.remoteExpRootDir, 'trials', trialJob.id, '.nni', 'code');

        try {
            const killResult: number = (await SSHClientUtility.remoteExeCommand(`kill -0 \`cat ${jobpidPath}\``, sshClient)).exitCode;
            // if the process of jobpid is not alive any more
            if (killResult !== 0) {
                const trailReturnCode: string = await SSHClientUtility.getRemoteFileContent(trialReturnCodeFilePath, sshClient);
                this.log.debug(`trailjob ${trialJob.id} return code: ${trailReturnCode}`);
                const match: RegExpMatchArray | null = trailReturnCode.trim().match(/^(\d+)\s+(\d+)$/);
                if (match) {
                    const { 1: code, 2: timestamp } = match;
                    // Update trial job's status based on result code
                    if (parseInt(code, 10) === 0) {
                        trialJob.status = 'SUCCEEDED';
                    } else {
                        trialJob.status = 'FAILED';
                    }
                    trialJob.endTime = new Date(parseInt(timestamp, 10));
                }
                this.log.info(`trailJob status update: ${trialJob.id}, ${trialJob.status}`);
            }
            deferred.resolve(trialJob);
        } catch (error) {
            this.log.error(`Update job status exception, error is ${error.message}`);
            if (error instanceof NNIError && error.name === NNIErrorNames.NOT_FOUND) {
                deferred.resolve(trialJob);
            } else {
                trialJob.status = 'UNKNOWN';
                deferred.resolve(trialJob);
            }
        }

        return deferred.promise;
    }

    private getRemoteScriptsPath(): string {
        return path.join(path.dirname(path.dirname(this.remoteExpRootDir)), 'scripts');
    }

    private getHostJobRemoteDir(jobId: string): string {
        return path.join(this.remoteExpRootDir, 'hostjobs', jobId);
    }

    private getRemoteModeExperimentRootDir(): string{
        return path.join(os.tmpdir(), 'nni', 'experiments', getExperimentId());
    }

    private getJobPidPath(jobId: string): string {
        const trialJobDetail: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(jobId);
        if (trialJobDetail === undefined) {
            throw new NNIError(NNIErrorNames.INVALID_JOB_DETAIL, `Invalid job detail information for trial job ${jobId}`);
        }

        let jobpidPath: string;
        if (trialJobDetail.form.jobType === 'TRIAL') {
            jobpidPath = path.join(trialJobDetail.workingDirectory, '.nni', 'jobpid');
        } else if (trialJobDetail.form.jobType === 'HOST') {
            jobpidPath = path.join(this.getHostJobRemoteDir(jobId), 'jobpid');
        } else {
            throw new Error(`Job type not supported: ${trialJobDetail.form.jobType}`);
        }

        return jobpidPath;
    }
}

export { RemoteMachineTrainingService };
