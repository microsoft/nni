// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as path from 'path';
import { Deferred } from 'ts-deferred';
import { String } from 'typescript-string-operations';
import * as component from '../../common/component';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import { ObservableTimer } from '../../common/observableTimer';
import {
    HyperParameters, NNIManagerIpConfig, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric
} from '../../common/trainingService';
import {
    delay, generateParamFileName, getExperimentRootDir, getIPV4Address, getJobCancelStatus, getRemoteTmpDir,
    getVersion, uniqueString, unixPathJoin
} from '../../common/utils';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { GPUSummary } from '../common/gpuData';
import { TrialConfig } from '../common/trialConfig';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { execCopydir, execMkdir, validateCodeDir, getGpuMetricsCollectorBashScriptContent } from '../common/util';
import { GPUScheduler } from './gpuScheduler';
import {
    REMOTEMACHINE_TRIAL_COMMAND_FORMAT, RemoteMachineMeta,
    RemoteMachineScheduleInfo, RemoteMachineScheduleResult, RemoteMachineTrialJobDetail,
    ScheduleResultType, ExecutorManager
} from './remoteMachineData';
import { RemoteMachineJobRestServer } from './remoteMachineJobRestServer';
import { ShellExecutor } from 'training_service/remote_machine/shellExecutor';
import { threadId } from 'worker_threads';

/**
 * Training Service implementation for Remote Machine (Linux)
 */
@component.Singleton
class RemoteMachineTrainingService implements TrainingService {
    private readonly machineExecutorManagerMap: Map<RemoteMachineMeta, ExecutorManager>; //machine excutor map
    private readonly machineCopyExpCodeDirPromiseMap: Map<RemoteMachineMeta, Promise<void>>;
    private readonly trialExecutorMap: Map<string, ShellExecutor>; //trial excutor map
    private readonly trialJobsMap: Map<string, RemoteMachineTrialJobDetail>;
    private readonly MAX_TRIAL_NUMBER_PER_EXECUTOR: number = 5; // every excutor has a max trial concurrency number
    private readonly expRootDir: string;
    private readonly remoteExpRootDir: string;
    private readonly remoteExpCodeDir: string;
    private trialConfig: TrialConfig | undefined;
    private gpuScheduler?: GPUScheduler;
    private readonly jobQueue: string[];
    private readonly timer: ObservableTimer;
    private stopping: boolean = false;
    private readonly metricsEmitter: EventEmitter;
    private readonly log: Logger;
    private isMultiPhase: boolean = false;
    private trialSequenceId: number;
    private remoteRestServerPort?: number;
    private readonly remoteOS: string;
    private nniManagerIpConfig?: NNIManagerIpConfig;
    private versionCheck: boolean = true;
    private logCollection: string;

    constructor(@component.Inject timer: ObservableTimer) {
        this.remoteOS = 'linux';
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, RemoteMachineTrialJobDetail>();
        this.trialExecutorMap = new Map<string, ShellExecutor>();
        this.machineExecutorManagerMap = new Map<RemoteMachineMeta, ExecutorManager>();
        this.machineCopyExpCodeDirPromiseMap = new Map<RemoteMachineMeta, Promise<void>>();
        this.jobQueue = [];
        this.expRootDir = getExperimentRootDir();
        this.remoteExpRootDir = this.getRemoteExperimentRootDir();
        this.remoteExpCodeDir = unixPathJoin(this.remoteExpRootDir, 'nni-code');
        this.timer = timer;
        this.log = getLogger();
        this.trialSequenceId = -1;
        this.logCollection = 'none';
        this.log.info('Construct remote machine training service.');
    }

    /**
     * Loop to launch trial jobs and collect trial metrics
     */
    public async run(): Promise<void> {
        const restServer: RemoteMachineJobRestServer = component.get(RemoteMachineJobRestServer);
        await restServer.start();
        restServer.setEnableVersionCheck = this.versionCheck;
        this.log.info('Run remote machine training service.');
        while (!this.stopping) {
            while (this.jobQueue.length > 0) {
                this.updateGpuReservation();
                const trialJobId: string = this.jobQueue[0];
                const prepareResult: boolean = await this.prepareTrialJob(trialJobId);
                if (prepareResult) {
                    // Remove trial job with trialJobId from job queue
                    this.jobQueue.shift();
                } else {
                    // Break the while loop since no GPU resource is available right now,
                    // Wait to schedule job in next time iteration
                    break;
                }
            }
            if (restServer.getErrorMessage !== undefined) {
                throw new Error(restServer.getErrorMessage);
                this.stopping = true;
            }
            await delay(3000);
        }
        this.log.info('Remote machine training service exit.');
    }

    /**
     * give trial an executor
     * @param trial remote machine trial job detail
     */
    public async allocateExecutorForTrial(trial: RemoteMachineTrialJobDetail): Promise<void> {
        if (trial.rmMeta === undefined) {
            throw new Error(`rmMeta not set in trial ${trial.id}`);
        }
        const executorManager: ExecutorManager | undefined = this.machineExecutorManagerMap.get(trial.rmMeta);
        if (executorManager === undefined) {
            throw new Error(`executorManager not initialized`);
        }
        const shellExecutor: ShellExecutor = await executorManager.getAvailableExecutor();
        this.trialExecutorMap.set(trial.id, shellExecutor);
    }

    /**
     * If a trial is finished, release the connection resource
     * @param trial remote machine trial job detail
     */
    public releaseTrialExecutor(trial: RemoteMachineTrialJobDetail): void {
        if (trial.rmMeta === undefined) {
            throw new Error(`rmMeta not set in trial ${trial.id}`);
        }
        const executorManager: ExecutorManager | undefined = this.machineExecutorManagerMap.get(trial.rmMeta);
        if (executorManager === undefined) {
            throw new Error(`executorManager not initialized`);
        }
        executorManager.releaseConnection(this.trialExecutorMap.get(trial.id));
    }

    /**
     * List submitted trial jobs
     */
    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        const jobs: TrialJobDetail[] = [];
        const deferred: Deferred<TrialJobDetail[]> = new Deferred<TrialJobDetail[]>();

        for (const [key,] of this.trialJobsMap) {
            jobs.push(await this.getTrialJob(key));
        }
        deferred.resolve(jobs);

        return deferred.promise;
    }

    /**
     * Get trial job detail information
     * @param trialJobId ID of trial job
     */
    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        const trialJob: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (trialJob === undefined) {
            throw new NNIError(NNIErrorNames.NOT_FOUND, `trial job id ${trialJobId} not found`);
        }
        //TO DO: add another job status, and design new job status change logic
        if (trialJob.status === 'RUNNING' || trialJob.status === 'UNKNOWN') {
            // Get executor where the job is running
            if (trialJob.rmMeta === undefined) {
                throw new Error(`rmMeta not set for submitted job ${trialJobId}`);
            }
            const executor: ShellExecutor | undefined = this.trialExecutorMap.get(trialJob.id);
            if (executor === undefined) {
                throw new Error(`Invalid job id: ${trialJobId}, cannot find executor`);
            }

            return this.updateTrialJobStatus(trialJob, executor);
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
    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        if (this.trialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }

        // Generate trial job id(random)
        const trialJobId: string = uniqueString(5);
        const trialWorkingFolder: string = unixPathJoin(this.remoteExpRootDir, 'trials', trialJobId);

        const trialJobDetail: RemoteMachineTrialJobDetail = new RemoteMachineTrialJobDetail(
            trialJobId,
            'WAITING',
            Date.now(),
            trialWorkingFolder,
            form
        );
        this.jobQueue.push(trialJobId);
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        return Promise.resolve(trialJobDetail);
    }

    /**
     * Update trial job for multi-phase
     * @param trialJobId trial job id
     * @param form job application form
     */
    public async updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const trialJobDetail: undefined | TrialJobDetail = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }
        await this.writeParameterFile(trialJobId, form.hyperParameters);

        return trialJobDetail;
    }

    /**
     * Is multiphase job supported in current training service
     */
    public get isMultiPhaseJobSupported(): boolean {
        return true;
    }

    /**
     * Cancel trial job
     * @param trialJobId ID of trial job
     */
    public async cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trialJob: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (trialJob === undefined) {
            throw new Error(`trial job id ${trialJobId} not found`);
        }

        // Remove the job with trialJobId from job queue
        const index: number = this.jobQueue.indexOf(trialJobId);
        if (index >= 0) {
            this.jobQueue.splice(index, 1);
        }

        // Get executor where the job is running
        if (trialJob.rmMeta !== undefined) {
            // If the trial job is already scheduled, check its status and kill the trial process in remote machine
            const executor: ShellExecutor | undefined = this.trialExecutorMap.get(trialJob.id);
            if (executor === undefined) {
                throw new Error(`Invalid job id ${trialJobId}, cannot find executor`);
            }

            if (trialJob.status === 'UNKNOWN') {
                this.releaseTrialExecutor(trialJob);
                trialJob.status = 'USER_CANCELED';
                return
            }

            const jobpidPath: string = this.getJobPidPath(trialJob.id);
            try {
                // Mark the toEarlyStop tag here
                trialJob.isEarlyStopped = isEarlyStopped;
                await executor.killChildProcesses(jobpidPath);
                this.releaseTrialExecutor(trialJob);
            } catch (error) {
                // Not handle the error since pkill failed will not impact trial job's current status
                this.log.error(`remoteTrainingService.cancelTrialJob: ${error.message}`);
            }
        } else {
            // Job is not scheduled yet, set status to 'USER_CANCELLED' directly
            assert(isEarlyStopped === false, 'isEarlyStopped is not supposed to be true here.');
            trialJob.status = getJobCancelStatus(isEarlyStopped);
        }
    }

    /**
     * Set culster metadata
     * @param key metadata key
     * //1. MACHINE_LIST -- create executor of machine list
     * //2. TRIAL_CONFIG -- trial configuration
     * @param value metadata value
     */
    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;
            case TrialConfigMetadataKey.MACHINE_LIST:
                await this.setupConnections(value);
                this.gpuScheduler = new GPUScheduler(this.machineExecutorManagerMap);
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG: {
                const remoteMachineTrailConfig: TrialConfig = <TrialConfig>JSON.parse(value);
                // Parse trial config failed, throw Error
                if (remoteMachineTrailConfig === undefined) {
                    throw new Error('trial config parsed failed');
                }
                // codeDir is not a valid directory, throw Error
                if (!fs.lstatSync(remoteMachineTrailConfig.codeDir)
                    .isDirectory()) {
                    throw new Error(`codeDir ${remoteMachineTrailConfig.codeDir} is not a directory`);
                }

                try {
                    // Validate to make sure codeDir doesn't have too many files
                    await validateCodeDir(remoteMachineTrailConfig.codeDir);
                    // Copy codeDir to remote machine
                    for (const [rmMeta, executorManager] of this.machineExecutorManagerMap.entries()) {
                        const executor: ShellExecutor = await executorManager.getAvailableExecutor();
                        if (executor !== undefined) {
                            await executor.createFolder(this.remoteExpCodeDir);
                            this.machineCopyExpCodeDirPromiseMap.set(
                                rmMeta,
                                executor.copyDirectoryToRemote(remoteMachineTrailConfig.codeDir, this.remoteExpCodeDir, this.remoteOS)
                            ); 
                        }
                    }
                    
                } catch (error) {
                    this.log.error(error);

                    return Promise.reject(new Error(error));
                }

                this.trialConfig = remoteMachineTrailConfig;
                break;
            }
            case TrialConfigMetadataKey.MULTI_PHASE:
                this.isMultiPhase = (value === 'true' || value === 'True');
                break;
            case TrialConfigMetadataKey.VERSION_CHECK:
                this.versionCheck = (value === 'true' || value === 'True');
                break;
            case TrialConfigMetadataKey.LOG_COLLECTION:
                this.logCollection = value;
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
    public async getClusterMetadata(key: string): Promise<string> {
        return "";
    }

    /**
     * cleanup() has a time out of 10s to clean remote connections
     */
    public async cleanUp(): Promise<void> {
        this.log.info('Stopping remote machine training service...');
        this.stopping = true;
        await Promise.race([delay(10000), this.cleanupConnections()]);
    }

    /**
     * remove gpu reversion when job is not running
     */
    private updateGpuReservation(): void {
        if (this.gpuScheduler) {
            for (const [key, value] of this.trialJobsMap) {
                if (!['WAITING', 'RUNNING'].includes(value.status)) {
                    this.gpuScheduler.removeGpuReservation(key, this.trialJobsMap);
                }
            }
        }
    }

    /**
     * stop gpu_metric_collector process in remote machine and remove unused scripts
     */
    private async cleanupConnections(): Promise<void> {
        try {
            for (const [rmMeta, executorManager] of this.machineExecutorManagerMap.entries()) {
                const jobpidPath: string = unixPathJoin(this.getRemoteScriptsPath(rmMeta.username), 'pid');
                const executor: ShellExecutor | undefined = executorManager.getFirstExecutor();
                if (executor !== undefined) {
                    await executor.killChildProcesses(jobpidPath);
                    await executor.removeFolder(this.getRemoteScriptsPath(rmMeta.username));
                }
                executorManager.closeAllExecutor();
            }
        } catch (error) {
            //ignore error, this function is called to cleanup remote connections when experiment is stopping
            this.log.error(`Cleanup connection exception, error is ${error.message}`);
        }

        return Promise.resolve();
    }

    private async setupConnections(machineList: string): Promise<void> {
        this.log.debug(`Connecting to remote machines: ${machineList}`);
        const deferred: Deferred<void> = new Deferred<void>();
        //TO DO: verify if value's format is wrong, and json parse failed, how to handle error
        const rmMetaList: RemoteMachineMeta[] = <RemoteMachineMeta[]>JSON.parse(machineList);
        let connectedRMNum: number = 0;

        rmMetaList.forEach(async (rmMeta: RemoteMachineMeta) => {
            rmMeta.occupiedGpuIndexMap = new Map<number, number>();
            const executorManager: ExecutorManager = new ExecutorManager([], this.MAX_TRIAL_NUMBER_PER_EXECUTOR, rmMeta);
            const executor: ShellExecutor = await executorManager.getAvailableExecutor();
            this.machineExecutorManagerMap.set(rmMeta, executorManager);
            await this.initRemoteMachineOnConnected(rmMeta, executor);
            if (++connectedRMNum === rmMetaList.length) {
                deferred.resolve();
            }
        });

        return deferred.promise;
    }

    private async initRemoteMachineOnConnected(rmMeta: RemoteMachineMeta, executor: ShellExecutor): Promise<void> {
        // Create root working directory after executor is ready
        const nniRootDir: string = unixPathJoin(getRemoteTmpDir(this.remoteOS), 'nni');
        await executor.createFolder(this.remoteExpRootDir);

        // the directory to store temp scripts in remote machine
        const remoteGpuScriptCollectorDir: string = this.getRemoteScriptsPath(rmMeta.username);
        await executor.createFolder(remoteGpuScriptCollectorDir, true);
        await executor.allowPermission(false, nniRootDir, `${nniRootDir}/*`, `${nniRootDir}/scripts/*`);

        //Begin to execute gpu_metrics_collection scripts
        const script = getGpuMetricsCollectorBashScriptContent(remoteGpuScriptCollectorDir);
        executor.executeScript(script, false, true);

        const disposable: Rx.IDisposable = this.timer.subscribe(
            async () => {
                const cmdresult = await executor.readLastLines(unixPathJoin(remoteGpuScriptCollectorDir, 'gpu_metrics'));
                if (cmdresult !== "") {
                    rmMeta.gpuSummary = <GPUSummary>JSON.parse(cmdresult);
                    if (rmMeta.gpuSummary.gpuCount === 0) {
                        this.log.warning(`No GPU found on remote machine ${rmMeta.ip}`);
                        this.timer.unsubscribe(disposable);
                    }
                }
            }
        );
    }

    private async prepareTrialJob(trialJobId: string): Promise<boolean> {
        const deferred: Deferred<boolean> = new Deferred<boolean>();

        if (this.trialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        if (this.gpuScheduler === undefined) {
            throw new Error('gpuScheduler is not initialized');
        }
        const trialJobDetail: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new NNIError(NNIErrorNames.INVALID_JOB_DETAIL, `Invalid job detail information for trial job ${trialJobId}`);
        }
        // If job is not WATIING, Don't prepare and resolve true immediately
        if (trialJobDetail.status !== 'WAITING') {
            deferred.resolve(true);

            return deferred.promise;
        }
        // get an executor from scheduler
        const rmScheduleResult: RemoteMachineScheduleResult = this.gpuScheduler.scheduleMachine(this.trialConfig.gpuNum, trialJobDetail);
        if (rmScheduleResult.resultType === ScheduleResultType.REQUIRE_EXCEED_TOTAL) {
            const errorMessage: string = `Required GPU number ${this.trialConfig.gpuNum} is too large, no machine can meet`;
            this.log.error(errorMessage);
            deferred.reject();
            throw new NNIError(NNIErrorNames.RESOURCE_NOT_AVAILABLE, errorMessage);
        } else if (rmScheduleResult.resultType === ScheduleResultType.SUCCEED
            && rmScheduleResult.scheduleInfo !== undefined) {
            const rmScheduleInfo: RemoteMachineScheduleInfo = rmScheduleResult.scheduleInfo;
            const trialWorkingFolder: string = unixPathJoin(this.remoteExpRootDir, 'trials', trialJobId);

            trialJobDetail.rmMeta = rmScheduleInfo.rmMeta;
            const copyExpCodeDirPromise = this.machineCopyExpCodeDirPromiseMap.get(trialJobDetail.rmMeta);
            if (copyExpCodeDirPromise !== undefined) {
                await copyExpCodeDirPromise;
            }

            await this.allocateExecutorForTrial(trialJobDetail);
            await this.launchTrialOnScheduledMachine(
                trialJobId, trialWorkingFolder, trialJobDetail.form, rmScheduleInfo);

            trialJobDetail.status = 'RUNNING';
            trialJobDetail.url = `file://${rmScheduleInfo.rmMeta.ip}:${trialWorkingFolder}`;
            trialJobDetail.startTime = Date.now();

            this.trialJobsMap.set(trialJobId, trialJobDetail);
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
        if (this.trialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        const cudaVisibleDevice: string = rmScheduleInfo.cudaVisibleDevice;
        const executor: ShellExecutor | undefined = this.trialExecutorMap.get(trialJobId);
        if (executor === undefined) {
            assert(false, 'ShellExecutor is undefined.');

            // for lint
            return;
        }
        const trialJobDetail: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`Can not get trial job detail for job: ${trialJobId}`);
        }

        const trialLocalTempFolder: string = path.join(this.expRootDir, 'trials-local', trialJobId);

        await executor.createFolder(trialWorkingFolder);
        await executor.createFolder(unixPathJoin(trialWorkingFolder, '.nni'));

        // RemoteMachineRunShellFormat is the run shell format string,
        // See definition in remoteMachineData.ts

        let command: string;
        // Set CUDA_VISIBLE_DEVICES environment variable based on cudaVisibleDevice
        // If no valid cudaVisibleDevice is defined, set CUDA_VISIBLE_DEVICES to empty string to hide GPU device
        // If gpuNum is undefined, will not set CUDA_VISIBLE_DEVICES in script
        if (this.trialConfig.gpuNum === undefined) {
            command = this.trialConfig.command;
        } else {
            if (typeof cudaVisibleDevice === 'string' && cudaVisibleDevice.length > 0) {
                command = `CUDA_VISIBLE_DEVICES=${cudaVisibleDevice} ${this.trialConfig.command}`;
            } else {
                command = `CUDA_VISIBLE_DEVICES=" " ${this.trialConfig.command}`;
            }
        }
        const nniManagerIp: string = this.nniManagerIpConfig ? this.nniManagerIpConfig.nniManagerIp : getIPV4Address();
        if (this.remoteRestServerPort === undefined) {
            const restServer: RemoteMachineJobRestServer = component.get(RemoteMachineJobRestServer);
            this.remoteRestServerPort = restServer.clusterRestServerPort;
        }
        const version: string = this.versionCheck ? await getVersion() : '';
        const runScriptTrialContent: string = String.Format(
            REMOTEMACHINE_TRIAL_COMMAND_FORMAT,
            trialWorkingFolder,
            trialWorkingFolder,
            trialJobId,
            getExperimentId(),
            trialJobDetail.form.sequenceId.toString(),
            this.isMultiPhase,
            this.remoteExpCodeDir,
            unixPathJoin(trialWorkingFolder, '.nni', 'jobpid'),
            command,
            nniManagerIp,
            this.remoteRestServerPort,
            version,
            this.logCollection,
            unixPathJoin(trialWorkingFolder, '.nni', 'code')
        );

        //create tmp trial working folder locally.
        await execMkdir(path.join(trialLocalTempFolder, '.nni'));
        // Write install_nni.sh
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), CONTAINER_INSTALL_NNI_SHELL_FORMAT, { encoding: 'utf8' });
        // Write file content ( run.sh and parameter.cfg ) to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run.sh'), runScriptTrialContent, { encoding: 'utf8' });
        await this.writeParameterFile(trialJobId, form.hyperParameters);
        // Copy files in codeDir to remote working directory
        await executor.copyDirectoryToRemote(trialLocalTempFolder, trialWorkingFolder, this.remoteOS);
        // Execute command in remote machine
        executor.executeScript(unixPathJoin(trialWorkingFolder, 'run.sh'), true, true);
    }

    private getRmMetaByHost(host: string): RemoteMachineMeta {
        for (const rmMeta of this.machineExecutorManagerMap.keys()) {
            if (rmMeta.ip === host) {
                return rmMeta;
            }
        }
        throw new Error(`Host not found: ${host}`);
    }

    private async updateTrialJobStatus(trialJob: RemoteMachineTrialJobDetail, executor: ShellExecutor): Promise<TrialJobDetail> {
        const deferred: Deferred<TrialJobDetail> = new Deferred<TrialJobDetail>();
        const jobpidPath: string = this.getJobPidPath(trialJob.id);
        const trialReturnCodeFilePath: string = unixPathJoin(this.remoteExpRootDir, 'trials', trialJob.id, '.nni', 'code');
        /* eslint-disable require-atomic-updates */
        try {
            const isAlive = await executor.isProcessAlive(jobpidPath);
            // if the process of jobpid is not alive any more
            if (!isAlive) {
                const trialReturnCode: string = await executor.getRemoteFileContent(trialReturnCodeFilePath);
                this.log.debug(`trailjob ${trialJob.id} return code: ${trialReturnCode}`);
                const match: RegExpMatchArray | null = trialReturnCode.trim()
                    .match(/^(\d+)\s+(\d+)$/);
                if (match !== null) {
                    const { 1: code, 2: timestamp } = match;
                    // Update trial job's status based on result code
                    if (parseInt(code, 10) === 0) {
                        trialJob.status = 'SUCCEEDED';
                    } else {
                        // isEarlyStopped is never set, mean it's not cancelled by NNI, so if the process's exit code >0, mark it as FAILED
                        if (trialJob.isEarlyStopped === undefined) {
                            trialJob.status = 'FAILED';
                        } else {
                            trialJob.status = getJobCancelStatus(trialJob.isEarlyStopped);
                        }
                    }
                    trialJob.endTime = parseInt(timestamp, 10);
                    this.releaseTrialExecutor(trialJob);
                }
                this.log.debug(`trailJob status update: ${trialJob.id}, ${trialJob.status}`);
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
        /* eslint-enable require-atomic-updates */
        return deferred.promise;
    }

    private getRemoteScriptsPath(userName: string): string {
        return unixPathJoin(getRemoteTmpDir(this.remoteOS), userName, 'nni', 'scripts');
    }

    private getHostJobRemoteDir(jobId: string): string {
        return unixPathJoin(this.remoteExpRootDir, 'hostjobs', jobId);
    }

    private getRemoteExperimentRootDir(): string {
        return unixPathJoin(getRemoteTmpDir(this.remoteOS), 'nni', 'experiments', getExperimentId());
    }

    public get MetricsEmitter(): EventEmitter {
        return this.metricsEmitter;
    }

    private getJobPidPath(jobId: string): string {
        const trialJobDetail: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(jobId);
        if (trialJobDetail === undefined) {
            throw new NNIError(NNIErrorNames.INVALID_JOB_DETAIL, `Invalid job detail information for trial job ${jobId}`);
        }

        return unixPathJoin(trialJobDetail.workingDirectory, '.nni', 'jobpid');
    }

    private async writeParameterFile(trialJobId: string, hyperParameters: HyperParameters): Promise<void> {
        const executor: ShellExecutor | undefined = this.trialExecutorMap.get(trialJobId);
        if (executor === undefined) {
            throw new Error('ShellExecutor is undefined.');
        }

        const trialWorkingFolder: string = unixPathJoin(this.remoteExpRootDir, 'trials', trialJobId);
        const trialLocalTempFolder: string = path.join(this.expRootDir, 'trials-local', trialJobId);

        const fileName: string = generateParamFileName(hyperParameters);
        const localFilepath: string = path.join(trialLocalTempFolder, fileName);
        await fs.promises.writeFile(localFilepath, hyperParameters.value, { encoding: 'utf8' });

        await executor.copyFileToRemote(localFilepath, unixPathJoin(trialWorkingFolder, fileName));
    }
}

export { RemoteMachineTrainingService };
