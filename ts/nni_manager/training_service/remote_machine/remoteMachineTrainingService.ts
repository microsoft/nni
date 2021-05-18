// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as assert from 'assert';
import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as path from 'path';
import { ShellExecutor } from 'training_service/remote_machine/shellExecutor';
import { Deferred } from 'ts-deferred';
import * as component from '../../common/component';
import { NNIError, NNIErrorNames, MethodNotImplementedError } from '../../common/errors';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import { ObservableTimer } from '../../common/observableTimer';
import {
    HyperParameters, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric, LogType
} from '../../common/trainingService';
import {
    delay, generateParamFileName, getExperimentRootDir, getIPV4Address, getJobCancelStatus,
    getVersion, uniqueString
} from '../../common/utils';
import { ExperimentConfig, RemoteConfig, RemoteMachineConfig, flattenConfig } from '../../common/experimentConfig';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { GPUSummary, ScheduleResultType } from '../common/gpuData';
import { execMkdir, validateCodeDir } from '../common/util';
import { GPUScheduler } from './gpuScheduler';
import {
    ExecutorManager, RemoteMachineScheduleInfo, RemoteMachineScheduleResult, RemoteMachineTrialJobDetail
} from './remoteMachineData';
import { RemoteMachineJobRestServer } from './remoteMachineJobRestServer';

interface FlattenRemoteConfig extends ExperimentConfig, RemoteConfig { }

/**
 * Training Service implementation for Remote Machine (Linux)
 */
@component.Singleton
class RemoteMachineTrainingService implements TrainingService {
    private readonly initExecutorId = "initConnection";
    private readonly machineExecutorManagerMap: Map<RemoteMachineConfig, ExecutorManager>; //machine excutor map
    private readonly machineCopyExpCodeDirPromiseMap: Map<RemoteMachineConfig, Promise<void>>;
    private readonly trialExecutorManagerMap: Map<string, ExecutorManager>; //trial excutor map
    private readonly trialJobsMap: Map<string, RemoteMachineTrialJobDetail>;
    private readonly expRootDir: string;
    private gpuScheduler?: GPUScheduler;
    private readonly jobQueue: string[];
    private readonly timer: ObservableTimer;
    private stopping: boolean = false;
    private readonly metricsEmitter: EventEmitter;
    private readonly log: Logger;
    private remoteRestServerPort?: number;
    private versionCheck: boolean = true;
    private logCollection: string = 'none';
    private sshConnectionPromises: any[];
    private config: FlattenRemoteConfig;

    constructor(config: ExperimentConfig) {
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, RemoteMachineTrialJobDetail>();
        this.trialExecutorManagerMap = new Map<string, ExecutorManager>();
        this.machineCopyExpCodeDirPromiseMap = new Map<RemoteMachineConfig, Promise<void>>();
        this.machineExecutorManagerMap = new Map<RemoteMachineConfig, ExecutorManager>();
        this.jobQueue = [];
        this.sshConnectionPromises = [];
        this.expRootDir = getExperimentRootDir();
        this.timer = component.get(ObservableTimer);
        this.log = getLogger();
        this.log.info('Construct remote machine training service.');
        this.config = flattenConfig(config, 'remote');

        if (!fs.lstatSync(this.config.trialCodeDirectory).isDirectory()) {
            throw new Error(`codeDir ${this.config.trialCodeDirectory} is not a directory`);
        }
        validateCodeDir(this.config.trialCodeDirectory);

        this.sshConnectionPromises = this.config.machineList.map(
            machine => this.initRemoteMachineOnConnected(machine)
        );
    }

    /**
     * Loop to launch trial jobs and collect trial metrics
     */
    public async run(): Promise<void> {
        const restServer = new RemoteMachineJobRestServer(this);
        await restServer.start();
        restServer.setEnableVersionCheck = this.versionCheck;
        this.log.info('Run remote machine training service.');
        if (this.sshConnectionPromises.length > 0) {
            await Promise.all(this.sshConnectionPromises);
            this.log.info('ssh connection initialized!');
            // set sshConnectionPromises to [] to avoid log information duplicated
            this.sshConnectionPromises = [];
            // initialize gpuScheduler
            this.gpuScheduler = new GPUScheduler(this.machineExecutorManagerMap);
            // Copy codeDir to remote machine
            for (const [machineConfig, executorManager] of this.machineExecutorManagerMap.entries()) {
                const executor: ShellExecutor = await executorManager.getExecutor(this.initExecutorId);
                if (executor !== undefined) {
                    this.machineCopyExpCodeDirPromiseMap.set(
                        machineConfig,
                        executor.copyDirectoryToRemote(this.config.trialCodeDirectory, executor.getRemoteCodePath(getExperimentId()))
                    );
                }
            }
        }
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
                this.stopping = true;
                throw new Error(restServer.getErrorMessage);
            }
            await delay(3000);
        }
        this.log.info('RemoteMachineTrainingService run loop exited.');
    }

    /**
     * give trial an executor
     * @param trial remote machine trial job detail
     */
    public allocateExecutorManagerForTrial(trial: RemoteMachineTrialJobDetail): void {
        if (trial.rmMeta === undefined) {
            throw new Error(`rmMeta not set in trial ${trial.id}`);
        }
        const executorManager: ExecutorManager | undefined = this.machineExecutorManagerMap.get(trial.rmMeta.config);
        if (executorManager === undefined) {
            throw new Error(`executorManager not initialized`);
        }
        this.trialExecutorManagerMap.set(trial.id, executorManager);
    }

    /**
     * If a trial is finished, release the connection resource
     * @param trial remote machine trial job detail
     */
    public releaseTrialResource(trial: RemoteMachineTrialJobDetail): void {
        if (trial.rmMeta === undefined) {
            throw new Error(`rmMeta not set in trial ${trial.id}`);
        }
        const executorManager = this.trialExecutorManagerMap.get(trial.id);
        if (executorManager === undefined) {
            throw new Error(`ExecutorManager is not assigned for trial ${trial.id}`);
        }
        // Note, it still keep reference in trialExecutorManagerMap, as there may be following requests from nni manager.
        executorManager.releaseExecutor(trial.id);
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
            const executor = await this.getExecutor(trialJob.id);

            return this.updateTrialJobStatus(trialJob, executor);
        } else {
            return trialJob;
        }
    }

    /**
     * Get trial job log
     * @param _trialJobId ID of trial job
     * @param _logType 'TRIAL_LOG' | 'TRIAL_STDERR'
     */
    public async getTrialLog(_trialJobId: string, _logType: LogType): Promise<string> {
        throw new MethodNotImplementedError();
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
        // Generate trial job id(random)
        const trialJobId: string = uniqueString(5);

        const trialJobDetail: RemoteMachineTrialJobDetail = new RemoteMachineTrialJobDetail(
            trialJobId,
            'WAITING',
            Date.now(),
            "unset",
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
            const executor = await this.getExecutor(trialJob.id);

            if (trialJob.status === 'UNKNOWN') {
                trialJob.status = 'USER_CANCELED';
                this.releaseTrialResource(trialJob);
                return
            }

            const jobpidPath: string = this.getJobPidPath(executor, trialJob.id);
            try {
                // Mark the toEarlyStop tag here
                trialJob.isEarlyStopped = isEarlyStopped;
                await executor.killChildProcesses(jobpidPath);
                this.releaseTrialResource(trialJob);
            } catch (error) {
                // Not handle the error since pkill failed will not impact trial job's current status
                this.log.error(`remoteTrainingService.cancelTrialJob: ${error}`);
            }
        } else {
            // Job is not scheduled yet, set status to 'USER_CANCELLED' directly
            assert(isEarlyStopped === false, 'isEarlyStopped is not supposed to be true here.');
            trialJob.status = getJobCancelStatus(isEarlyStopped);
        }
    }

    public async setClusterMetadata(_key: string, _value: string): Promise<void> { return; }
    public async getClusterMetadata(_key: string): Promise<string> { return ''; }

    /**
     * cleanup() has a time out of 10s to clean remote connections
     */
    public async cleanUp(): Promise<void> {
        this.log.info('Stopping remote machine training service...');
        this.stopping = true;
        await this.cleanupConnections();
    }

    private async getExecutor(trialId: string): Promise<ShellExecutor> {
        const executorManager = this.trialExecutorManagerMap.get(trialId);
        if (executorManager === undefined) {
            throw new Error(`ExecutorManager is not assigned for trial ${trialId}`);
        }
        return await executorManager.getExecutor(trialId);
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
            for (const executorManager of this.machineExecutorManagerMap.values()) {
                const executor = await executorManager.getExecutor(this.initExecutorId);
                if (executor !== undefined) {
                    this.log.info(`killing gpu metric collector on ${executor.name}`);
                    const gpuJobPidPath: string = executor.joinPath(executor.getRemoteScriptsPath(getExperimentId()), 'pid');
                    await executor.killChildProcesses(gpuJobPidPath, true);
                }
                executorManager.releaseAllExecutor();
            }
        } catch (error) {
            //ignore error, this function is called to cleanup remote connections when experiment is stopping
            this.log.error(`Cleanup connection exception, error is ${error}`);
        }
    }

    private async initRemoteMachineOnConnected(machineConfig: RemoteMachineConfig): Promise<void> {
        const executorManager: ExecutorManager = new ExecutorManager(machineConfig);
        this.log.info(`connecting to ${machineConfig.user}@${machineConfig.host}:${machineConfig.port}`);
        const executor: ShellExecutor = await executorManager.getExecutor(this.initExecutorId);
        this.log.debug(`reached ${executor.name}`);
        this.machineExecutorManagerMap.set(machineConfig, executorManager);
        this.log.debug(`initializing ${executor.name}`);

        // Create root working directory after executor is ready
        const nniRootDir: string = executor.joinPath(executor.getTempPath(), 'nni');
        await executor.createFolder(executor.getRemoteExperimentRootDir(getExperimentId()));

        // the directory to store temp scripts in remote machine
        const remoteGpuScriptCollectorDir: string = executor.getRemoteScriptsPath(getExperimentId());

        // clean up previous result.
        await executor.createFolder(remoteGpuScriptCollectorDir, true);
        await executor.allowPermission(true, nniRootDir);

        //Begin to execute gpu_metrics_collection scripts
        const script = executor.generateGpuStatsScript(getExperimentId());
        executor.executeScript(script, false, true);
        // the timer is trigger in 1 second, it causes multiple runs on server.
        // So reduce it's freqeunce, only allow one of it run.
        const collectingCount: boolean[] = [];

        const disposable: Rx.IDisposable = this.timer.subscribe(
            async () => {
                if (collectingCount.length == 0) {
                    collectingCount.push(true);
                    const cmdresult = await executor.readLastLines(executor.joinPath(remoteGpuScriptCollectorDir, 'gpu_metrics'));
                    if (cmdresult !== "") {
                        executorManager.rmMeta.gpuSummary = <GPUSummary>JSON.parse(cmdresult);
                        if (executorManager.rmMeta.gpuSummary.gpuCount === 0) {
                            this.log.warning(`No GPU found on remote machine ${machineConfig.host}`);
                            this.timer.unsubscribe(disposable);
                        }
                    }
                    if (this.stopping) {
                        this.timer.unsubscribe(disposable);
                        this.log.debug(`Stopped GPU collector on ${machineConfig.host}, since experiment is exiting.`);
                    }
                    collectingCount.pop();
                }
            }
        );
    }

    private async prepareTrialJob(trialJobId: string): Promise<boolean> {
        const deferred: Deferred<boolean> = new Deferred<boolean>();

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
        const rmScheduleResult: RemoteMachineScheduleResult = this.gpuScheduler.scheduleMachine(this.config.trialGpuNumber, trialJobDetail);
        if (rmScheduleResult.resultType === ScheduleResultType.REQUIRE_EXCEED_TOTAL) {
            const errorMessage: string = `Required GPU number ${this.config.trialGpuNumber} is too large, no machine can meet`;
            this.log.error(errorMessage);
            deferred.reject();
            throw new NNIError(NNIErrorNames.RESOURCE_NOT_AVAILABLE, errorMessage);
        } else if (rmScheduleResult.resultType === ScheduleResultType.SUCCEED
            && rmScheduleResult.scheduleInfo !== undefined) {
            const rmScheduleInfo: RemoteMachineScheduleInfo = rmScheduleResult.scheduleInfo;

            trialJobDetail.rmMeta = rmScheduleInfo.rmMeta;
            const copyExpCodeDirPromise = this.machineCopyExpCodeDirPromiseMap.get(rmScheduleInfo.rmMeta.config);
            if (copyExpCodeDirPromise !== undefined) {
                await copyExpCodeDirPromise;
            }

            this.allocateExecutorManagerForTrial(trialJobDetail);
            const executor = await this.getExecutor(trialJobDetail.id);

            trialJobDetail.workingDirectory = executor.joinPath(executor.getRemoteExperimentRootDir(getExperimentId()), 'trials', trialJobDetail.id);

            await this.launchTrialOnScheduledMachine(
                trialJobId, trialJobDetail.form, rmScheduleInfo);

            trialJobDetail.status = 'RUNNING';
            trialJobDetail.url = `file://${rmScheduleInfo.rmMeta.config.host}:${trialJobDetail.workingDirectory}`;
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

    private async launchTrialOnScheduledMachine(trialJobId: string, form: TrialJobApplicationForm,
        rmScheduleInfo: RemoteMachineScheduleInfo): Promise<void> {
        const cudaVisibleDevice: string = rmScheduleInfo.cudaVisibleDevice;
        const executor = await this.getExecutor(trialJobId);
        const trialJobDetail: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`Can not get trial job detail for job: ${trialJobId}`);
        }

        const trialLocalTempFolder: string = path.join(this.expRootDir, 'trials-local', trialJobId);

        await executor.createFolder(executor.joinPath(trialJobDetail.workingDirectory, '.nni'));

        // RemoteMachineRunShellFormat is the run shell format string,
        // See definition in remoteMachineData.ts

        let cudaVisible: string;
        // Set CUDA_VISIBLE_DEVICES environment variable based on cudaVisibleDevice
        // If no valid cudaVisibleDevice is defined, set CUDA_VISIBLE_DEVICES to empty string to hide GPU device
        // If gpuNum is undefined, will not set CUDA_VISIBLE_DEVICES in script
        if (this.config.trialGpuNumber === undefined) {
            cudaVisible = ""
        } else {
            if (typeof cudaVisibleDevice === 'string' && cudaVisibleDevice.length > 0) {
                cudaVisible = `CUDA_VISIBLE_DEVICES=${cudaVisibleDevice}`;
            } else {
                cudaVisible = `CUDA_VISIBLE_DEVICES=" "`;
            }
        }
        const nniManagerIp: string = this.config.nniManagerIp ? this.config.nniManagerIp : getIPV4Address();
        if (this.remoteRestServerPort === undefined) {
            const restServer: RemoteMachineJobRestServer = component.get(RemoteMachineJobRestServer);
            this.remoteRestServerPort = restServer.clusterRestServerPort;
        }
        const version: string = this.versionCheck ? await getVersion() : '';
        const runScriptTrialContent: string = executor.generateStartScript(
            trialJobDetail.workingDirectory,
            trialJobId,
            getExperimentId(),
            trialJobDetail.form.sequenceId.toString(),
            false,  // multi-phase
            this.config.trialCommand,
            nniManagerIp,
            this.remoteRestServerPort,
            version,
            this.logCollection,
            cudaVisible);

        //create tmp trial working folder locally.
        await execMkdir(path.join(trialLocalTempFolder, '.nni'));

        // Write install_nni.sh, it's not used in Windows platform.
        await fs.promises.writeFile(path.join(trialLocalTempFolder, executor.getScriptName("install_nni")), CONTAINER_INSTALL_NNI_SHELL_FORMAT, { encoding: 'utf8' });
        // Write file content ( run.sh and parameter.cfg ) to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, executor.getScriptName("run")), runScriptTrialContent, { encoding: 'utf8' });
        await this.writeParameterFile(trialJobId, form.hyperParameters);
        // Copy files in codeDir to remote working directory
        await executor.copyDirectoryToRemote(trialLocalTempFolder, trialJobDetail.workingDirectory);
        // Execute command in remote machine
        executor.executeScript(executor.joinPath(trialJobDetail.workingDirectory, executor.getScriptName("run")), true, true);
    }

    private async updateTrialJobStatus(trialJob: RemoteMachineTrialJobDetail, executor: ShellExecutor): Promise<TrialJobDetail> {
        const deferred: Deferred<TrialJobDetail> = new Deferred<TrialJobDetail>();
        const jobpidPath: string = this.getJobPidPath(executor, trialJob.id);
        const trialReturnCodeFilePath: string = executor.joinPath(executor.getRemoteExperimentRootDir(getExperimentId()), 'trials', trialJob.id, '.nni', 'code');
        /* eslint-disable require-atomic-updates */
        try {
            const isAlive = await executor.isProcessAlive(jobpidPath);
            // if the process of jobpid is not alive any more
            if (!isAlive) {
                const trialReturnCode: string = await executor.getRemoteFileContent(trialReturnCodeFilePath);
                this.log.debug(`trailjob ${trialJob.id} return code: ${trialReturnCode}`);
                const match: RegExpMatchArray | null = trialReturnCode.trim()
                    .match(/^-?(\d+)\s+(\d+)$/);
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
                    this.releaseTrialResource(trialJob);
                }
                this.log.debug(`trailJob status update: ${trialJob.id}, ${trialJob.status}`);
            }
            deferred.resolve(trialJob);
        } catch (error) {
            this.log.debug(`(Ignorable mostly)Update job status exception, error is ${error.message}`);
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

    public get MetricsEmitter(): EventEmitter {
        return this.metricsEmitter;
    }

    private getJobPidPath(executor: ShellExecutor, jobId: string): string {
        const trialJobDetail: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(jobId);
        if (trialJobDetail === undefined) {
            throw new NNIError(NNIErrorNames.INVALID_JOB_DETAIL, `Invalid job detail information for trial job ${jobId}`);
        }

        return executor.joinPath(trialJobDetail.workingDirectory, '.nni', 'jobpid');
    }

    private async writeParameterFile(trialJobId: string, hyperParameters: HyperParameters): Promise<void> {
        const executor = await this.getExecutor(trialJobId);

        const trialWorkingFolder: string = executor.joinPath(executor.getRemoteExperimentRootDir(getExperimentId()), 'trials', trialJobId);
        const trialLocalTempFolder: string = path.join(this.expRootDir, 'trials-local', trialJobId);

        const fileName: string = generateParamFileName(hyperParameters);
        const localFilepath: string = path.join(trialLocalTempFolder, fileName);
        await fs.promises.writeFile(localFilepath, hyperParameters.value, { encoding: 'utf8' });

        await executor.copyFileToRemote(localFilepath, executor.joinPath(trialWorkingFolder, fileName));
    }

    public getTrialOutputLocalPath(_trialJobId: string): Promise<string> {
        throw new MethodNotImplementedError();
    }

    public fetchTrialOutput(_trialJobId: string, _subpath: string): Promise<void> {
        throw new MethodNotImplementedError();
    }
}

export { RemoteMachineTrainingService };
