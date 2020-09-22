// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as yaml from 'js-yaml';
import * as request from 'request';
import * as path from 'path';
import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';
import * as component from '../../../common/component';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../../common/log';
import { EnvironmentInformation, EnvironmentService } from '../environment';
import { StorageService } from '../storageService';
import { NNIError, NNIErrorNames, MethodNotImplementedError } from '../../../common/errors';
import { ObservableTimer } from '../../../common/observableTimer';
import {
    HyperParameters, NNIManagerIpConfig, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric, LogType
} from '../../../common/trainingService';
import {
    delay, generateParamFileName, getExperimentRootDir, getIPV4Address, getJobCancelStatus,
    getVersion, uniqueString
} from '../../../common/utils';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../../common/containerJobData';
import { GPUSummary, ScheduleResultType } from '../../common/gpuData';
import { TrialConfig } from '../../common/trialConfig';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { execMkdir, validateCodeDir } from '../../common/util';
import { GPUScheduler } from '../../remote_machine/gpuScheduler';
import {
    ExecutorManager, RemoteMachineMeta,
    RemoteMachineScheduleInfo, RemoteMachineScheduleResult, RemoteMachineTrialJobDetail
} from '../../remote_machine/remoteMachineData';
import { RemoteMachineJobRestServer } from '../../remote_machine/remoteMachineJobRestServer';
import { ShellExecutor } from 'training_service/remote_machine/shellExecutor';
import { RemoteMachineEnvironmentInformation, RemoteMachineMetaDetail } from '../remote/remoteConfig';


@component.Singleton
export class RemoteEnvironmentService extends EnvironmentService {

    private readonly initExecutorId = "initConnection";
    private readonly machineExecutorManagerMap: Map<RemoteMachineMeta, ExecutorManager>; //machine excutor map
    private readonly machineCopyExpCodeDirPromiseMap: Map<RemoteMachineMeta, Promise<void>>;
    private readonly environmentExecutorManagerMap: Map<string, ExecutorManager>; //trial excutor map
    private readonly environmentJobsMap: Map<string, RemoteMachineEnvironmentInformation>;
    private readonly expRootDir: string;
    private trialConfig: TrialConfig | undefined;
    private gpuScheduler?: GPUScheduler;
    private readonly jobQueue: string[];
    private readonly timer: ObservableTimer;
    private stopping: boolean = false;
    private readonly metricsEmitter: EventEmitter;
    private readonly log: Logger;
    private sshConnectionPromises: any[];
    private experimentRootDir: string;
    private experimentId: string;

    constructor(@component.Inject timer: ObservableTimer) {
        super();
        this.experimentId = getExperimentId();
        this.metricsEmitter = new EventEmitter();
        this.environmentJobsMap = new Map<string, RemoteMachineEnvironmentInformation>();
        this.environmentExecutorManagerMap = new Map<string, ExecutorManager>();
        this.machineCopyExpCodeDirPromiseMap = new Map<RemoteMachineMeta, Promise<void>>();
        this.machineExecutorManagerMap = new Map<RemoteMachineMeta, ExecutorManager>();
        this.jobQueue = [];
        this.sshConnectionPromises = [];
        this.expRootDir = getExperimentRootDir();
        this.experimentRootDir = getExperimentRootDir();
        this.experimentId = getExperimentId();
        this.timer = timer;
        this.log = getLogger();
        this.log.info('Construct remote machine training service.');
    }

    public get environmentMaintenceLoopInterval(): number {
        return 5000;
    }

    public get hasStorageService(): boolean {
        return false;
    }

    /**
     * Set culster metadata
     * @param key metadata key
     * //1. MACHINE_LIST -- create executor of machine list
     * //2. TRIAL_CONFIG -- trial configuration
     * @param value metadata value
     */
    public async config(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.MACHINE_LIST:
                await this.setupConnections(value);
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
                } catch (error) {
                    this.log.error(error);
                    return Promise.reject(new Error(error));
                }

                this.trialConfig = remoteMachineTrailConfig;
                break;
            }
            default:
                //Reject for unknown keys
                throw new Error(`Uknown key: ${key}`);
        }
    }

    private async setupConnections(machineList: string): Promise<void> {
        this.log.debug(`Connecting to remote machines: ${machineList}`);
        //TO DO: verify if value's format is wrong, and json parse failed, how to handle error
        const rmMetaList: RemoteMachineMeta[] = <RemoteMachineMeta[]>JSON.parse(machineList);

        for (const rmMeta of rmMetaList) {
            this.sshConnectionPromises.push(this.initRemoteMachineOnConnected(rmMeta));
        }
    }

    private async initRemoteMachineOnConnected(rmMeta: RemoteMachineMeta): Promise<void> {
        rmMeta.occupiedGpuIndexMap = new Map<number, number>();
        const executorManager: ExecutorManager = new ExecutorManager(rmMeta);
        this.log.info(`connecting to ${rmMeta.username}@${rmMeta.ip}:${rmMeta.port}`);
        const executor: ShellExecutor = await executorManager.getExecutor(this.initExecutorId);
        this.log.debug(`reached ${executor.name}`);
        this.machineExecutorManagerMap.set(rmMeta, executorManager);
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
                        rmMeta.gpuSummary = <GPUSummary>JSON.parse(cmdresult);
                        if (rmMeta.gpuSummary.gpuCount === 0) {
                            this.log.warning(`No GPU found on remote machine ${rmMeta.ip}`);
                            this.timer.unsubscribe(disposable);
                        }
                    }
                    if (this.stopping) {
                        this.timer.unsubscribe(disposable);
                        this.log.debug(`Stopped GPU collector on ${rmMeta.ip}, since experiment is exiting.`);
                    }
                    collectingCount.pop();
                }
            }
        );
    }

    public async refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void> {
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();

        if (this.sshConnectionPromises.length > 0) {
            await Promise.all(this.sshConnectionPromises);
            this.log.info('ssh connection initialized!');
            // set sshConnectionPromises to [] to avoid log information duplicated
            this.sshConnectionPromises = [];
            if (this.trialConfig ===  undefined) {
                throw new Error("trial config not initialized!");
            }
            const environmentLocalTempFolder = path.join(this.experimentRootDir, this.experimentId, "environment-temp");
        }

        await this.prepareEnvironment(environment);
        await this.launchEnvironmentOnScheduledMachine(environment);
        return deferred.promise;
    }

    private async prepareEnvironment(environment: RemoteMachineEnvironmentInformation): Promise<boolean> {
        const deferred: Deferred<boolean> = new Deferred<boolean>();

        if (this.trialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        if (this.gpuScheduler === undefined) {
            throw new Error('gpuScheduler is not initialized');
        }

        if (environment.rmMachineMetaDetail === undefined) {
            throw new NNIError(NNIErrorNames.INVALID_JOB_DETAIL, 
                `Invalid rmMachineMetaDetail for environment ${environment.id}`);
        }
        // If job is not WATIING, Don't prepare and resolve true immediately
        if (environment.status !== 'WAITING') {
            deferred.resolve(true);
            return deferred.promise;
        }
        // get an executor from scheduler
        const rmScheduleResult: RemoteMachineScheduleResult = 
        this.gpuScheduler.scheduleMachine(0, environment.rmMachineMetaDetail);
        if (rmScheduleResult.resultType === ScheduleResultType.REQUIRE_EXCEED_TOTAL) {
            const errorMessage: string = `Required GPU number 0 is too large, no machine can meet`;
            this.log.error(errorMessage);
            deferred.reject();
            throw new NNIError(NNIErrorNames.RESOURCE_NOT_AVAILABLE, errorMessage);
        } else if (rmScheduleResult.resultType === ScheduleResultType.SUCCEED
            && rmScheduleResult.scheduleInfo !== undefined) {
            const rmScheduleInfo: RemoteMachineScheduleInfo = rmScheduleResult.scheduleInfo;

            environment.rmMachineMetaDetail.rmMeta = rmScheduleInfo.rmMeta;
            const copyExpCodeDirPromise = this.machineCopyExpCodeDirPromiseMap.get(
                environment.rmMachineMetaDetail.rmMeta);
            if (copyExpCodeDirPromise !== undefined) {
                await copyExpCodeDirPromise;
            }

            this.allocateExecutorManagerForEnvironment(environment);
            const executor = await this.getExecutor(environment.id);
            environment.runnerWorkingFolder = 
                executor.joinPath(executor.getRemoteExperimentRootDir(getExperimentId()), 
                'envs', environment.id)

            await this.launchEnvironmentOnScheduledMachine(environment);

            environment.status = 'RUNNING';
            environment.trackingUrl = `file://${rmScheduleInfo.rmMeta.ip}:${environment.runnerWorkingFolder}`;

            this.environmentJobsMap.set(environment.id, environment);
            deferred.resolve(true);
        } else if (rmScheduleResult.resultType === ScheduleResultType.TMP_NO_AVAILABLE_GPU) {
            this.log.info(`Right now no available GPU can be allocated for trial ${environment.id}, will try to schedule later`);
            deferred.resolve(false);
        } else {
            deferred.reject(`Invalid schedule resutl type: ${rmScheduleResult.resultType}`);
        }

        return deferred.promise;
    }

    /**
     * give environment an executor
     * @param environment RemoteMachineEnvironmentDetail
     */
    public allocateExecutorManagerForEnvironment(environment: RemoteMachineEnvironmentInformation): void {
        if (environment.rmMachineMetaDetail === undefined) {
            throw new Error(`rmMeta not set in trial ${environment.id}`);
        }
        if (environment.rmMachineMetaDetail.rmMeta === undefined) {
            throw new Error(`rmMeta not set in trial ${environment.id}`);
        }
        const executorManager: ExecutorManager | undefined = this.machineExecutorManagerMap.get(environment.rmMachineMetaDetail.rmMeta);
        if (executorManager === undefined) {
            throw new Error(`executorManager not initialized`);
        }
        this.environmentExecutorManagerMap.set(environment.id, executorManager);
    }

    private async getExecutor(environmentId: string): Promise<ShellExecutor> {
        const executorManager = this.environmentExecutorManagerMap.get(environmentId);
        if (executorManager === undefined) {
            throw new Error(`ExecutorManager is not assigned for environment ${environmentId}`);
        }
        return await executorManager.getExecutor(environmentId);
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
    }

    private async launchEnvironmentOnScheduledMachine(environment: RemoteMachineEnvironmentInformation): Promise<void> {
        if (this.trialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        const executor = await this.getExecutor(environment.id);
        const environmentLocalTempFolder: string =  
            path.join(this.experimentRootDir, this.experimentId, environment.id);
        await executor.createFolder(environment.runnerWorkingFolder);
        await execMkdir(environmentLocalTempFolder);
        await fs.promises.writeFile(path.join(environmentLocalTempFolder, executor.getScriptName("run")),
        environment.command, { encoding: 'utf8' });
        // Copy files in codeDir to remote working directory
        await executor.copyDirectoryToRemote(environmentLocalTempFolder, environment.runnerWorkingFolder);
        // Execute command in remote machine
        executor.executeScript(executor.joinPath(environment.runnerWorkingFolder, executor.getScriptName("run")), true, true);
    }
}
