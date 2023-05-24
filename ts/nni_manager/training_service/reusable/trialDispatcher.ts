// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import { Writable } from 'stream';
import { String } from 'typescript-string-operations';
import { NNIError, NNIErrorNames, MethodNotImplementedError } from 'common/errors';
import { getBasePort, getExperimentId } from 'common/experimentStartupInfo';
import { IocShim } from 'common/ioc_shim';
import { getLogger, Logger } from 'common/log';
import { TrainingService, TrialJobApplicationForm, TrialJobMetric, TrialJobStatus } from 'common/trainingService';
import { delay, getExperimentRootDir, getIPV4Address, getLogLevel, getVersion, mkDirPSync, randomSelect, uniqueString } from 'common/utils';
import { ExperimentConfig, SharedStorageConfig } from 'common/experimentConfig';
import { GPU_INFO, INITIALIZED, KILL_TRIAL_JOB, NEW_TRIAL_JOB, REPORT_METRIC_DATA, SEND_TRIAL_JOB_PARAMETER, STDOUT, TRIAL_END, VERSION_CHECK } from 'core/commands';
import { ScheduleResultType } from 'training_service/common/gpuData';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT_FOR_WIN } from '../common/containerJobData';
import { TrialConfig } from '../common/trialConfig';
import { validateCodeDir } from '../common/util';
import { Command, CommandChannel } from './commandChannel';
import { EnvironmentInformation, EnvironmentService, NodeInformation, RunnerSettings, TrialGpuSummary } from './environment';
import { createEnvironmentService } from './environments/environmentServiceFactory';
import { GpuScheduler } from './gpuScheduler';
import { MountedStorageService } from './storages/mountedStorageService';
import { StorageService } from './storageService';
import { SharedStorageService } from './sharedStorage';
import { NFSSharedStorageService } from './shared_storages/nfsStorageService'
import { AzureBlobSharedStorageService } from './shared_storages/azureblobStorageService'
import { TrialDetail } from './trial';

/**
 * It uses to manage jobs on training platforms 
 * and expose trial as trial job to upper level.
**/
//@component.Singleton  MARK
class TrialDispatcher implements TrainingService {
    private log: Logger;
    private isDeveloping: boolean = false;
    private stopping: boolean = false;

    private metricsEmitter: EventEmitter;
    private experimentId: string;
    private experimentRootDir: string;

    private enableVersionCheck: boolean = true;

    private trialConfig: TrialConfig | undefined;

    private trials: Map<string, TrialDetail>;
    private environments: Map<string, EnvironmentInformation>;
    // make public for ut
    public environmentServiceList: EnvironmentService[] = [];
    public commandChannelSet: Set<CommandChannel>;
    public commandEmitter: EventEmitter;
    public environmentMaintenceLoopInterval: number = -1;

    // uses to accelerate trial manager loop
    // true means there is updates, and trial loop should run a cycle immediately.
    private shouldUpdateTrials: boolean = true;
    // uses to decide environment assign strategy.
    // true means use gpu scheduler to decide if there is free resource for new trial.
    // false means one env run one trial in same time.
    private enableGpuScheduler: boolean = false;
    // uses to save if user like to reuse environment
    private reuseEnvironment: boolean = true;
    private logCollection: string = 'none';

    private gpuScheduler: GpuScheduler;

    // uses to reduce log count.
    private isLoggedNoMoreEnvironment: boolean = false;
    private isLoggedNoGpuAvailable: boolean = false;

    // uses to mark whether to use shared storage
    private useSharedStorage: boolean = false;
    private fileCopyCompleted: boolean = false;

    private config: ExperimentConfig;

    public static async construct(config: ExperimentConfig): Promise<TrialDispatcher> {
        const instance = new TrialDispatcher(config);
        await instance.asyncConstructor(config);
        return instance;
    }

    private constructor(config: ExperimentConfig) {
        this.log = getLogger('TrialDispatcher');
        this.trials = new Map<string, TrialDetail>();
        this.environments = new Map<string, EnvironmentInformation>();
        this.metricsEmitter = new EventEmitter();
        this.experimentId = getExperimentId();
        this.experimentRootDir = getExperimentRootDir();
        this.commandChannelSet = new Set<CommandChannel>();

        const logLevel = getLogLevel();
        this.log.debug(`current folder ${__dirname}`);
        // different source folder in Linux and Windows
        if (logLevel == "debug" && (fs.existsSync("../../../src/nni_manager") || __dirname.endsWith("src\\nni_manager\\dist\\training_service\\reusable"))) {
            this.log.debug("log level is debug, and exist code folder, so set to developing mode.");
            this.isDeveloping = true;
        }

        this.commandEmitter = new EventEmitter();

        this.gpuScheduler = new GpuScheduler();

        this.config = config;

        this.enableGpuScheduler = !!config.trialGpuNumber;
        if (this.enableGpuScheduler) {
            this.log.info(`TrialDispatcher: GPU scheduler is enabled.`)
        }
    }

    private async asyncConstructor(config: ExperimentConfig): Promise<void> {
        await validateCodeDir(config.trialCodeDirectory);

        const serviceConfigs = Array.isArray(config.trainingService) ? config.trainingService : [ config.trainingService ];
        const servicePromises = serviceConfigs.map(serviceConfig => createEnvironmentService(serviceConfig));
        this.environmentServiceList = await Promise.all(servicePromises);

        this.environmentMaintenceLoopInterval = Math.max(
            ...this.environmentServiceList.map((env) => env.environmentMaintenceLoopInterval)
        );

        for (const env of this.environmentServiceList) {
            env.initCommandChannel(this.commandEmitter);
            this.commandChannelSet.add(env.getCommandChannel);
        }

        if (this.config.sharedStorage !== undefined) {
            await this.initializeSharedStorage(this.config.sharedStorage);
        }
    }

    public async listTrialJobs(): Promise<TrialDetail[]> {
        const trials: TrialDetail[] = [];

        for (const key of this.trials.keys()) {
            trials.push(await this.getTrialJob(key));
        }

        return trials;
    }

    public async getTrialJob(trialJobId: string): Promise<TrialDetail> {
        const trial: TrialDetail | undefined = this.trials.get(trialJobId);
        if (trial === undefined) {
            throw new Error(`trial job ${trialJobId} not found`);
        }

        return trial;
    }

    public async getTrialFile(_trialJobId: string, _fileName: string): Promise<string | Buffer> {
        throw new MethodNotImplementedError();
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialDetail> {
        const trialId: string = form.id === undefined ? uniqueString(5) : form.id;

        const trialJobDetail: TrialDetail = new TrialDetail(trialId, "WAITING", Date.now(), "", form);

        this.trials.set(trialId, trialJobDetail);

        return trialJobDetail;
    }

    // to support multi phase
    public async updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialDetail> {
        const trialDetail = await this.getTrialJob(trialJobId);
        const environment = trialDetail.environment;
        if (environment === undefined) {
            throw new Error(`TrialDispatcher: trial ${trialJobId}'s env shouldn't be undefined in updateTrialJob.`);
        }
        if (environment.environmentService === undefined) {
            throw new Error(`Environment ${environment.id} does not assigned environment service.`);
        }

        const message = {
            "trialId": trialJobId,
            "parameters": form.hyperParameters,
        }
        await environment.environmentService.getCommandChannel.sendCommand(environment, SEND_TRIAL_JOB_PARAMETER, message);

        return trialDetail;
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean | undefined): Promise<void> {
        const trial = await this.getTrialJob(trialJobId);
        switch (trial.status) {
            case "RUNNING":
            case "WAITING":
            case "UNKNOWN":
                {
                    const environment = trial.environment;
                    if (environment && environment.environmentService) {
                        await environment.environmentService.getCommandChannel.sendCommand(environment, KILL_TRIAL_JOB, trial.id);
                        trial.isEarlyStopped = isEarlyStopped;
                        trial.status = trial.isEarlyStopped === true ?
                            'EARLY_STOPPED' : 'USER_CANCELED';
                        this.releaseEnvironment(trial);
                    }
                }
                break;
        }
    }

    private getStorageService(environmentService: EnvironmentService): StorageService {
        let storageService: StorageService;
        if (this.useSharedStorage) {
            this.log.debug(`TrialDispatcher: use shared storage service.`);
            storageService = IocShim.get<SharedStorageService>(SharedStorageService).storageService;
        } else if (environmentService.hasStorageService) {
            this.log.debug(`TrialDispatcher: use existing storage service.`);
            storageService = IocShim.get<StorageService>(StorageService);
        } else {
            this.log.debug(`TrialDispatcher: create temp storage service to temp folder.`);
            storageService = new MountedStorageService();
            const environmentLocalTempFolder = path.join(this.experimentRootDir, "environment-temp");
            storageService.initialize(this.config.trialCodeDirectory, environmentLocalTempFolder);
        }
        return storageService;
    }
    public async run(): Promise<void> {
        await Promise.all(this.environmentServiceList.map(env => env.init()));
        for(const environmentService of this.environmentServiceList) {
            
            

            await environmentService.getCommandChannel.start();
            this.log.info(`TrialDispatcher: started channel: ${environmentService.getCommandChannel.constructor.name}`);
    
            this.log.info(`TrialDispatcher: copying code.`);
            if (this.useSharedStorage) {
                if (this.fileCopyCompleted) {
                    continue;
                }
            }
            const storageService: StorageService = this.getStorageService(environmentService);

            // Copy the compressed file to remoteDirectory and delete it
            const codeDir = path.resolve(this.config.trialCodeDirectory);
            const envDir = storageService.joinPath("envs");
            const codeFileName = await storageService.copyDirectory(codeDir, envDir, true);
            storageService.rename(codeFileName, "nni-code.tar.gz");

            const installFileName = storageService.joinPath(envDir, `install_nni.sh`);
            const installFileNameForWin = storageService.joinPath(envDir, `install_nni.ps1`);
            await storageService.save(CONTAINER_INSTALL_NNI_SHELL_FORMAT, installFileName);
            await storageService.save(CONTAINER_INSTALL_NNI_SHELL_FORMAT_FOR_WIN, installFileNameForWin);

            if (this.isDeveloping) {
                let trialToolsPath = path.join(__dirname, "../../../../../tools/nni_trial_tool");
                if (false === fs.existsSync(trialToolsPath)) {
                    trialToolsPath = path.join(__dirname, "..\\..\\..\\..\\..\\tools\\nni_trial_tool");
                }
                await storageService.copyDirectory(trialToolsPath, envDir, true);
            }

            if (this.useSharedStorage) {
                this.fileCopyCompleted = true;
            }
        }
        // start channel
        this.commandEmitter.on("command", (command: Command): void => {
            this.handleCommand(command).catch((err: Error) => {
                this.log.error(`TrialDispatcher: error on handle env ${command.environment.id} command: ${command.command}, data: ${command.data}, error: ${err}`);
            })
        });
        await this.prefetchEnvironments();
        this.log.info(`TrialDispatcher: run loop started.`);
        const promiseList: Promise<void>[] = [];
        for(const commandChannel of this.commandChannelSet) {
            promiseList.push(commandChannel.run());
        }
        promiseList.push(this.environmentMaintenanceLoop());
        promiseList.push(this.trialManagementLoop());
        await Promise.all(promiseList);
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.off('metric', listener);
    }

    public async setClusterMetadata(_key: string, _value: string): Promise<void> { return; }
    public async getClusterMetadata(_key: string): Promise<string> { return ""; }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        if (environment.environmentService === undefined) {
            throw new Error(`${environment.id} do not have environmentService!`);
        }
        this.log.info(`stopping environment ${environment.id}...`);
        await environment.environmentService.stopEnvironment(environment);
        this.log.info(`stopped environment ${environment.id}.`);
        return;
    }

    public async cleanUp(): Promise<void> {
        if (this.commandEmitter === undefined) {
            throw new Error(`TrialDispatcher: commandEmitter shouldn't be undefined in cleanUp.`);
        }
        this.stopping = true;
        this.shouldUpdateTrials = true;
        const environments = [...this.environments.values()];
        
        const stopEnvironmentPromise: Promise<void>[] = []; 
        for (let index = 0; index < environments.length; index++) {
            stopEnvironmentPromise.push(this.stopEnvironment(environments[index]));
        }
        await Promise.all(stopEnvironmentPromise);
        this.commandEmitter.off("command", this.handleCommand);
        for (const commandChannel of this.commandChannelSet) {
            await commandChannel.stop();
        }
        if (this.useSharedStorage) {
            this.log.info(`stopping shared storage...`)
            await IocShim.get<SharedStorageService>(SharedStorageService).cleanUp();
            this.log.info(`shared storage stopped.`)
        }
    }

    private async environmentMaintenanceLoop(): Promise<void> {
        while (!this.stopping) {
            const environments: EnvironmentInformation[] = [];
            for (const environment of this.environments.values()) {
                if (environment.isAlive === true) {
                    environments.push(environment);
                } else {
                    if (environment.environmentService === undefined) {
                        throw new Error(`${environment.id} do not have environment service!`);
                    }
                    await environment.environmentService.getCommandChannel.close(environment);
                }
            }
            // Group environments according to environmentService
            const environmentServiceDict: Map<EnvironmentService, EnvironmentInformation[]> =
                new Map<EnvironmentService, EnvironmentInformation[]>();
            for (const environment of environments) {
                if (environment.environmentService === undefined) {
                    throw new Error(`${environment.id} do not have environment service!`);
                }
                if (!environmentServiceDict.has(environment.environmentService)) {
                    environmentServiceDict.set(environment.environmentService, [environment]);
                } else {
                    const environmentsList: EnvironmentInformation[] | undefined = environmentServiceDict.get(environment.environmentService);
                    if (environmentsList === undefined) {
                        throw new Error(`Environment list not initialized!`);
                    }
                    environmentsList.push(environment);
                    environmentServiceDict.set(environment.environmentService, environmentsList);
                }
            }
            // Refresh all environments
            const taskList: Promise<void>[] = [];
            for (const environmentService of environmentServiceDict.keys()) {
                const environmentsList: EnvironmentInformation[] | undefined = environmentServiceDict.get(environmentService);
                if (environmentsList) {
                    taskList.push(environmentService.refreshEnvironmentsStatus(environmentsList));
                }
            }
            await Promise.all(taskList);

            for (const environment of environments) {
                if (environment.environmentService === undefined) {
                    throw new Error(`${environment.id} do not have environment service!`);
                }
                const oldIsAlive = environment.isAlive;
                switch (environment.status) {
                    case 'WAITING':
                    case 'RUNNING':
                    case 'UNKNOWN':
                        environment.isAlive = true;
                        break;
                    default:
                        environment.isAlive = false;
                        break;
                }
                if (oldIsAlive !== environment.isAlive) {
                    this.log.debug(`set environment ${environment.id} isAlive from ${oldIsAlive} to ${environment.isAlive} due to status is ${environment.status}.`);
                }
            }
            this.shouldUpdateTrials = true;
            if (this.environmentMaintenceLoopInterval === -1) {
                throw new Error("EnvironmentMaintenceLoopInterval not initialized!");
            }
            await delay(this.environmentMaintenceLoopInterval);
        }
    }

    private async trialManagementLoop(): Promise<void> {
        const interval = 1;

        while (!this.stopping) {
            let totalInterval = 1000;
            while (totalInterval > 0) {
                if (this.shouldUpdateTrials) {
                    this.shouldUpdateTrials = false;
                    break;
                }
                totalInterval -= interval;
                await delay(interval);
            }

            const toRefreshedTrials: TrialDetail[] = [];
            for (const trial of this.trials.values()) {
                if (trial.status === "RUNNING" || trial.status === "WAITING" || trial.status === "UNKNOWN") {
                    toRefreshedTrials.push(trial);
                }
            }

            if (toRefreshedTrials.length == 0) {
                continue;
            }

            let waitingTrials: TrialDetail[] = [];
            let liveTrialsCount = 0;
            for (const trial of toRefreshedTrials) {
                const currentStatus = trial.status;
                switch (currentStatus) {
                    case "RUNNING":
                        {
                            const environment = trial.environment;

                            if (environment === undefined) {
                                this.log.error(`found running trial ${trial.id} has no environment, set trial to UNKNOWN.`);
                                trial.status = "UNKNOWN";
                                liveTrialsCount++;
                                continue;
                            }

                            if (environment.environmentService === undefined) {
                                throw new Error(`${environment.id} does not has environment service!`);
                            }
                            trial.url = environment.trackingUrl;
                            const environmentStatus = environment.status;

                            // any node exit, then make sure the whole trial stopped.
                            if (trial.nodes.size > 0) {
                                const completedCount = trial.nodes.size;
                                let finalStatus: TrialJobStatus = "SUCCEEDED";
                                let lastTimestamp: number | undefined;
                                this.log.debug(`found ${completedCount} completed trial node(s), nodeCount: ${environment.nodeCount}`);

                                // if some trial processes doesn't exit, kill it for next one.
                                // for example, in horovod, it's just sleep command, has no impact on trial result.
                                if (environment.nodeCount > completedCount) {
                                    this.log.info(`stop partial completed trial ${trial.id}`);
                                    await environment.environmentService.getCommandChannel.sendCommand(environment, KILL_TRIAL_JOB, trial.id);
                                }
                                for (const node of trial.nodes.values()) {
                                    if (node.status === "FAILED") {
                                        finalStatus = "FAILED";
                                    }
                                    if (node.endTime !== undefined) {
                                        if (lastTimestamp === undefined) {
                                            lastTimestamp = node.endTime
                                        } else {
                                            lastTimestamp = Math.max(node.endTime, lastTimestamp);
                                        }
                                    }
                                }
                                trial.status = finalStatus;
                                if (lastTimestamp === undefined) {
                                    trial.endTime = lastTimestamp;
                                }
                                this.releaseEnvironment(trial);
                            } else if (environmentStatus !== "RUNNING") {
                                this.log.error(`found running trial ${trial.id} on '${environment.envId}' with '${environmentStatus}', set trial to environment status.`);
                                this.releaseEnvironment(trial);
                                trial.status = environmentStatus;
                            } else {
                                liveTrialsCount++;
                            }
                        }
                        break;
                    case "WAITING":
                    case "UNKNOWN":
                        // deal it later, if there is free environment.
                        waitingTrials.push(trial);
                        liveTrialsCount++;
                        break;
                }
            }

            let liveEnvironmentsCount = 0;
            const reusableEnvironments: EnvironmentInformation[] = [];
            for (const environment of this.environments.values()) {
                if (environment.isAlive === true) {
                    liveEnvironmentsCount++;
                    if (environment.status === "RUNNING" && environment.isRunnerReady) {
                        // if environment is not reusable and used, stop and not count as idle;
                        const reuseMode = Array.isArray(this.config.trainingService) || (this.config.trainingService as any).reuseMode;
                        if (
                            0 === environment.runningTrialCount &&
                            reuseMode === false &&
                            environment.assignedTrialCount > 0
                        ) {
                            if (environment.environmentService === undefined) {
                                throw new Error(`${environment.id} does not has environment service!`);
                            }
                            await environment.environmentService.stopEnvironment(environment);
                            liveEnvironmentsCount--;
                            continue;
                        }

                        // if gpu scheduler is not enabled, and there is running trial, skip it.
                        if (false === this.enableGpuScheduler && environment.runningTrialCount > 0) {
                            continue;
                        }

                        reusableEnvironments.push(environment);
                    }
                }
            }

            let neededEnvironmentCount = 0;
            if (true === this.enableGpuScheduler) {
                let noGpuAvailable: boolean = false;
                while (waitingTrials.length > 0) {
                    // skip following trials, if first trial doesn't find available GPU.
                    if (true === noGpuAvailable) {
                        // break loop to try next time.
                        break;
                    }
                    const trial = waitingTrials.shift();
                    if (undefined === trial) {
                        throw new Error(`TrialDispatcher: waiting trial shouldn't be undefined!`);
                    }
                    const defaultGpuNum = this.config.trialGpuNumber;
                    const result = this.gpuScheduler.scheduleMachine(reusableEnvironments, trial.form.placementConstraint!, defaultGpuNum, trial);
                    switch (result.resultType) {
                        case ScheduleResultType.REQUIRE_EXCEED_TOTAL:
                            {
                                if (liveEnvironmentsCount == 0) {
                                    this.log.debug(`TrialDispatcher: no live environment, so request one.`);
                                    neededEnvironmentCount = 1;
                                    waitingTrials = [];
                                    this.isLoggedNoGpuAvailable = false;
                                } else if (reusableEnvironments.length > 0) {
                                    const errorMessage: string = `TrialDispatcher: REQUIRE_EXCEED_TOTAL Required GPU number ${defaultGpuNum} is too large, no machine can meet`;
                                    this.log.error(errorMessage);
                                    throw new NNIError(NNIErrorNames.RESOURCE_NOT_AVAILABLE, errorMessage);
                                } else {
                                    if (false === this.isLoggedNoGpuAvailable) {
                                        this.log.debug(`TrialDispatcher: wait GPU, live environment ${liveEnvironmentsCount}, no reusable, REQUIRE_EXCEED_TOTAL.`)
                                        this.isLoggedNoGpuAvailable = true;
                                    }
                                }
                                break;
                            }
                        case ScheduleResultType.TMP_NO_AVAILABLE_GPU:
                            {
                                if (false === this.isLoggedNoGpuAvailable) {
                                    this.log.debug(`TrialDispatcher: wait GPU, live environment ${liveEnvironmentsCount}, reusable ${reusableEnvironments.length}, TMP_NO_AVAILABLE_GPU.`)
                                    this.isLoggedNoGpuAvailable = true;
                                }

                                // if some environment is alive, but not ready, no need to create more.
                                if (liveEnvironmentsCount <= reusableEnvironments.length) {
                                    neededEnvironmentCount = 1;
                                    this.isLoggedNoGpuAvailable = false;
                                    this.log.info(`TrialDispatcher: ${liveEnvironmentsCount} live env, and ${reusableEnvironments.length} reusable, but no GPU available so request a new one.`);
                                }
                                noGpuAvailable = true;
                            }
                            break
                        case ScheduleResultType.SUCCEED:
                            {
                                const environment = result.environment;
                                if (undefined === environment) {
                                    throw new Error(`TrialDispatcher: scheduled env shouldn't be undefined!`);
                                }
                                trial.assignedGpus = result.gpuIndices;
                                await this.allocateEnvironment(trial, environment);
                                this.isLoggedNoGpuAvailable = false;
                            }
                            break
                        default:
                            throw new Error(`TrialDispatcher: Unknown gpu schecduler type: ${result.resultType}`);
                    }
                }
            } else {
                while (reusableEnvironments.length > 0 && waitingTrials.length > 0) {
                    const trial = waitingTrials.shift();
                    const idleEnvironment = reusableEnvironments.shift();
                    if (trial !== undefined && idleEnvironment != undefined) {
                        await this.allocateEnvironment(trial, idleEnvironment);
                    }
                }
                neededEnvironmentCount = liveTrialsCount - liveEnvironmentsCount;
            }

            if (neededEnvironmentCount > 0) {
                let requestedCount = 0;
                let hasMoreEnvironments = false;
                for (let index = 0; index < neededEnvironmentCount; index++) {
                    const environmentService: EnvironmentService | undefined = this.selectEnvironmentService();
                    if (environmentService !== undefined) {
                        hasMoreEnvironments = true;
                        await this.requestEnvironment(environmentService);
                        requestedCount++;
                        this.isLoggedNoMoreEnvironment = false;
                    } else {
                        if (this.isLoggedNoMoreEnvironment === false) {
                            this.isLoggedNoMoreEnvironment = true;
                            this.log.info(`no more environment so far, so skip to request environment.`)
                        }
                    }
                }
                if (hasMoreEnvironments === true || requestedCount > 0) {
                    this.log.info(`requested new environment, live trials: ${liveTrialsCount}, ` +
                        `live environments: ${liveEnvironmentsCount}, neededEnvironmentCount: ${neededEnvironmentCount}, ` +
                        `requestedCount: ${requestedCount}`);
                }
            }

        }
    }

    // Schedule a environment platform for environment
    private selectEnvironmentService(): EnvironmentService | undefined {
        const validEnvironmentServiceList = [];
        for(const environmentService of this.environmentServiceList){
            if (environmentService.hasMoreEnvironments) {
                validEnvironmentServiceList.push(environmentService);
            }
        }
        if (validEnvironmentServiceList.length === 0) {
            return undefined;
        }
        // Random scheduler
        return randomSelect(validEnvironmentServiceList);
    }

    private async prefetchEnvironments (): Promise<void> {
        for (const environmentService of this.environmentServiceList) {
            const number = environmentService.prefetchedEnvironmentCount;
            this.log.info(`Initialize environments total number: ${number}`);
            for (let index = 0; index < number; index++) {
                await this.requestEnvironment(environmentService);
            }
        }
    }

    private async setEnvironmentSetting(environment: EnvironmentInformation): Promise<void> {
        if (environment.environmentService === undefined) {
            throw new Error(`Environmentservice for ${environment.id} not initialized!`);
        }
        const environmentService = environment.environmentService;
        const runnerSettings: RunnerSettings = new RunnerSettings();
        runnerSettings.nniManagerIP = this.config.nniManagerIp === undefined? await getIPV4Address() : this.config.nniManagerIp;
        runnerSettings.nniManagerPort = getBasePort() + 1;
        runnerSettings.commandChannel = environmentService.getCommandChannel.channelName;
        runnerSettings.enableGpuCollector = this.enableGpuScheduler;
        runnerSettings.command = this.config.trialCommand;
        runnerSettings.nniManagerVersion = this.enableVersionCheck ? await getVersion() : '';
        runnerSettings.logCollection = this.logCollection;
        runnerSettings.platform = environmentService.getName;
        runnerSettings.experimentId = this.experimentId;
        const storageService: StorageService = this.getStorageService(environmentService);
        const envDir = storageService.joinPath("envs");
        const runnerSettingsConfig = storageService.joinPath(envDir, environment.id, "settings.json");
        await storageService.save(JSON.stringify(runnerSettings), runnerSettingsConfig);
    }

    private async requestEnvironment(environmentService: EnvironmentService): Promise<void> {
        if (this.stopping) {
            this.log.info(`Experiment is stopping, stop creating new environment`);
            return;
        }
        const envId = uniqueString(5);
        const envName = `nni_exp_${this.experimentId}_env_${envId}`;
        const environment = environmentService.createEnvironmentInformation(envId, envName);
        environment.environmentService = environmentService;
        this.log.info(`Assign environment service ${environmentService.getName} to environment ${envId}`);
        environment.command = `sh ../install_nni.sh && python3 -m nni.tools.trial_tool.trial_runner`;

        if (this.isDeveloping) {
            environment.command = "[ -d \"nni_trial_tool\" ] && echo \"nni_trial_tool exists already\" || (mkdir ./nni_trial_tool && tar -xof ../nni_trial_tool.tar.gz -C ./nni_trial_tool) && pip3 install websockets && " + environment.command;
        }

        environment.command = `mkdir -p envs/${envId} && cd envs/${envId} && ${environment.command}`;

        environment.useSharedStorage = this.useSharedStorage;
        // Generate setting.json file per environment to avoid conflict
        await this.setEnvironmentSetting(environment);

        await environmentService.startEnvironment(environment);
        this.environments.set(environment.id, environment);

        if (environment.status === "FAILED") {
            environment.isAlive = false;
            throw new Error(`error on request environment ${environment.envId}, please check log for more details.`);
        } else {
            environment.isAlive = true;
        }
        await environment.environmentService.getCommandChannel.open(environment);
        this.log.info(`requested environment ${environment.id} and job id is ${environment.envId}.`);
    }

    private async allocateEnvironment(trial: TrialDetail, environment: EnvironmentInformation): Promise<void> {
        if (trial.environment) {
            throw new Error(`TrialDispatcher: trial ${trial.id} has assigned environment ${trial.environment.id} already, not assign to ${environment.id}!`);
        }
        if (environment.runningTrialCount > 0 && false === this.enableGpuScheduler) {
            throw new Error(`TrialDispatcher: environment ${environment.id} has running trial, and gpu scheduler is not enabled, it cannot be assigned again!`);
        }
        this.log.info(`assigning environment ${environment.id} to trial ${trial.id}.`);

        // convert assigned gpus to string for nvidia visible settings
        // undefined means no constraint, [] means no gpu visible.
        let gpuIndices: string | undefined = undefined;
        if (undefined !== this.config.trialGpuNumber) {
            const gpuArray: number[] = [];
            if (undefined !== trial.assignedGpus) {
                trial.assignedGpus.map((value) => {
                    gpuArray.push(value.index);
                });
            }
            gpuIndices = gpuArray.join(',');
        }

        environment.runningTrialCount++;
        environment.assignedTrialCount++;
        trial.environment = environment;
        if (environment.environmentService === undefined) {
            throw new Error(`${environment.id} environmentService not initialized!`);
        }
        trial.message = `Platform: ${environment.environmentService.getName}, environment: ${environment.id}`;
        if (this.useSharedStorage) {
            const storageService = IocShim.get<SharedStorageService>(SharedStorageService).storageService;
            trial.workingDirectory = storageService.joinPath('trials', trial.id);
        } else if (environment.environmentService.hasStorageService) {	
            const storageService = IocShim.get<StorageService>(StorageService);
            trial.workingDirectory = storageService.joinPath('trials', trial.id);
        }	
        trial.settings = {
            trialId: trial.id,
            gpuIndices: gpuIndices,
            sequenceId: trial.form.sequenceId,
            parameter: trial.form.hyperParameters,
        }
        trial.startTime = Date.now();
        trial.status = "RUNNING";
        if (environment.environmentService === undefined) {
            throw new Error(`${environment.id} does not have environment service!`);
        }
        await environment.environmentService.getCommandChannel.sendCommand(trial.environment, NEW_TRIAL_JOB, trial.settings);
    }
    
    /**
     * release the trial assigned environment resources
     * @param trial 
     */
    private releaseEnvironment(trial: TrialDetail): void {
        if (true === this.enableGpuScheduler) {
            this.gpuScheduler.removeGpuReservation(trial);
        }
        if (trial.environment !== undefined) {
            if (trial.environment.runningTrialCount <= 0) {
                throw new Error(`TrialDispatcher: environment ${trial.environment.id} has no counted running trial!`);
            }
            trial.environment.runningTrialCount--;
            trial.environment.latestTrialReleasedTime = Date.now();
            trial.environment = undefined;
        }
    }

    private async handleMetricData(trialId: string, data: any): Promise<void> {
        if (Array.isArray(data)) {
            for (const subItem of data) {
                this.metricsEmitter.emit('metric', {
                    id: trialId,
                    data: subItem
                });
            }
        } else {
            this.metricsEmitter.emit('metric', {
                id: trialId,
                data: data
            });
        }
    }

    private async handleStdout(commandData: any): Promise<void> {
        const metricPattern: RegExp = /NNISDK_MEb'(?<metrics>.*a?)'$/gm;
        const trialLogDir: string = path.join(getExperimentRootDir(), 'trials', commandData["trial"]);
        mkDirPSync(trialLogDir);
        const trialLogPath: string = path.join(trialLogDir, 'stdout_log_collection.log');
        try {
            let skipLogging: boolean = false;
            if (commandData["tag"] === 'trial' && commandData["msg"] !== undefined) {
                const message: string = commandData["msg"];
                let metricsContent = metricPattern.exec(message);
                while (metricsContent && metricsContent.groups) {
                    const key: string = 'metrics';
                    const data = metricsContent.groups[key];
                    await this.handleMetricData(commandData["trial"], data);
                    metricsContent = metricPattern.exec(message);
                    skipLogging = true;
                }
            }

            if (!skipLogging) {
                // Construct write stream to write remote trial's log into local file
                const writeStream: Writable = fs.createWriteStream(trialLogPath, {
                    flags: 'a+',
                    encoding: 'utf8',
                    autoClose: true
                });

                writeStream.write(String.Format('{0}\n', commandData["msg"]));
                writeStream.end();
            }
        } catch (err) {
            this.log.error(`TrialDispatcher: handleStdout error: ${err}`);
        }
    }

    private async handleCommand(command: Command): Promise<void> {
        this.log.debug(`TrialDispatcher: env ${command.environment.id} received command ${command.command}.`);
        const environment = command.environment;
        const data = command.data;
        const nodeId = data["node"];
        switch (command.command) {
            case REPORT_METRIC_DATA:
                this.log.error(`TrialDispatcher: TODO: not implement to handle direct REPORT_METRIC_DATA command yet.`);
                break;
            case STDOUT:
                await this.handleStdout(data);
                break;
            case INITIALIZED:
                {
                    let isAllReady = true;
                    if (environment.nodeCount > 1) {
                        let node = environment.nodes.get(nodeId);
                        if (node === undefined) {
                            node = new NodeInformation(nodeId);
                            environment.nodes.set(nodeId, node);
                        }
                        const oldNodeStatus = node.status;
                        if (oldNodeStatus === "UNKNOWN" || oldNodeStatus === "WAITING") {
                            node.status = "RUNNING";
                        }

                        if (environment.nodes.size === environment.nodeCount) {
                            for (const node of environment.nodes.values()) {
                                if (node.status !== "RUNNING") {
                                    isAllReady = false;
                                    break;
                                }
                            }
                        } else {
                            isAllReady = false;
                        }
                    }

                    // single node is always ready to set env status
                    if (isAllReady) {
                        environment.isRunnerReady = true;
                        this.log.info(`TrialDispatcher: env ${environment.id} received initialized message and runner is ready, env status: ${environment.status}.`);
                    }
                }
                break;
            case VERSION_CHECK:
                {
                    if (this.enableVersionCheck) {
                        const checkResultSuccess: boolean = data["tag"] === 'VCSuccess' ? true : false;
                        if (checkResultSuccess) {
                            this.log.info(`TrialDispatcher: Version check in trialKeeper success!`);
                        } else {
                            const errorMessage = `TrialDispatcher: Version check error, ${data["msg"]}!`;
                            this.log.error(errorMessage);
                        }
                    }
                }
                break;
            case GPU_INFO:
                {
                    const gpuData = <TrialGpuSummary>(data);
                    environment.setGpuSummary(nodeId, gpuData);
                }
                break;
            case TRIAL_END:
                {
                    const trialId = data["trial"];
                    const trial = await this.getTrialJob(trialId);
                    const code = parseInt(data["code"]);
                    const timestamp = parseInt(data["time"]);
                    let exitStatus: TrialJobStatus = "SUCCEEDED";
                    if (code !== 0) {
                        exitStatus = "FAILED";
                    }

                    let node = environment.nodes.get(nodeId);
                    if (node === undefined) {
                        node = new NodeInformation(nodeId);
                        trial.nodes.set(nodeId, node);
                    }
                    if (undefined === node) {
                        throw new Error("node is impossible to be undefined (see above code), but make eslint happy!");
                    }
                    node.status = exitStatus;
                    node.endTime = timestamp;
                }
                break;
        }
        this.shouldUpdateTrials = true;
    }

    private async initializeSharedStorage(config: SharedStorageConfig): Promise<void> {
        switch (config.storageType) {
            case 'NFS':
                IocShim.bind(SharedStorageService, NFSSharedStorageService);
                break;
            case 'AzureBlob':
                IocShim.bind(SharedStorageService, AzureBlobSharedStorageService);
                break;
            default: {
                const errorMessage = `Shared storage type '${config.storageType}' not support.`;
                this.log.error(errorMessage)
                return Promise.reject(errorMessage);
            }
        }
        await IocShim.get<SharedStorageService>(SharedStorageService).config(config);
        this.useSharedStorage = true;
        return Promise.resolve();
    }

    public async getTrialOutputLocalPath(trialJobId: string): Promise<string> {
        // TODO: support non shared storage
        if (this.useSharedStorage) {
            const localWorkingRoot = IocShim.get<SharedStorageService>(SharedStorageService).localWorkingRoot;
            return Promise.resolve(path.join(localWorkingRoot, 'trials', trialJobId));
        } else {
            return Promise.reject(new Error('Only support shared storage right now.'));
        }
    }

    public async fetchTrialOutput(trialJobId: string, subpath: string | undefined): Promise<void> {
        // TODO: support non shared storage
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

export { TrialDispatcher };
