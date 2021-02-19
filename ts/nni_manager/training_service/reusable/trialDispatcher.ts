// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as path from 'path';
import { Writable } from 'stream';
import { String } from 'typescript-string-operations';
import * as component from '../../common/component';
import { NNIError, NNIErrorNames, MethodNotImplementedError } from '../../common/errors';
import { getBasePort, getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import { NNIManagerIpConfig, TrainingService, TrialJobApplicationForm, TrialJobMetric, TrialJobStatus, LogType } from '../../common/trainingService';
import { delay, getExperimentRootDir, getIPV4Address, getLogLevel, getVersion, mkDirPSync, randomSelect, uniqueString } from '../../common/utils';
import { GPU_INFO, INITIALIZED, KILL_TRIAL_JOB, NEW_TRIAL_JOB, REPORT_METRIC_DATA, SEND_TRIAL_JOB_PARAMETER, STDOUT, TRIAL_END, VERSION_CHECK } from '../../core/commands';
import { ScheduleResultType } from '../../training_service/common/gpuData';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { TrialConfig } from '../common/trialConfig';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { validateCodeDir } from '../common/util';
import { Command, CommandChannel } from './commandChannel';
import { EnvironmentInformation, EnvironmentService, NodeInformation, RunnerSettings, TrialGpuSummary } from './environment';
import { EnvironmentServiceFactory } from './environments/environmentServiceFactory';
import { GpuScheduler } from './gpuScheduler';
import { MountedStorageService } from './storages/mountedStorageService';
import { StorageService } from './storageService';
import { TrialDetail } from './trial';


/**
 * It uses to manage jobs on training platforms 
 * and expose trial as trial job to upper level.
**/
@component.Singleton
class TrialDispatcher implements TrainingService {
    private readonly log: Logger;
    private readonly isDeveloping: boolean = false;
    private stopping: boolean = false;

    private readonly metricsEmitter: EventEmitter;
    private readonly experimentId: string;
    private readonly experimentRootDir: string;

    private enableVersionCheck: boolean = true;

    private trialConfig: TrialConfig | undefined;

    private readonly trials: Map<string, TrialDetail>;
    private readonly environments: Map<string, EnvironmentInformation>;
    // make public for ut
    public environmentServiceList: EnvironmentService[] = [];
    public commandChannelSet: Set<CommandChannel>;
    public commandEmitter: EventEmitter;
    public environmentMaintenceLoopInterval: number = -1;

    private nniManagerIp: string | undefined;

    // uses to accelerate trial manager loop
    // true means there is updates, and trial loop should run a cycle immediately.
    private shouldUpdateTrials: boolean = true;
    // uses to decide environment assign strategy.
    // true means use gpu scheduler to decide if there is free resource for new trial.
    // false means one env run one trial in same time.
    private enableGpuScheduler: boolean = false;
    // uses to save if user like to reuse environment
    private reuseEnvironment: boolean = true;
    private logCollection: string = '';

    private gpuScheduler: GpuScheduler;

    // uses to reduce log count.
    private isLoggedNoMoreEnvironment: boolean = false;
    private isLoggedNoGpuAvailable: boolean = false;

    constructor() {
        this.log = getLogger();
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

    public async getTrialLog(_trialJobId: string, _logType: LogType): Promise<string> {
        throw new MethodNotImplementedError();
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialDetail> {
        if (this.trialConfig === undefined) {
            throw new Error(`trialConfig not initialized!`);
        }

        const trialId: string = uniqueString(5);

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

    public async run(): Promise<void> {
        if (this.trialConfig === undefined) {
            throw new Error(`trial config shouldn't be undefined in run()`);
        }
        for(const environmentService of this.environmentServiceList) {
            
            const runnerSettings: RunnerSettings = new RunnerSettings();
            runnerSettings.nniManagerIP = this.nniManagerIp === undefined? getIPV4Address() : this.nniManagerIp;
            runnerSettings.nniManagerPort = getBasePort() + 1;
            runnerSettings.commandChannel = environmentService.getCommandChannel.channelName;
            runnerSettings.enableGpuCollector = this.enableGpuScheduler;
            runnerSettings.command = this.trialConfig.command;
            runnerSettings.nniManagerVersion = this.enableVersionCheck ? await getVersion() : '';
            runnerSettings.logCollection = this.logCollection;
            runnerSettings.platform = environmentService.getName;
            runnerSettings.experimentId = this.experimentId;

            await environmentService.getCommandChannel.start();
            this.log.info(`TrialDispatcher: started channel: ${environmentService.getCommandChannel.constructor.name}`);
    
            this.log.info(`TrialDispatcher: copying code and settings.`);
            let storageService: StorageService;
            if (environmentService.hasStorageService) {
                this.log.debug(`TrialDispatcher: use existing storage service.`);
                storageService = component.get<StorageService>(StorageService);
            } else {
                this.log.debug(`TrialDispatcher: create temp storage service to temp folder.`);
                storageService = new MountedStorageService();
                const environmentLocalTempFolder = path.join(this.experimentRootDir, this.experimentId, "environment-temp");
                storageService.initialize(this.trialConfig.codeDir, environmentLocalTempFolder);
            }
            // Copy the compressed file to remoteDirectory and delete it
            const codeDir = path.resolve(this.trialConfig.codeDir);
            const envDir = storageService.joinPath("envs");
            const codeFileName = await storageService.copyDirectory(codeDir, envDir, true);
            storageService.rename(codeFileName, "nni-code.tar.gz");

            const installFileName = storageService.joinPath(envDir, 'install_nni.sh');
            await storageService.save(CONTAINER_INSTALL_NNI_SHELL_FORMAT, installFileName);

            const runnerSettingsConfig = storageService.joinPath(envDir, "settings.json");
            await storageService.save(JSON.stringify(runnerSettings), runnerSettingsConfig);

            if (this.isDeveloping) {
                let trialToolsPath = path.join(__dirname, "../../../../../tools/nni_trial_tool");
                if (false === fs.existsSync(trialToolsPath)) {
                    trialToolsPath = path.join(__dirname, "..\\..\\..\\..\\..\\tools\\nni_trial_tool");
                }
                await storageService.copyDirectory(trialToolsPath, envDir, true);
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

    public get isMultiPhaseJobSupported(): boolean {
        return true;
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIp = (<NNIManagerIpConfig>JSON.parse(value)).nniManagerIp;
                break;
            case TrialConfigMetadataKey.VERSION_CHECK:
                this.enableVersionCheck = (value === 'true' || value === 'True');
                
                break;
            case TrialConfigMetadataKey.LOG_COLLECTION:
                this.logCollection = value;
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                this.trialConfig = <TrialConfig>JSON.parse(value);

                if (this.trialConfig.reuseEnvironment !== undefined) {
                    this.reuseEnvironment = this.trialConfig.reuseEnvironment;
                }
                if (this.trialConfig.gpuNum !== undefined && this.trialConfig.gpuNum > 0) {
                    this.log.info(`TrialDispatcher: GPU scheduler is enabled.`)
                    this.enableGpuScheduler = true;
                }

                // Validate to make sure codeDir doesn't have too many files
                await validateCodeDir(this.trialConfig.codeDir);
                break;
            case TrialConfigMetadataKey.PLATFORM_LIST: {
                const platforms: string[] = value.split(",");
                for(const platform of platforms) {
                    const environmentService: EnvironmentService = EnvironmentServiceFactory.createEnvironmentService(platform);
                    environmentService.initCommandChannel(this.commandEmitter);
                    this.environmentMaintenceLoopInterval =
                      Math.max(environmentService.environmentMaintenceLoopInterval, this.environmentMaintenceLoopInterval);
                    this.commandChannelSet.add(environmentService.getCommandChannel);
                    this.environmentServiceList.push(environmentService);
                }
            }
        }
        for(const environmentService of this.environmentServiceList) {
            await environmentService.config(key, value);
        }
    }

    public getClusterMetadata(_key: string): Promise<string> {
        throw new Error('Not implemented!');
    }

    public async cleanUp(): Promise<void> {
        if (this.commandEmitter === undefined) {
            throw new Error(`TrialDispatcher: commandEmitter shouldn't be undefined in cleanUp.`);
        }
        this.stopping = true;
        this.shouldUpdateTrials = true;
        const environments = [...this.environments.values()];

        for (let index = 0; index < environments.length; index++) {
            const environment = environments[index];
            this.log.info(`stopping environment ${environment.id}...`);
            if (environment.environmentService === undefined) {
                throw new Error(`${environment.id} do not have environmentService!`);
            }
            await environment.environmentService.stopEnvironment(environment);
            this.log.info(`stopped environment ${environment.id}.`);
        }
        this.commandEmitter.off("command", this.handleCommand);
        for (const commandChannel of this.commandChannelSet) {
            await commandChannel.stop();
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
                        if (
                            0 === environment.runningTrialCount &&
                            false === this.reuseEnvironment &&
                            environment.assignedTrialCount > 0
                        ) {
                            if (environment.environmentService === undefined) {
                                throw new Error(`${environment.id} does not has environment service!`);
                            }
                            await environment.environmentService.stopEnvironment(environment);
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
                    const gpuNum = this.trialConfig ? this.trialConfig.gpuNum : undefined;
                    const result = this.gpuScheduler.scheduleMachine(reusableEnvironments, gpuNum, trial);
                    switch (result.resultType) {
                        case ScheduleResultType.REQUIRE_EXCEED_TOTAL:
                            {
                                if (liveEnvironmentsCount == 0) {
                                    this.log.debug(`TrialDispatcher: no live environment, so request one.`);
                                    neededEnvironmentCount = 1;
                                    waitingTrials = [];
                                    this.isLoggedNoGpuAvailable = false;
                                } else if (reusableEnvironments.length > 0) {
                                    const errorMessage: string = `TrialDispatcher: REQUIRE_EXCEED_TOTAL Required GPU number ${gpuNum} is too large, no machine can meet`;
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

    private async requestEnvironment(environmentService: EnvironmentService): Promise<void> {
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
        if (this.trialConfig === undefined) {
            throw new Error(`TrialDispatcher: trialConfig shouldn't be undefined in allocateEnvironment.`);
        }

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
        if (undefined !== this.trialConfig.gpuNum) {
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
        if (environment.environmentService.hasStorageService) {	
            const storageService = component.get<StorageService>(StorageService);	
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
        trial.message = `Assign trial ${trial.id} to environment ${environment.id}, running platform is ${environment.name}.`;
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
        if (trial.environment !== undefined) {
            if (trial.environment.runningTrialCount <= 0) {
                throw new Error(`TrialDispatcher: environment ${trial.environment.id} has no counted running trial!`);
            }
            trial.environment.runningTrialCount--;
            trial.environment = undefined;
        }
        if (true === this.enableGpuScheduler) {
            this.gpuScheduler.removeGpuReservation(trial);
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
}

export { TrialDispatcher };
