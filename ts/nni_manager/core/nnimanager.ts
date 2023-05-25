// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert';
import { ChildProcess, StdioOptions } from 'child_process';
import { Deferred } from 'ts-deferred';
import { IocShim } from 'common/ioc_shim';
import { DataStore, MetricDataRecord, MetricType, TrialJobInfo } from '../common/datastore';
import { NNIError } from '../common/errors';
import { getExperimentId } from '../common/experimentStartupInfo';
import globals from 'common/globals';
import { Logger, getLogger } from '../common/log';
import {
    ExperimentProfile, Manager, ExperimentStatus,
    NNIManagerStatus, ProfileUpdateType, TrialJobStatistics
} from '../common/manager';
import {
    ExperimentConfig, TrainingServiceConfig, toSeconds, toCudaVisibleDevices
} from '../common/experimentConfig';
import { getExperimentsManager } from 'extensions/experiments_manager';
import { TensorboardManager } from '../common/tensorboardManager';
import {
    TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric, TrialJobStatus, TrialCommandContent, PlacementConstraint
} from '../common/trainingService';
import { delay, getCheckpointDir, getExperimentRootDir, getLogDir, getMsgDispatcherCommand, mkDirP, getTunerProc, getLogLevel, isAlive, killPid } from '../common/utils';
import {
    INITIALIZE, INITIALIZED, KILL_TRIAL_JOB, NEW_TRIAL_JOB, NO_MORE_TRIAL_JOBS, PING,
    REPORT_METRIC_DATA, REQUEST_TRIAL_JOBS, SEND_TRIAL_JOB_PARAMETER, TERMINATE, TRIAL_END, UPDATE_SEARCH_SPACE, IMPORT_DATA, ADD_CUSTOMIZED_TRIAL_JOB
} from './commands';
import { createDispatcherInterface, IpcInterface } from './ipcInterface';

/**
 * NNIManager which implements Manager interface
 */
class NNIManager implements Manager {
    private pollInterval: number; // for unittest to modify the polling interval
    private trainingService!: TrainingService;
    private dispatcher: IpcInterface | undefined;
    private currSubmittedTrialNum: number;  // need to be recovered
    private trialConcurrencyChange: number; // >0: increase, <0: decrease
    private log: Logger;
    private dataStore: DataStore;
    private experimentProfile!: ExperimentProfile;
    private dispatcherPid: number;
    private status: NNIManagerStatus;
    private waitingTrials: TrialJobApplicationForm[];
    private trialJobs: Map<string, TrialJobDetail>;
    private trialDataForTuner: string;
    private trialDataForResume: string;
    private readonly: boolean;
    private config!: ExperimentConfig;

    private trialJobMetricListener: (metric: TrialJobMetric) => void;

    constructor() {
        this.pollInterval = 5;
        this.currSubmittedTrialNum = 0;
        this.trialConcurrencyChange = 0;
        this.dispatcherPid = 0;
        this.waitingTrials = [];
        this.trialJobs = new Map<string, TrialJobDetail>();
        this.trialDataForTuner = '';
        this.trialDataForResume = '';
        this.readonly = false;

        this.log = getLogger('NNIManager');
        this.dataStore = IocShim.get(DataStore);
        this.status = {
            status: 'INITIALIZED',
            errors: []
        };
        this.trialJobMetricListener = (metric: TrialJobMetric): void => {
            this.onTrialJobMetrics(metric).catch((err: Error) => {
                this.criticalError(NNIError.FromError(err, 'Job metrics error: '));
            });
        };

        globals.shutdown.register('NniManager', this.stopExperiment.bind(this));
    }

    public updateExperimentProfile(experimentProfile: ExperimentProfile, updateType: ProfileUpdateType): Promise<void> {
        if (this.readonly) {
            return Promise.reject(new Error('Error: can not update experiment profile in readonly mode!'));
        }
        switch (updateType) {
            case 'TRIAL_CONCURRENCY':
                this.updateTrialConcurrency(experimentProfile.params.trialConcurrency);
                break;
            case 'MAX_EXEC_DURATION':
                this.experimentProfile.params.maxExperimentDuration = experimentProfile.params.maxExperimentDuration;
                break;
            case 'SEARCH_SPACE':
                this.updateSearchSpace(experimentProfile.params.searchSpace);
                break;
            case 'MAX_TRIAL_NUM':
                this.experimentProfile.params.maxTrialNumber = experimentProfile.params.maxTrialNumber;
                break;
            default:
                throw new Error('Error: unrecognized updateType');
        }

        return this.storeExperimentProfile();
    }

    public importData(data: string): Promise<void> {
        if (this.readonly) {
            return Promise.reject(new Error('Error: can not import data in readonly mode!'));
        }
        if (this.dispatcher === undefined) {
            return Promise.reject(
                new Error('tuner has not been setup')
            );
        }
        this.dispatcher.sendCommand(IMPORT_DATA, data);

        return this.dataStore.storeTrialJobEvent('IMPORT_DATA', '', data);
    }

    public getImportedData(): Promise<string[]> {
        return this.dataStore.getImportedData();
    }

    public async exportData(): Promise<string> {
        return this.dataStore.exportTrialHpConfigs();
    }

    public addRecoveredTrialJob(allTrialJobs: Array<TrialJobInfo>): number {
        const jobs: Array<TrialJobInfo> = allTrialJobs.filter((job: TrialJobInfo) => job.status === 'WAITING' || job.status === 'RUNNING');
        const trialData: any[] = [];
        let maxSequeceId = 0;
        for (const job of jobs) {
            if (job.sequenceId === undefined || job.hyperParameters === undefined) {
                this.log.warning('The trial to be recovered missing sequenceId and/or hyperParameters', job);
                continue;
            }
            const params: string = job.hyperParameters[0];
            const sequenceId: number = job.sequenceId;
            maxSequeceId = Math.max(maxSequeceId, sequenceId);
            
            const hyperParams = JSON.parse(params);
            const packedParameter = {
                parameter_id: hyperParams['parameter_id'],
                parameter_source: 'resumed',
                parameters: hyperParams['parameters'],
                parameter_index: hyperParams['parameter_index'],
            }
            const form: TrialJobApplicationForm = {
                id: job.trialJobId,
                sequenceId: sequenceId,
                hyperParameters: {
                    value: JSON.stringify(packedParameter),
                    index: 0
                },
                envId: job.envId,
            };

            this.waitingTrials.push(form);
            trialData.push(packedParameter);
            this.dataStore.storeTrialJobEvent('ADD_RESUMED', job.trialJobId, '');
        }
        this.trialDataForResume = JSON.stringify(trialData);

        // next sequenceId
        this.experimentProfile.nextSequenceId = maxSequeceId + 1;
        return trialData.length;
    }

    public addCustomizedTrialJob(hyperParams: string): Promise<number> {
        if (this.readonly) {
            return Promise.reject(new Error('Error: can not add customized trial job in readonly mode!'));
        }
        if (this.currSubmittedTrialNum >= this.maxTrialNum) {
            return Promise.reject(new Error('reach maxTrialNum'));
        }

        // TODO: NNI manager should not peek tuner's internal protocol, let's refactor this later
        const packedParameter = {
            parameter_id: null,
            parameter_source: 'customized',
            parameters: JSON.parse(hyperParams)
        }

        const form: TrialJobApplicationForm = {
            sequenceId: this.experimentProfile.nextSequenceId++,
            hyperParameters: {
                value: JSON.stringify(packedParameter),
                index: 0
            }
        };
        this.waitingTrials.push(form);

        // trial id has not been generated yet, thus use '' instead
        this.dataStore.storeTrialJobEvent('ADD_CUSTOMIZED', '', hyperParams);

        return Promise.resolve(form.sequenceId);
    }

    public async cancelTrialJobByUser(trialJobId: string): Promise<void> {
        if (this.readonly) {
            return Promise.reject(new Error('Error: can not cancel trial job in readonly mode!'));
        }
        this.log.info(`User cancelTrialJob: ${trialJobId}`);
        await this.trainingService.cancelTrialJob(trialJobId);
        await this.dataStore.storeTrialJobEvent('USER_TO_CANCEL', trialJobId, '');
    }

    public async startExperiment(config: ExperimentConfig): Promise<string> {
        this.experimentProfile = {
            params: config,
            id: getExperimentId(),
            execDuration: 0,
            logDir: getExperimentRootDir(),
            startTime: Date.now(),
            endTime: undefined,
            nextSequenceId: 0,
            revision: 0
        };
        this.config = config;
        this.log.info(`Starting experiment: ${this.experimentProfile.id}`);
        await this.storeExperimentProfile();

        if (this.trainingService === undefined) {
            this.log.info('Setup training service...');
            this.trainingService = await this.initTrainingService(config);
        }

        this.log.info('Setup tuner...');
        const dispatcherCommand: string[] = getMsgDispatcherCommand(config);
        this.log.debug(`dispatcher command: ${dispatcherCommand}`);
        const checkpointDir: string = await this.createCheckpointDir();
        await this.setupTuner(dispatcherCommand, undefined, 'start', checkpointDir);
        this.setStatus('RUNNING');
        await this.storeExperimentProfile();
        this.run().catch((err: Error) => {
            this.criticalError(err);
        });

        return this.experimentProfile.id;
    }

    public async resumeExperiment(readonly: boolean): Promise<void> {
        //Fetch back the experiment profile
        const experimentId: string = getExperimentId();
        this.log.info(`Resuming experiment: ${experimentId}`);
        this.experimentProfile = await this.dataStore.getExperimentProfile(experimentId);

        const config: ExperimentConfig = this.experimentProfile.params;
        this.config = config;
        if (this.trainingService === undefined) {
            this.log.info('Setup training service...');
            this.trainingService = await this.initTrainingService(config);
        }

        this.readonly = readonly;
        if (readonly) {
            this.setStatus('VIEWED');
            return;
        }

        this.log.info('Setup tuner...');
        const dispatcherCommand: string[] = getMsgDispatcherCommand(config);
        this.log.debug(`dispatcher command: ${dispatcherCommand}`);
        const checkpointDir: string = await this.createCheckpointDir();
        await this.setupTuner(dispatcherCommand, undefined, 'resume', checkpointDir);

        const allTrialJobs: TrialJobInfo[] = await this.dataStore.listTrialJobs();

        // Resume currSubmittedTrialNum
        this.currSubmittedTrialNum = allTrialJobs.length;
        const recoveredTrialNum = this.addRecoveredTrialJob(allTrialJobs);
        // minus the number of the recovered trials,
        // the recovered trials should not be counted in maxTrialNumber.
        this.log.info(`Number of current submitted trials: ${this.currSubmittedTrialNum}, where ${recoveredTrialNum} is resuming.`);
        this.currSubmittedTrialNum -= recoveredTrialNum;

        // Collect generated trials and imported trials
        const finishedTrialData: string = await this.exportData();
        const importedData: string[] = await this.dataStore.getImportedData();
        let trialData: Record<string, any>[] = JSON.parse(finishedTrialData);
        for (const oneImportedData of importedData) {
            // do not deduplicate
            trialData = trialData.concat(<Record<string, any>[]>JSON.parse(oneImportedData));
        }
        this.trialDataForTuner = JSON.stringify(trialData);

        if (this.experimentProfile.execDuration < this.maxDuration &&
            this.currSubmittedTrialNum < this.maxTrialNum &&
            this.experimentProfile.endTime) {
            delete this.experimentProfile.endTime;
        }
        this.setStatus('RUNNING');

        // TO DO: update database record for resume event
        this.run().catch((err: Error) => {
            this.criticalError(err);
        });
    }

    public getTrialJob(trialJobId: string): Promise<TrialJobInfo> {
        return this.dataStore.getTrialJob(trialJobId);
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        // Hack for supporting v2 config, need refactor
        if (this.trainingService === undefined) {
            this.log.info('Setup training service...');
            switch (key) {
                case 'kubeflow_config': {
                    const kubeflowModule = await import('../training_service/kubernetes/kubeflow/kubeflowTrainingService');
                    this.trainingService = new kubeflowModule.KubeflowTrainingService();
                    break;
                }
                case 'frameworkcontroller_config': {
                    const fcModule = await import('../training_service/kubernetes/frameworkcontroller/frameworkcontrollerTrainingService');
                    this.trainingService = new fcModule.FrameworkControllerTrainingService();
                    break;
                }
                default:
                    throw new Error("Setup training service failed.");
            }
        }
        await this.trainingService.setClusterMetadata(key, value);
    }

    public getClusterMetadata(key: string): Promise<string> {
        return this.trainingService.getClusterMetadata(key);
    }

    public async getTrialJobStatistics(): Promise<TrialJobStatistics[]> {
        return this.dataStore.getTrialJobStatistics();
    }

    private async stopExperiment(): Promise<void> {
        await this.stopExperimentTopHalf();
        await this.stopExperimentBottomHalf();
    }

    private async stopExperimentTopHalf(): Promise<void> {
        this.setStatus('STOPPING');
        this.log.info('Stopping experiment, cleaning up ...');

        if (this.dispatcher === undefined) {
            this.log.error('Tuner has not been setup');
            return;
        }

        this.trainingService.removeTrialJobMetricListener(this.trialJobMetricListener);
        // NOTE: this sending TERMINATE should be out of the if clause,
        // because when python dispatcher is started before nnimanager
        // this.dispatcherPid would not have a valid value (i.e., not >0).
        this.dispatcher.sendCommand(TERMINATE);
        if (this.dispatcherPid > 0) {
            // gracefully terminate tuner and assessor here, wait at most 30 seconds.
            for (let i: number = 0; i < 30; i++) {
                if (!await isAlive(this.dispatcherPid)) {
                    break;
                }
                await delay(1000);
            }
            await killPid(this.dispatcherPid);
        }
        this.dispatcher = undefined;
    }

    private async stopExperimentBottomHalf(): Promise<void> {
        try {
            const trialJobList: TrialJobDetail[] = await this.trainingService.listTrialJobs();

            // DON'T try to make it in parallel, the training service may not handle it well.
            // If there is performance concern, consider to support batch cancellation on training service.
            for (const trialJob of trialJobList) {
                if (trialJob.status === 'RUNNING' ||
                    trialJob.status === 'WAITING') {
                    try {
                        this.log.info(`cancelTrialJob: ${trialJob.id}`);
                        await this.trainingService.cancelTrialJob(trialJob.id);
                    } catch (error) {
                        this.log.debug(`ignorable error on canceling trial ${trialJob.id}. ${error}`);
                    }
                }
            }
            await this.trainingService.cleanUp();
        } catch (err) {
            this.log.error(`${(err as any).stack}`);
        }
        if (this.experimentProfile.endTime === undefined) {
            this.setEndtime();
        }
        await this.storeExperimentProfile();
        this.setStatus('STOPPED');
        this.log.info('Experiment stopped.');

        await IocShim.get<TensorboardManager>(TensorboardManager).stop();
        await this.dataStore.close();
    }

    public async getMetricData(trialJobId?: string, metricType?: MetricType): Promise<MetricDataRecord[]> {
        return this.dataStore.getMetricData(trialJobId, metricType);
    }

    public async getMetricDataByRange(minSeqId: number, maxSeqId: number): Promise<MetricDataRecord[]> {
        const trialJobs = await this.dataStore.listTrialJobs();
        const targetTrials = trialJobs.filter(trial => (
            // FIXME: can this be undefined?
            trial.sequenceId !== undefined && minSeqId <= trial.sequenceId && trial.sequenceId <= maxSeqId
        ));
        const targetTrialIds = new Set(targetTrials.map(trial => trial.trialJobId));

        const allMetrics = await this.dataStore.getMetricData();
        return allMetrics.filter(metric => targetTrialIds.has(metric.trialJobId));
    }

    public async getLatestMetricData(): Promise<MetricDataRecord[]> {
        // FIXME: this can take a long time
        const allMetrics: MetricDataRecord[] = await this.dataStore.getMetricData();
        const finals: MetricDataRecord[] = [];
        const latestIntermediates: Map<string, MetricDataRecord> = new Map<string, MetricDataRecord>();
        for (const metric of allMetrics) {
            if (metric.type !== 'PERIODICAL') {
                finals.push(metric);
            } else {
                const old: MetricDataRecord | undefined = latestIntermediates.get(metric.trialJobId);
                if (old === undefined || old.sequence <= metric.sequence) {
                    latestIntermediates.set(metric.trialJobId, metric);
                }
            }
        }
        return finals.concat(Array.from(latestIntermediates.values()));
        // FIXME: unit test
    }

    public async getTrialFile(trialJobId: string, fileName: string): Promise<Buffer | string> {
        return this.trainingService.getTrialFile(trialJobId, fileName);
    }

    public getExperimentProfile(): Promise<ExperimentProfile> {
        // TO DO: using Promise.resolve()
        const deferred: Deferred<ExperimentProfile> = new Deferred<ExperimentProfile>();
        deferred.resolve(this.experimentProfile);

        return deferred.promise;
    }

    public getStatus(): NNIManagerStatus {
        return this.status;
    }

    public async listTrialJobs(status?: TrialJobStatus): Promise<TrialJobInfo[]> {
        return this.dataStore.listTrialJobs(status);
    }

    private get maxDuration(): number {
        const value = this.experimentProfile.params.maxExperimentDuration;
        return (value === undefined ? Infinity : toSeconds(value));
    }

    private get maxTrialNum(): number {
        const value = this.experimentProfile.params.maxTrialNumber;
        return (value === undefined ? Infinity : value);
    }

    private get maxTrialDuration(): number {
        const value = this.experimentProfile.params.maxTrialDuration;
        return (value === undefined ? Infinity : toSeconds(value));
    }

    private async initTrainingService(config: ExperimentConfig): Promise<TrainingService> {
        let platform: string;
        if (Array.isArray(config.trainingService)) {
            platform = 'hybrid';
        } else if (config.trainingService.platform) {
            platform = config.trainingService.platform;
        } else {
            platform = (config as any).trainingServicePlatform;
        }
        if (!platform) {
            throw new Error('Cannot detect training service platform');
        }
        const reuseMode = Array.isArray(config.trainingService) || (config.trainingService as any).reuseMode;

        if (reuseMode) {
            const module_ = await import('../training_service/reusable/routerTrainingService');
            return await module_.RouterTrainingService.construct(config);
        } else if (platform === 'kubeflow') {
            const module_ = await import('../training_service/kubernetes/kubeflow/kubeflowTrainingService');
            return new module_.KubeflowTrainingService();
        } else if (platform === 'frameworkcontroller') {
            const module_ = await import('../training_service/kubernetes/frameworkcontroller/frameworkcontrollerTrainingService');
            return new module_.FrameworkControllerTrainingService();
        } else {
            this.pollInterval = 0.5;
            const module_ = await import('../training_service/v3/compat');
            return new module_.V3asV1(config.trainingService as TrainingServiceConfig);
        }
    }

    private async setupTuner(command: string[], cwd: string | undefined, mode: 'start' | 'resume', dataDirectory: string): Promise<void> {
        if (this.dispatcher !== undefined) {
            return;
        }

        let tunerWs: string;
        if (globals.args.urlPrefix) {
            tunerWs = `ws://localhost:${globals.args.port}/${globals.args.urlPrefix}/tuner`;
        } else {
            tunerWs = `ws://localhost:${globals.args.port}/tuner`;
        }

        if (globals.args.tunerCommandChannel) {
            // TODO: this will become configurable after refactoring rest handler interface
            assert.equal(tunerWs, globals.args.tunerCommandChannel);
            this.dispatcher = await createDispatcherInterface();
            return;
        }

        const stdio: StdioOptions = ['ignore', process.stdout, process.stderr];
        let newCwd: string;
        if (cwd === undefined || cwd === '') {
            newCwd = getLogDir();
        } else {
            newCwd = cwd;
        }
        // TO DO: add CUDA_VISIBLE_DEVICES
        const includeIntermediateResultsEnv = !!(this.config.deprecated && this.config.deprecated.includeIntermediateResults);

        const nniEnv = {
            SDK_PROCESS: 'dispatcher',
            NNI_MODE: mode,
            NNI_CHECKPOINT_DIRECTORY: dataDirectory,
            NNI_LOG_DIRECTORY: getLogDir(),
            NNI_LOG_LEVEL: getLogLevel(),
            NNI_INCLUDE_INTERMEDIATE_RESULTS: includeIntermediateResultsEnv,
            NNI_TUNER_COMMAND_CHANNEL: tunerWs,
            CUDA_VISIBLE_DEVICES: toCudaVisibleDevices(this.experimentProfile.params.tunerGpuIndices)
        };
        const newEnv = Object.assign({}, process.env, nniEnv);
        const tunerProc: ChildProcess = getTunerProc(command, stdio, newCwd, newEnv);
        this.dispatcherPid = tunerProc.pid!;
        this.dispatcher = await createDispatcherInterface();

        return;
    }

    private updateTrialConcurrency(trialConcurrency: number): void {
        // we assume trialConcurrency >= 0, which is checked by restserver
        this.trialConcurrencyChange += (trialConcurrency - this.experimentProfile.params.trialConcurrency);
        this.experimentProfile.params.trialConcurrency = trialConcurrency;

        return;
    }

    private updateSearchSpace(searchSpace: object): void {
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner has not been setup');
        }
        this.log.info(`Updated search space ${searchSpace}`);
        this.dispatcher.sendCommand(UPDATE_SEARCH_SPACE, JSON.stringify(searchSpace));
        this.experimentProfile.params.searchSpace = searchSpace;

        return;
    }

    private async periodicallyUpdateExecDuration(): Promise<void> {
        let count: number = 1;
        while (!['ERROR', 'STOPPING', 'STOPPED'].includes(this.status.status)) {
            await delay(1000 * 1); // 1 seconds
            if (['RUNNING', 'NO_MORE_TRIAL', 'TUNER_NO_MORE_TRIAL'].includes(this.status.status)) {
                this.experimentProfile.execDuration += 1;
                if (count % 10 === 0) {
                    await this.storeExperimentProfile();
                }
            }
            count += 1;
        }
    }

    private async pingDispatcher(): Promise<void> {
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner has not been setup');
        }
        while (!['ERROR', 'STOPPING', 'STOPPED'].includes(this.status.status)) {
            this.dispatcher.sendCommand(PING);
            await delay(1000 * this.pollInterval); // 5 seconds
        }
    }

    private async stopTrialIfOverMaxDurationLimit(): Promise<void> {
        if(this.maxTrialDuration === Infinity){
            return;
        }

        for (const trialJobId of Array.from(this.trialJobs.keys())) {
            const trialJobDetail: TrialJobDetail | undefined = this.trialJobs.get(trialJobId);
            if(undefined !== trialJobDetail &&
                trialJobDetail.status === 'RUNNING' &&
                trialJobDetail.startTime !== undefined){
                const currentTrialDuration = (new Date().getTime() - trialJobDetail.startTime) / 1000;
                if(currentTrialDuration>this.maxTrialDuration) {
                    const isEarlyStopped = true;
                    await this.trainingService.cancelTrialJob(trialJobId, isEarlyStopped);
                    this.log.info(`Trial job ${trialJobDetail.id} has been canceled because it is over max trial duration.`);
                }
            }
        }
    }

    private async requestTrialJobsStatus(): Promise<number> {
        let finishedTrialJobNum: number = 0;
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner has not been setup');
        }
        for (const trialJobId of Array.from(this.trialJobs.keys())) {
            const trialJobDetail: TrialJobDetail = await this.trainingService.getTrialJob(trialJobId);
            const oldTrialJobDetail: TrialJobDetail | undefined = this.trialJobs.get(trialJobId);
            if (oldTrialJobDetail !== undefined && oldTrialJobDetail.status !== trialJobDetail.status) {
                this.log.info(`Trial job ${trialJobDetail.id} status changed from ${oldTrialJobDetail.status} to ${trialJobDetail.status}`);
                this.trialJobs.set(trialJobId, Object.assign({}, trialJobDetail));
                await this.dataStore.storeTrialJobEvent(trialJobDetail.status, trialJobDetail.id, undefined, trialJobDetail);
            }
            const newTrialJobDetail: TrialJobDetail | undefined = this.trialJobs.get(trialJobId);
            if (newTrialJobDetail !== undefined) {
                newTrialJobDetail.message = trialJobDetail.message;
            }
            let hyperParams: string | undefined = undefined;
            switch (trialJobDetail.status) {
                case 'SUCCEEDED':
                case 'USER_CANCELED':
                case 'EARLY_STOPPED':
                    this.trialJobs.delete(trialJobId);
                    finishedTrialJobNum++;
                    hyperParams = trialJobDetail.form.hyperParameters.value;
                    this.dispatcher.sendCommand(TRIAL_END, JSON.stringify({
                        trial_job_id: trialJobDetail.id,
                        event: trialJobDetail.status,
                        hyper_params: hyperParams
                    }));
                    break;
                case 'FAILED':
                case 'SYS_CANCELED':
                    // In the current version, we do not retry
                    // TO DO: push this job to queue for retry
                    this.trialJobs.delete(trialJobId);
                    finishedTrialJobNum++;
                    hyperParams = trialJobDetail.form.hyperParameters.value;
                    this.dispatcher.sendCommand(TRIAL_END, JSON.stringify({
                        trial_job_id: trialJobDetail.id,
                        event: trialJobDetail.status,
                        hyper_params: hyperParams
                    }));
                    break;
                case 'WAITING':
                case 'RUNNING':
                case 'UNKNOWN':
                    // Do nothing
                    break;
                default:
                // TO DO: add warning in log
            }
        }

        return finishedTrialJobNum;
    }

    private async manageTrials(): Promise<void> {
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner has not been setup');
        }
        let allFinishedTrialJobNum: number = this.currSubmittedTrialNum;
        let waitSubmittedToFinish: number;
        while (!['ERROR', 'STOPPING', 'STOPPED'].includes(this.status.status)) {
            await this.stopTrialIfOverMaxDurationLimit();

            const finishedTrialJobNum: number = await this.requestTrialJobsStatus();
            allFinishedTrialJobNum += finishedTrialJobNum;

            // requestTrialNum is the number of trials that will be requested from tuner.
            // If trialConcurrency does not change, requestTrialNum equals finishedTrialJobNum.
            // If trialConcurrency changes, for example, trialConcurrency increases by 2 (trialConcurrencyChange=2), then
            // requestTrialNum equals 2 + finishedTrialJobNum and trialConcurrencyChange becomes 0.
            // If trialConcurrency changes, for example, trialConcurrency decreases by 4 (trialConcurrencyChange=-4) and
            // finishedTrialJobNum is 2, then requestTrialNum becomes -2. No trial will be requested from tuner,
            // and trialConcurrencyChange becomes -2.
            const requestTrialNum: number = this.trialConcurrencyChange + finishedTrialJobNum;
            if (requestTrialNum >= 0) {
                this.trialConcurrencyChange = 0;
            } else {
                this.trialConcurrencyChange = requestTrialNum;
            }

            // check maxtrialnum and maxduration here
            // NO_MORE_TRIAL is more like a subset of RUNNING, because during RUNNING tuner
            // might tell nnimanager that this is no more trials. In NO_MORE_TRIAL state, the experiment is viewed
            // as still running. DONE could be transfered from RUNNING or NO_MORE_TRIAL.
            assert(this.status.status === 'RUNNING' ||
                this.status.status === 'DONE' ||
                this.status.status === 'NO_MORE_TRIAL' ||
                this.status.status === 'TUNER_NO_MORE_TRIAL', `Actual status: ${this.status.status}`);
            if (this.experimentProfile.execDuration > this.maxDuration ||
                this.currSubmittedTrialNum >= this.maxTrialNum) {
                if (this.status.status !== 'DONE') {
                    this.setStatus('NO_MORE_TRIAL');
                    waitSubmittedToFinish = this.currSubmittedTrialNum;

                    assert(allFinishedTrialJobNum <= waitSubmittedToFinish);
                    if (allFinishedTrialJobNum >= waitSubmittedToFinish) {
                        this.setStatus('DONE');
                        this.setEndtime();
                        await this.storeExperimentProfile();
                        // write this log for travis CI
                        this.log.info('Experiment done.');
                    }
                }
            } else {
                this.requestTrialJobs(requestTrialNum);

                if (this.status.status === 'DONE') {
                    delete this.experimentProfile.endTime;
                    await this.storeExperimentProfile();
                }
                if (this.status.status !== 'TUNER_NO_MORE_TRIAL') {
                    this.setStatus('RUNNING');
                }
                for (let i: number = this.trialJobs.size; i < this.experimentProfile.params.trialConcurrency; i++) {
                    if (this.waitingTrials.length === 0 ||
                        this.currSubmittedTrialNum >= this.maxTrialNum) {
                        break;
                    }
                    const form = this.waitingTrials.shift() as TrialJobApplicationForm;
                    this.currSubmittedTrialNum++;
                    this.log.info('submitTrialJob: form:', form);
                    const trialJobDetail: TrialJobDetail = await this.trainingService.submitTrialJob(form);
                    const Snapshot: TrialJobDetail = Object.assign({}, trialJobDetail);
                    await this.storeExperimentProfile();
                    this.trialJobs.set(trialJobDetail.id, Snapshot);
                    const trialJobDetailSnapshot: TrialJobDetail | undefined = this.trialJobs.get(trialJobDetail.id);
                    if (trialJobDetailSnapshot != undefined) {
                        await this.dataStore.storeTrialJobEvent(
                            trialJobDetailSnapshot.status, trialJobDetailSnapshot.id, form.hyperParameters.value, trialJobDetailSnapshot);
                    } else {
                        assert(false, `undefined trialJobDetail in trialJobs: ${trialJobDetail.id}`);
                    }
                }
            }
            await delay(1000 * this.pollInterval); // 5 seconds
        }
    }

    private storeExperimentProfile(): Promise<void> {
        this.experimentProfile.revision += 1;

        return this.dataStore.storeExperimentProfile(this.experimentProfile);
    }

    private async run(): Promise<void> {
        assert(this.dispatcher !== undefined);
        await this.dispatcher.init();

        this.addEventListeners();

        this.sendInitTunerCommands();

        await Promise.all([
            this.periodicallyUpdateExecDuration(),
            this.pingDispatcher().catch((err: Error) => {
                throw NNIError.FromError(err, 'Dispatcher error: ');
            }),
            this.trainingService.run().catch((err: Error) => {
                // FIXME: The error handling here could crash when err is undefined.
                throw NNIError.FromError(err, 'Training service error: ');
            }),
            this.manageTrials().catch((err: Error) => {
                throw NNIError.FromError(err, 'Job management error: ');
            })]);
    }

    private addEventListeners(): void {
        this.log.info('Add event listeners');
        // TO DO: cannot run this method more than once in one NNIManager instance
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner or job maintainer have not been setup');
        }
        this.trainingService.addTrialJobMetricListener(this.trialJobMetricListener);

        this.dispatcher.onCommand((commandType: string, content: string) => {
            this.onTunerCommand(commandType, content).catch((err: Error) => {
                this.criticalError(NNIError.FromError(err, 'Tuner command event error: '));
            });
        });
        this.dispatcher.onError((error: Error) => {
            this.log.error(`Dispatcher error: ${error.message}`);
            this.criticalError(new Error('Dispatcher stream error, tuner may have crashed.'));
        });
    }

    private sendInitTunerCommands(): void {
        if (this.dispatcher === undefined) {
            throw new Error('Dispatcher error: tuner has not been setup');
        }
        this.log.debug(`Send tuner command: INITIALIZE: ${this.experimentProfile.params.searchSpace}`);
        // Tuner need to be initialized with search space before generating any hyper parameters
        this.dispatcher.sendCommand(INITIALIZE, JSON.stringify(this.experimentProfile.params.searchSpace));
    }

    private async onTrialJobMetrics(metric: TrialJobMetric): Promise<void> {
        this.log.debug('NNIManager received trial job metrics:', metric);
        if (this.trialJobs.has(metric.id)) {
            await this.dataStore.storeMetricData(metric.id, metric.data);
            if (this.dispatcher === undefined) {
                throw new Error('Error: tuner has not been setup');
            }
            this.dispatcher.sendCommand(REPORT_METRIC_DATA, metric.data);
        } else {
            this.log.warning('NNIManager received non-existent trial job metrics:', metric);
        }
    }

    private requestTrialJobs(jobNum: number): void {
        if (jobNum < 1) {
            return;
        }
        if (this.dispatcher === undefined) {
            throw new Error('Dispatcher error: tuner has not been setup');
        }
        if (this.config.deprecated && this.config.deprecated.multiThread) {
            // Send multiple requests to ensure multiple hyper parameters are generated in non-blocking way.
            // For a single REQUEST_TRIAL_JOBS request, hyper parameters are generated one by one
            // sequentially.
            for (let i: number = 0; i < jobNum; i++) {
                this.dispatcher.sendCommand(REQUEST_TRIAL_JOBS, '1');
            }
        } else {
            this.dispatcher.sendCommand(REQUEST_TRIAL_JOBS, String(jobNum));
        }
    }

    private async onTunerCommand(commandType: string, content: string): Promise<void> {
        this.log.info(`NNIManager received command from dispatcher: ${commandType}, ${content}`);
        switch (commandType) {
            case INITIALIZED: {
                // Tuner is intialized, search space is set, request tuner to generate hyper parameters
                if (this.trialDataForTuner.length > 0) {
                    if (this.dispatcher === undefined) {
                        throw new Error('Dispatcher error: tuner has not been setup');
                    }
                    this.dispatcher.sendCommand(IMPORT_DATA, this.trialDataForTuner);
                }
                if (this.trialDataForResume.length > 0 ) {
                    if (this.dispatcher === undefined) {
                        throw new Error('Dispatcher error: tuner has not been setup');
                    }
                    this.dispatcher.sendCommand(ADD_CUSTOMIZED_TRIAL_JOB, this.trialDataForResume);
                }
                this.requestTrialJobs(this.experimentProfile.params.trialConcurrency);
                break;
            }
            case NEW_TRIAL_JOB: {
                if (this.status.status === 'TUNER_NO_MORE_TRIAL') {
                    this.log.warning('It is not supposed to receive more trials after NO_MORE_TRIAL is set');
                    this.setStatus('RUNNING');
                }
                const trialRequestContent: TrialCommandContent = JSON.parse(content);
                const noneConstraint: PlacementConstraint = {type: 'None', gpus: []};
                const form: TrialJobApplicationForm = {
                    sequenceId: this.experimentProfile.nextSequenceId++,
                    hyperParameters: {
                        value: content,
                        index: 0
                    },
                    placementConstraint: trialRequestContent.placement_constraint? trialRequestContent.placement_constraint : noneConstraint
                };
                this.waitingTrials.push(form);
                break;
            }
            case SEND_TRIAL_JOB_PARAMETER: {
                const tunerCommand: any = JSON.parse(content);
                assert(tunerCommand.parameter_index >= 0);
                assert(tunerCommand.trial_job_id !== undefined);

                const trialJobForm: TrialJobApplicationForm = {
                    sequenceId: -1,  // FIXME: multi-phase tuner should use sequence ID instead of trial job ID
                    hyperParameters: {
                        value: content,
                        index: tunerCommand.parameter_index
                    }
                };
                this.log.info('updateTrialJob: job id:', tunerCommand.trial_job_id, 'form:', trialJobForm);
                await this.trainingService.updateTrialJob(tunerCommand.trial_job_id, trialJobForm);
                if (tunerCommand['parameters'] !== null) {
                    // parameters field is set as empty string if no more hyper parameter can be generated by tuner.
                    await this.dataStore.storeTrialJobEvent(
                        'ADD_HYPERPARAMETER', tunerCommand.trial_job_id, content, undefined);
                }
                break;
            }
            case NO_MORE_TRIAL_JOBS: {
                if (!['ERROR', 'STOPPING', 'STOPPED'].includes(this.status.status)) {
                    this.setStatus('TUNER_NO_MORE_TRIAL');
                }
                break;
            }
            case KILL_TRIAL_JOB: {
                this.log.info('cancelTrialJob:', content);
                await this.trainingService.cancelTrialJob(JSON.parse(content), true);
                break;
            }
            default:
                throw new Error('Error: unsupported command type from tuner');
        }
    }

    private criticalError(err: Error): void {
        this.logError(err);
        console.error(err);
    }

    private logError(err: Error): void {
        if (err !== undefined) {
            // FIXME: I don't know why, but in some cases err could be undefined.
            if (err.stack !== undefined) {
                this.log.error(err.stack);
                this.status.errors.push(err.message);
            } else {
                this.status.errors.push(`Undefined error, stack: ${new Error().stack}`);
            }
        }
        this.setEndtime();
        this.setStatus('ERROR');
    }

    private setStatus(status: ExperimentStatus): void {
        if (status !== this.status.status) {
            this.log.info(`Change NNIManager status from: ${this.status.status} to: ${status}`);
            this.status.status = status;
            getExperimentsManager().setExperimentInfo(this.experimentProfile.id, 'status', this.status.status);
        }
    }

    private setEndtime(): void {
        this.experimentProfile.endTime = Date.now();
        getExperimentsManager().setExperimentInfo(this.experimentProfile.id, 'endTime', this.experimentProfile.endTime);
    }

    private async createCheckpointDir(): Promise<string> {
        // TODO: test
        const chkpDir: string = getCheckpointDir();
        await mkDirP(chkpDir);
        return chkpDir;
    }

    public async getTrialOutputLocalPath(trialJobId: string): Promise<string> {
        return this.trainingService.getTrialOutputLocalPath(trialJobId);
    }

    public async fetchTrialOutput(trialJobId: string, subpath: string): Promise<void> {
        return this.trainingService.fetchTrialOutput(trialJobId, subpath);
    }
}

export { NNIManager };
