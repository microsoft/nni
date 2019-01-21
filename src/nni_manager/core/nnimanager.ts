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

import * as assert from 'assert';
import * as cpp from 'child-process-promise';
import { ChildProcess, spawn, StdioOptions } from 'child_process';
import { Deferred } from 'ts-deferred';
import * as component from '../common/component';
import { DataStore, MetricDataRecord, MetricType, TrialJobInfo } from '../common/datastore';
import { NNIError } from '../common/errors';
import { getExperimentId, setInitTrialSequenceId } from '../common/experimentStartupInfo';
import { getLogger, Logger } from '../common/log';
import {
    ExperimentParams, ExperimentProfile, Manager,
    NNIManagerStatus, ProfileUpdateType, TrialJobStatistics
} from '../common/manager';
import {
    TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric, TrialJobStatus
} from '../common/trainingService';
import { delay, getLogDir, getCheckpointDir, getMsgDispatcherCommand, mkDirP } from '../common/utils';
import {
    ADD_CUSTOMIZED_TRIAL_JOB, INITIALIZE, INITIALIZED, KILL_TRIAL_JOB, NEW_TRIAL_JOB, NO_MORE_TRIAL_JOBS,
    REPORT_METRIC_DATA, REQUEST_TRIAL_JOBS, SEND_TRIAL_JOB_PARAMETER, TERMINATE, TRIAL_END, UPDATE_SEARCH_SPACE
} from './commands';
import { createDispatcherInterface, IpcInterface } from './ipcInterface';

/**
 * NNIManager
 */
class NNIManager implements Manager {
    private trainingService: TrainingService;
    private dispatcher: IpcInterface | undefined;
    private currSubmittedTrialNum: number;  // need to be recovered
    private trialConcurrencyChange: number; // >0: increase, <0: decrease
    private customizedTrials: string[]; // need to be recovered
    private log: Logger;
    private dataStore: DataStore;
    private experimentProfile: ExperimentProfile;
    private dispatcherPid: number;
    private status: NNIManagerStatus;
    private waitingTrials: string[];
    private trialJobs: Map<string, TrialJobDetail>;

    constructor() {
        this.currSubmittedTrialNum = 0;
        this.trialConcurrencyChange = 0;
        this.customizedTrials = [];
        this.trainingService = component.get(TrainingService);
        assert(this.trainingService);
        this.dispatcherPid = 0;
        this.waitingTrials = [];
        this.trialJobs = new Map<string, TrialJobDetail>();

        this.log = getLogger();
        this.dataStore = component.get(DataStore);
        this.experimentProfile = this.createEmptyExperimentProfile();
        this.status = {
            status: 'INITIALIZED',
            errors: []
        };
    }

    public updateExperimentProfile(experimentProfile: ExperimentProfile, updateType: ProfileUpdateType): Promise<void> {
        switch (updateType) {
            case 'TRIAL_CONCURRENCY':
                this.updateTrialConcurrency(experimentProfile.params.trialConcurrency);
                break;
            case 'MAX_EXEC_DURATION':
                this.updateMaxExecDuration(experimentProfile.params.maxExecDuration);
                break;
            case 'SEARCH_SPACE':
                this.updateSearchSpace(experimentProfile.params.searchSpace);
                break;
            case 'MAX_TRIAL_NUM':
                this.updateMaxTrialNum(experimentProfile.params.maxTrialNum);
                break;
            default:
                throw new Error('Error: unrecognized updateType');
        }

        return this.storeExperimentProfile();
    }

    public addCustomizedTrialJob(hyperParams: string): Promise<void> {
        if (this.currSubmittedTrialNum >= this.experimentProfile.params.maxTrialNum) {
            return Promise.reject(
                new Error('reach maxTrialNum')
            );
        }
        this.customizedTrials.push(hyperParams);

        // trial id has not been generated yet, thus use '' instead
        return this.dataStore.storeTrialJobEvent('ADD_CUSTOMIZED', '', hyperParams);
    }

    public async cancelTrialJobByUser(trialJobId: string): Promise<void> {
        await this.trainingService.cancelTrialJob(trialJobId);
        await this.dataStore.storeTrialJobEvent('USER_TO_CANCEL', trialJobId, '');
    }

    public async startExperiment(expParams: ExperimentParams): Promise<string> {
        this.log.debug(`Starting experiment: ${this.experimentProfile.id}`);
        this.experimentProfile.params = expParams;
        await this.storeExperimentProfile();
        this.log.debug('Setup tuner...');

        // Set up multiphase config
        if (expParams.multiPhase && this.trainingService.isMultiPhaseJobSupported) {
            this.trainingService.setClusterMetadata('multiPhase', expParams.multiPhase.toString());
        }

        const dispatcherCommand: string = getMsgDispatcherCommand(expParams.tuner, expParams.assessor, expParams.advisor,
            expParams.multiPhase, expParams.multiThread);
        this.log.debug(`dispatcher command: ${dispatcherCommand}`);
        const checkpointDir: string = await this.createCheckpointDir();
        this.setupTuner(
            dispatcherCommand,
            undefined,
            'start',
            checkpointDir);

        this.experimentProfile.startTime = Date.now();
        this.status.status = 'RUNNING';
        await this.storeExperimentProfile();
        this.run().catch((err: Error) => {
            this.criticalError(err);
        });

        return this.experimentProfile.id;
    }

    public async resumeExperiment(): Promise<void> {
        //Fetch back the experiment profile
        const experimentId: string = getExperimentId();
        this.experimentProfile = await this.dataStore.getExperimentProfile(experimentId);
        const expParams: ExperimentParams = this.experimentProfile.params;

        setInitTrialSequenceId(this.experimentProfile.maxSequenceId + 1);

        // Set up multiphase config
        if (expParams.multiPhase && this.trainingService.isMultiPhaseJobSupported) {
            this.trainingService.setClusterMetadata('multiPhase', expParams.multiPhase.toString());
        }

        const dispatcherCommand: string = getMsgDispatcherCommand(expParams.tuner, expParams.assessor, expParams.advisor,
            expParams.multiPhase, expParams.multiThread);
        this.log.debug(`dispatcher command: ${dispatcherCommand}`);
        const checkpointDir: string = await this.createCheckpointDir();
        this.setupTuner(
            dispatcherCommand,
            undefined,
            'resume',
            checkpointDir);

        const allTrialJobs: TrialJobInfo[] = await this.dataStore.listTrialJobs();

        // Resume currSubmittedTrialNum
        this.currSubmittedTrialNum = allTrialJobs.length;

        // Check the final status for WAITING and RUNNING jobs
        await Promise.all(allTrialJobs
            .filter((job: TrialJobInfo) => job.status === 'WAITING' || job.status === 'RUNNING')
            .map((job: TrialJobInfo) => this.dataStore.storeTrialJobEvent('FAILED', job.id)));

        if (this.experimentProfile.execDuration < this.experimentProfile.params.maxExecDuration &&
            this.currSubmittedTrialNum < this.experimentProfile.params.maxTrialNum &&
            this.experimentProfile.endTime) {
            delete this.experimentProfile.endTime;
        }
        this.status.status = 'RUNNING';

        // TO DO: update database record for resume event
        this.run().catch((err: Error) => {
            this.criticalError(err);
        });
    }

    public getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        return Promise.resolve(
            this.trainingService.getTrialJob(trialJobId)
        );
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        let timeoutId: NodeJS.Timer;
        // TO DO: move timeout value to constants file
        const delay1: Promise<{}> = new Promise((resolve: Function, reject: Function): void => {
            timeoutId = setTimeout(
                () => { reject(new Error('TrainingService setClusterMetadata timeout. Please check your config file.')); },
                10000);
        });
        await Promise.race([delay1, this.trainingService.setClusterMetadata(key, value)]).finally(() => {
            clearTimeout(timeoutId);
        });
    }

    public getClusterMetadata(key: string): Promise<string> {
        return Promise.resolve(
            this.trainingService.getClusterMetadata(key)
        );
    }

    public async getTrialJobStatistics(): Promise<TrialJobStatistics[]> {
        return this.dataStore.getTrialJobStatistics();
    }

    public async stopExperiment(): Promise<void> {
        this.status.status = 'STOPPING';
        this.log.info('Experiment done, cleaning up...');
        await this.experimentDoneCleanUp();
        this.log.info('Experiment done.');
    }

    public async getMetricData(trialJobId?: string, metricType?: MetricType): Promise<MetricDataRecord[]> {
        return this.dataStore.getMetricData(trialJobId, metricType);
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

    private setupTuner(command: string, cwd: string | undefined, mode: 'start' | 'resume', dataDirectory: string): void {
        if (this.dispatcher !== undefined) {
            return;
        }
        const stdio: StdioOptions = ['ignore', process.stdout, process.stderr, 'pipe', 'pipe'];
        let newCwd: string;
        if (cwd === undefined || cwd === '') {
            newCwd = getLogDir();
        } else {
            newCwd = cwd;
        }
        // TO DO: add CUDA_VISIBLE_DEVICES
        let nniEnv = {
            NNI_MODE: mode,
            NNI_CHECKPOINT_DIRECTORY: dataDirectory,
            NNI_LOG_DIRECTORY: getLogDir()
        };
        let newEnv = Object.assign({}, process.env, nniEnv);
        const tunerProc: ChildProcess = spawn(command, [], {
            stdio,
            cwd: newCwd,
            env: newEnv,
            shell: true
        });
        this.dispatcherPid = tunerProc.pid;
        this.dispatcher = createDispatcherInterface(tunerProc);

        return;
    }

    private updateTrialConcurrency(trialConcurrency: number): void {
        // we assume trialConcurrency >= 0, which is checked by restserver
        this.trialConcurrencyChange += (trialConcurrency - this.experimentProfile.params.trialConcurrency);
        this.experimentProfile.params.trialConcurrency = trialConcurrency;

        return;
    }

    private updateMaxExecDuration(duration: number): void {
        this.experimentProfile.params.maxExecDuration = duration;

        return;
    }

    private updateSearchSpace(searchSpace: string): void {
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner has not been setup');
        }
        this.dispatcher.sendCommand(UPDATE_SEARCH_SPACE, searchSpace);
        this.experimentProfile.params.searchSpace = searchSpace;

        return;
    }

    private updateMaxTrialNum(maxTrialNum: number): void {
        this.experimentProfile.params.maxTrialNum = maxTrialNum;

        return;
    }

    private async experimentDoneCleanUp(): Promise<void> {
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner has not been setup');
        }
        this.dispatcher.sendCommand(TERMINATE);
        let tunerAlive: boolean = true;
        // gracefully terminate tuner and assessor here, wait at most 30 seconds.
        for (let i: number = 0; i < 30; i++) {
            if (!tunerAlive) { break; }
            try {
                await cpp.exec(`kill -0 ${this.dispatcherPid}`);
            } catch (error) { tunerAlive = false; }
            await delay(1000);
        }
        try {
            await cpp.exec(`kill ${this.dispatcherPid}`);
        } catch (error) {
            // this.tunerPid does not exist, do nothing here
        }
        const trialJobList: TrialJobDetail[] = await this.trainingService.listTrialJobs();
        // TO DO: to promise all
        for (const trialJob of trialJobList) {
            if (trialJob.status === 'RUNNING' ||
                trialJob.status === 'WAITING') {
                try {
                    await this.trainingService.cancelTrialJob(trialJob.id);
                } catch (error) {
                    // pid does not exist, do nothing here
                }
            }
        }
        await this.trainingService.cleanUp();
        this.experimentProfile.endTime = Date.now();
        await this.storeExperimentProfile();
        this.status.status = 'STOPPED';
    }

    private async periodicallyUpdateExecDuration(): Promise<void> {
        let count: number = 1;
        while (this.status.status !== 'STOPPING' && this.status.status !== 'STOPPED') {
            await delay(1000 * 1); // 1 seconds
            if (this.status.status === 'RUNNING') {
                this.experimentProfile.execDuration += 1;
                if (count % 10 === 0) {
                    await this.storeExperimentProfile();
                }
            }
            count += 1;
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
                this.trialJobs.set(trialJobId, Object.assign({}, trialJobDetail));
                await this.dataStore.storeTrialJobEvent(trialJobDetail.status, trialJobDetail.id, undefined, trialJobDetail);
            }
            let hyperParams: string | undefined = undefined;
            switch (trialJobDetail.status) {
                case 'SUCCEEDED':
                case 'USER_CANCELED':
                case 'EARLY_STOPPED':
                    this.trialJobs.delete(trialJobId);
                    finishedTrialJobNum++;
                    if (trialJobDetail.form.jobType === 'TRIAL') {
                        hyperParams = (<TrialJobApplicationForm>trialJobDetail.form).hyperParameters.value;
                    } else {
                        throw new Error('Error: jobType error, not TRIAL');
                    }
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
                    if (trialJobDetail.form.jobType === 'TRIAL') {
                        hyperParams = (<TrialJobApplicationForm>trialJobDetail.form).hyperParameters.value;
                    } else {
                        throw new Error('Error: jobType error, not TRIAL');
                    }
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
        let allFinishedTrialJobNum: number = 0;
        let waitSubmittedToFinish: number;
        while (this.status.status !== 'STOPPING' && this.status.status !== 'STOPPED') {
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

            const requestCustomTrialNum: number = Math.min(requestTrialNum, this.customizedTrials.length);
            for (let i: number = 0; i < requestCustomTrialNum; i++) {
                // ask tuner for more trials
                if (this.customizedTrials.length > 0) {
                    const hyperParams: string | undefined = this.customizedTrials.shift();
                    this.dispatcher.sendCommand(ADD_CUSTOMIZED_TRIAL_JOB, hyperParams);
                }
            }

            if (requestTrialNum - requestCustomTrialNum > 0) {
                this.requestTrialJobs(requestTrialNum - requestCustomTrialNum);
            }

            // check maxtrialnum and maxduration here
            // NO_MORE_TRIAL is more like a subset of RUNNING, because during RUNNING tuner
            // might tell nnimanager that this is no more trials. In NO_MORE_TRIAL state, the experiment is viewed
            // as still running. DONE could be transfered from RUNNING or NO_MORE_TRIAL.
            assert(this.status.status === 'RUNNING' ||
                this.status.status === 'DONE' ||
                this.status.status === 'NO_MORE_TRIAL' ||
                this.status.status === 'TUNER_NO_MORE_TRIAL');
            if (this.experimentProfile.execDuration > this.experimentProfile.params.maxExecDuration ||
                this.currSubmittedTrialNum >= this.experimentProfile.params.maxTrialNum) {
                if (this.status.status !== 'DONE') {
                    this.status.status = 'NO_MORE_TRIAL';
                    waitSubmittedToFinish = this.currSubmittedTrialNum;

                    assert(allFinishedTrialJobNum <= waitSubmittedToFinish);
                    if (allFinishedTrialJobNum >= waitSubmittedToFinish) {
                        this.status.status = 'DONE';
                        this.experimentProfile.endTime = Date.now();
                        await this.storeExperimentProfile();
                        // write this log for travis CI
                        this.log.info('Experiment done.');
                    }
                }
            } else {
                if (this.status.status === 'DONE') {
                    delete this.experimentProfile.endTime;
                    await this.storeExperimentProfile();
                }
                if (this.status.status !== 'TUNER_NO_MORE_TRIAL') {
                    this.status.status = 'RUNNING';
                }
                for (let i: number = this.trialJobs.size; i < this.experimentProfile.params.trialConcurrency; i++) {
                    if (this.waitingTrials.length === 0 ||
                        this.currSubmittedTrialNum >= this.experimentProfile.params.maxTrialNum) {
                        break;
                    }
                    const hyperParams: string | undefined = this.waitingTrials.shift();
                    if (hyperParams === undefined) {
                        throw new Error(`Error: invalid hyper-parameters for job submission: ${hyperParams}`);
                    }
                    this.currSubmittedTrialNum++;
                    const trialJobAppForm: TrialJobApplicationForm = {
                        jobType: 'TRIAL',
                        hyperParameters: {
                            value: hyperParams,
                            index: 0
                        }
                    };
                    const trialJobDetail: TrialJobDetail = await this.trainingService.submitTrialJob(trialJobAppForm);
                    await this.storeMaxSequenceId(trialJobDetail.sequenceId);
                    this.trialJobs.set(trialJobDetail.id, Object.assign({}, trialJobDetail));
                    const trialJobDetailSnapshot: TrialJobDetail | undefined = this.trialJobs.get(trialJobDetail.id);
                    if (trialJobDetailSnapshot != undefined) {
                        await this.dataStore.storeTrialJobEvent(
                            trialJobDetailSnapshot.status, trialJobDetailSnapshot.id, hyperParams, trialJobDetailSnapshot);
                    } else {
                        assert(false, `undefined trialJobDetail in trialJobs: ${trialJobDetail.id}`);
                    }
                }
            }
            await delay(1000 * 5); // 5 seconds
        }
    }

    private storeExperimentProfile(): Promise<void> {
        this.experimentProfile.revision += 1;

        return this.dataStore.storeExperimentProfile(this.experimentProfile);
    }

    private async run(): Promise<void> {
        assert(this.dispatcher !== undefined);

        this.addEventListeners();

        this.sendInitTunerCommands();

        await Promise.all([
            this.periodicallyUpdateExecDuration(),
            this.trainingService.run().catch((err: Error) => {
                throw new NNIError('Training service error', `Training service error: ${err.message}`, err);
            }),
            this.manageTrials().catch((err: Error) => {
                throw new NNIError('Job management error', `Job management error: ${err.message}`, err);
            })]);
    }

    private addEventListeners(): void {
        // TO DO: cannot run this method more than once in one NNIManager instance
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner or job maintainer have not been setup');
        }
        this.trainingService.addTrialJobMetricListener((metric: TrialJobMetric) => {
            this.onTrialJobMetrics(metric).catch((err: Error) => {
                this.criticalError(new NNIError('Job metrics error', `Job metrics error: ${err.message}`, err));
            });
        });

        this.dispatcher.onCommand((commandType: string, content: string) => {
            this.onTunerCommand(commandType, content).catch((err: Error) => {
                this.criticalError(new NNIError('Tuner command event error', `Tuner command event error: ${err.message}`, err));
            });
        });
    }

    private sendInitTunerCommands(): void {
        if (this.dispatcher === undefined) {
            throw new Error('Dispatcher error: tuner has not been setup');
        }
        this.log.debug(`Send tuner command: INITIALIZE: ${this.experimentProfile.params.searchSpace}`);
        // Tuner need to be initialized with search space before generating any hyper parameters
        this.dispatcher.sendCommand(INITIALIZE, this.experimentProfile.params.searchSpace);
    }

    private async onTrialJobMetrics(metric: TrialJobMetric): Promise<void> {
        await this.dataStore.storeMetricData(metric.id, metric.data);
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner has not been setup');
        }
        this.dispatcher.sendCommand(REPORT_METRIC_DATA, metric.data);
    }

    private requestTrialJobs(jobNum: number): void {
        if (jobNum < 1) {
            return;
        }
        if (this.dispatcher === undefined) {
            throw new Error('Dispatcher error: tuner has not been setup');
        }
        if (this.experimentProfile.params.multiThread) {
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
        this.log.info(`Command from tuner: ${commandType}, ${content}`);
        switch (commandType) {
            case INITIALIZED:
                // Tuner is intialized, search space is set, request tuner to generate hyper parameters
                this.requestTrialJobs(this.experimentProfile.params.trialConcurrency);
                break;
            case NEW_TRIAL_JOB:
                if (this.status.status === 'TUNER_NO_MORE_TRIAL') {
                    this.log.warning('It is not supposed to receive more trials after NO_MORE_TRIAL is set');
                    this.status.status = 'RUNNING';
                }
                this.waitingTrials.push(content);
                break;
            case SEND_TRIAL_JOB_PARAMETER:
                const tunerCommand: any = JSON.parse(content);
                assert(tunerCommand.parameter_index >= 0);
                assert(tunerCommand.trial_job_id !== undefined);

                const trialJobForm: TrialJobApplicationForm = {
                    jobType: 'TRIAL',
                    hyperParameters: {
                        value: content,
                        index: tunerCommand.parameter_index
                    }
                };
                await this.trainingService.updateTrialJob(tunerCommand.trial_job_id, trialJobForm);
                await this.dataStore.storeTrialJobEvent(
                    'ADD_HYPERPARAMETER', tunerCommand.trial_job_id, content, undefined);
                break;
            case NO_MORE_TRIAL_JOBS:
                this.status.status = 'TUNER_NO_MORE_TRIAL';
                break;
            case KILL_TRIAL_JOB:
                await this.trainingService.cancelTrialJob(JSON.parse(content), true);
                break;
            default:
                throw new Error('Error: unsupported command type from tuner');
        }
    }

    private criticalError(err: Error): void {
        this.logError(err);
        console.error(err);
    }

    private logError(err: Error): void {
        if (err.stack !== undefined) {
            this.log.error(err.stack);
        }
        this.status.errors.push(err.message);
        this.status.status = 'ERROR';
    }

    private createEmptyExperimentProfile(): ExperimentProfile {
        return {
            id: getExperimentId(),
            revision: 0,
            execDuration: 0,
            logDir: getLogDir(),
            maxSequenceId: 0,
            params: {
                authorName: '',
                experimentName: '',
                trialConcurrency: 0,
                maxExecDuration: 0, // unit: second
                maxTrialNum: 0, // maxTrialNum includes all the submitted trial jobs
                trainingServicePlatform: '',
                searchSpace: ''
            }
        };
    }

    private async createCheckpointDir(): Promise<string> {
        // TODO: test
        const chkpDir: string = getCheckpointDir();
        // create checkpoint directory
        await mkDirP(chkpDir);
        // assign this directory to exp profile's checkpointDir
        if (this.experimentProfile.params.advisor) {
            this.experimentProfile.params.advisor.checkpointDir = chkpDir;
        }
        if (this.experimentProfile.params.tuner) {
            this.experimentProfile.params.tuner.checkpointDir = chkpDir;
        }
        if (this.experimentProfile.params.assessor) {
            this.experimentProfile.params.assessor.checkpointDir = chkpDir;
        }

        return Promise.resolve(chkpDir);
    }

    private async storeMaxSequenceId(sequenceId: number): Promise<void> {
        if (sequenceId > this.experimentProfile.maxSequenceId) {
            this.experimentProfile.maxSequenceId = sequenceId;
            await this.storeExperimentProfile();
        }
    }
}

export { NNIManager };
