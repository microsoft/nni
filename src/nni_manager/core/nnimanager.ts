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
import { ChildProcess, spawn } from 'child_process';
import { Deferred } from 'ts-deferred';
import * as component from '../common/component';
import { DataStore, MetricDataRecord, MetricType, TrialJobInfo } from '../common/datastore';
import { getExperimentId } from '../common/experimentStartupInfo';
import { getLogger, Logger } from '../common/log';
import {
    ExperimentParams, ExperimentProfile, Manager,
    ProfileUpdateType, TrialJobStatistics
} from '../common/manager';
import {
    TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric, TrialJobStatus
} from '../common/trainingService';
import { delay , getLogDir, getMsgDispatcherCommand} from '../common/utils';
import {
    ADD_CUSTOMIZED_TRIAL_JOB, KILL_TRIAL_JOB, NEW_TRIAL_JOB, NO_MORE_TRIAL_JOBS, REPORT_METRIC_DATA,
    REQUEST_TRIAL_JOBS, TERMINATE, TRIAL_END, UPDATE_SEARCH_SPACE
} from './commands';
import { createDispatcherInterface, IpcInterface } from './ipcInterface';
import { TrialJobMaintainerEvent, TrialJobs } from './trialJobs';

/**
 * NNIManager
 */
class NNIManager implements Manager {
    private trainingService: TrainingService;
    private dispatcher: IpcInterface | undefined;
    private trialJobsMaintainer: TrialJobs | undefined;
    private currSubmittedTrialNum: number; // need to be recovered
    private trialConcurrencyReduction: number;
    private customizedTrials: string[]; // need to be recovered
    private log: Logger;
    private dataStore: DataStore;
    private experimentProfile: ExperimentProfile;
    private dispatcherPid: number;

    constructor() {
        this.currSubmittedTrialNum = 0;
        this.trialConcurrencyReduction = 0;
        this.customizedTrials = [];
        const experimentId: string = getExperimentId();
        this.trainingService = component.get(TrainingService);
        assert(this.trainingService);
        this.dispatcherPid = 0;

        this.log = getLogger();
        this.dataStore = component.get(DataStore);
        this.experimentProfile = {
            id: experimentId,
            revision: 0,
            execDuration: 0,
            params: {
                authorName: '',
                experimentName: '',
                trialConcurrency: 0,
                maxExecDuration: 0, // unit: second
                maxTrialNum: 0, // maxTrialNum includes all the submitted trial jobs
                searchSpace: '',
                tuner: {
                    className: '',
                    classArgs: {},
                    checkpointDir: ''
                }
            }
        };
    }

    public updateExperimentProfile(experimentProfile: ExperimentProfile, updateType: ProfileUpdateType): Promise<void> {
        // TO DO: remove this line, and let rest server do data type validation
        experimentProfile.startTime = new Date(<string><any>experimentProfile.startTime);
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

        const dispatcherCommand: string = getMsgDispatcherCommand(expParams.tuner, expParams.assessor);
        console.log(`dispatcher command: ${dispatcherCommand}`);
        this.setupTuner(
            //expParams.tuner.tunerCommand,
            dispatcherCommand,
            undefined,
            'start',
            expParams.tuner.checkpointDir);

        this.experimentProfile.startTime = new Date();
        await this.storeExperimentProfile();
        this.run().catch(err => {
            this.log.error(err.stack);
        });
        return this.experimentProfile.id;
    }

    public async resumeExperiment(): Promise<void> {
        //Fetch back the experiment profile
        const experimentId: string = getExperimentId();
        this.experimentProfile = await this.dataStore.getExperimentProfile(experimentId);
        const expParams: ExperimentParams = this.experimentProfile.params;

        const dispatcherCommand: string = getMsgDispatcherCommand(expParams.tuner, expParams.assessor);
        console.log(`dispatcher command: ${dispatcherCommand}`);
        this.setupTuner(
            dispatcherCommand,
            undefined,
            'resume',
            expParams.tuner.checkpointDir);

        const allTrialJobs: TrialJobInfo[] = await this.dataStore.listTrialJobs();

        // Resume currSubmittedTrialNum
        this.currSubmittedTrialNum = allTrialJobs.length;

        // Check the final status for WAITING and RUNNING jobs
        await Promise.all(allTrialJobs
            .filter((job: TrialJobInfo) => job.status === 'WAITING' || job.status === 'RUNNING')
            .map((job: TrialJobInfo) => this.dataStore.storeTrialJobEvent('FAILED', job.id)));

        // TO DO: update database record for resume event
        this.run().catch(console.error);
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

    public stopExperiment(): Promise<void> {
        if (this.trialJobsMaintainer !== undefined) {
            this.trialJobsMaintainer.setStopLoop();

            return Promise.resolve();
        } else {
            return Promise.reject(new Error('Error: undefined trialJobsMaintainer'));
        }
    }

    public async getMetricData(trialJobId: string, metricType: MetricType): Promise<MetricDataRecord[]> {
        return this.dataStore.getMetricData(trialJobId, metricType);
    }

    public getExperimentProfile(): Promise<ExperimentProfile> {
        // TO DO: using Promise.resolve()
        const deferred: Deferred<ExperimentProfile> = new Deferred<ExperimentProfile>();
        deferred.resolve(this.experimentProfile);

        return deferred.promise;
    }

    public async listTrialJobs(status?: TrialJobStatus): Promise<TrialJobInfo[]> {
        return this.dataStore.listTrialJobs(status);
    }

    private setupTuner(command: string, cwd: string | undefined, mode: 'start' | 'resume', dataDirectory: string): void {
        if (this.dispatcher !== undefined) {
            return;
        }
        const stdio: (string | NodeJS.WriteStream)[] = ['ignore', process.stdout, process.stderr, 'pipe', 'pipe'];
        let newCwd: string;
        if (cwd === undefined || cwd === '') {
            newCwd = getLogDir();
        } else {
            newCwd = cwd;
        }
        // TO DO: add CUDA_VISIBLE_DEVICES
        const tunerProc: ChildProcess = spawn(command, [], {
            stdio,
            cwd: newCwd,
            env: {
                NNI_MODE: mode,
                NNI_CHECKPOINT_DIRECTORY: dataDirectory,
                NNI_LOG_DIRECTORY: getLogDir()
            },
            shell: true
        });
        this.dispatcherPid = tunerProc.pid;
        this.dispatcher = createDispatcherInterface(tunerProc);

        return;
    }

    private updateTrialConcurrency(trialConcurrency: number): void {
        // TO DO: this method can only be called after startExperiment/resumeExperiment
        if (trialConcurrency > this.experimentProfile.params.trialConcurrency) {
            if (this.dispatcher === undefined) {
                throw new Error('Error: tuner has to be initialized');
            }
            this.dispatcher.sendCommand(
                REQUEST_TRIAL_JOBS,
                String(trialConcurrency - this.experimentProfile.params.trialConcurrency)
            );
        } else {
            // we assume trialConcurrency >= 0, which is checked by restserver
            this.trialConcurrencyReduction += (this.experimentProfile.params.trialConcurrency - trialConcurrency);
        }
        this.experimentProfile.params.trialConcurrency = trialConcurrency;

        return;
    }

    private updateMaxExecDuration(duration: number): void {
        if (this.trialJobsMaintainer !== undefined) {
            this.trialJobsMaintainer.updateMaxExecDuration(duration);
        }
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
        this.experimentProfile.endTime = new Date();
        await this.storeExperimentProfile();
    }

    private async periodicallyUpdateExecDuration(): Promise<void> {
        const startTime: Date = new Date();
        const execDuration: number = this.experimentProfile.execDuration;
        for (; ;) {
            await delay(1000 * 60 * 10); // 10 minutes
            this.experimentProfile.execDuration = execDuration + (Date.now() - startTime.getTime()) / 1000;
            await this.storeExperimentProfile();
        }
    }

    private storeExperimentProfile(): Promise<void> {
        this.experimentProfile.revision += 1;

        return this.dataStore.storeExperimentProfile(this.experimentProfile);
    }

    // tslint:disable-next-line:max-func-body-length
    private runInternal(): Promise<void> {
        // TO DO: cannot run this method more than once in one NNIManager instance
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner has not been setup');
        }
        this.trainingService.addTrialJobMetricListener(async (metric: TrialJobMetric) => {
            await this.dataStore.storeMetricData(metric.id, metric.data);
            if (this.dispatcher === undefined) {
                throw new Error('Error: tuner has not been setup');
            }
            this.dispatcher.sendCommand(REPORT_METRIC_DATA, metric.data);
        });

        this.trialJobsMaintainer = new TrialJobs(
            this.trainingService,
            this.experimentProfile.execDuration,
            this.experimentProfile.params.maxExecDuration);
        this.trialJobsMaintainer.on(async (event: TrialJobMaintainerEvent, trialJobDetail: TrialJobDetail) => {
            if (trialJobDetail !== undefined) {
                this.log.debug(`Job event: ${event}, id: ${trialJobDetail.id}`);
            } else {
                this.log.debug(`Job event: ${event}`);
            }
            if (this.dispatcher === undefined) {
                throw new Error('Error: tuner has not been setup');
            }
            switch (event) {
                case 'SUCCEEDED':
                case 'FAILED':
                case 'USER_CANCELED':
                case 'SYS_CANCELED':
                    if (this.trialConcurrencyReduction > 0) {
                        this.trialConcurrencyReduction--;
                    } else {
                        if (this.currSubmittedTrialNum < this.experimentProfile.params.maxTrialNum) {
                            if (this.customizedTrials.length > 0) {
                                const hyperParams: string | undefined = this.customizedTrials.shift();
                                this.dispatcher.sendCommand(ADD_CUSTOMIZED_TRIAL_JOB, hyperParams);
                            } else {
                                this.dispatcher.sendCommand(REQUEST_TRIAL_JOBS, '1');
                            }
                        }
                    }
                    this.dispatcher.sendCommand(TRIAL_END, JSON.stringify({trial_job_id: trialJobDetail.id, event: event}));
                    await this.dataStore.storeTrialJobEvent(event, trialJobDetail.id, undefined, trialJobDetail.url);
                    break;
                case 'RUNNING':
                    await this.dataStore.storeTrialJobEvent(event, trialJobDetail.id, undefined, trialJobDetail.url);
                    break;
                case 'EXPERIMENT_DONE':
                    this.log.info('Experiment done, cleaning up...');
                    await this.experimentDoneCleanUp();
                    this.log.info('Experiment done.');
                    break;
                default:
                    throw new Error('Error: unrecognized event from trialJobsMaintainer');
            }
        });

        // TO DO: we should send INITIALIZE command to tuner if user's tuner needs to run init method in tuner
        this.log.debug(`Send tuner command: update search space: ${this.experimentProfile.params.searchSpace}`);
        this.dispatcher.sendCommand(UPDATE_SEARCH_SPACE, this.experimentProfile.params.searchSpace);
        if (this.trialConcurrencyReduction !== 0) {
            return Promise.reject(new Error('Error: cannot modify trialConcurrency before startExperiment'));
        }
        this.log.debug(`Send tuner command: ${this.experimentProfile.params.trialConcurrency}`)
        this.dispatcher.sendCommand(REQUEST_TRIAL_JOBS, String(this.experimentProfile.params.trialConcurrency));
        this.dispatcher.onCommand(async (commandType: string, content: string) => {
            this.log.info(`Command from tuner: ${commandType}, ${content}`);
            if (this.trialJobsMaintainer === undefined) {
                throw new Error('Error: trialJobsMaintainer not initialized');
            }
            switch (commandType) {
                case NEW_TRIAL_JOB:
                    if (this.currSubmittedTrialNum < this.experimentProfile.params.maxTrialNum) {
                        this.currSubmittedTrialNum++;
                        const trialJobAppForm: TrialJobApplicationForm = {
                            jobType: 'TRIAL',
                            hyperParameters: content
                        };
                        const trialJobDetail: TrialJobDetail = await this.trainingService.submitTrialJob(trialJobAppForm);
                        this.trialJobsMaintainer.setTrialJob(trialJobDetail.id, Object.assign({}, trialJobDetail));
                        assert(trialJobDetail.status === 'WAITING');
                        await this.dataStore.storeTrialJobEvent(trialJobDetail.status, trialJobDetail.id, content, trialJobDetail.url);
                        if (this.currSubmittedTrialNum === this.experimentProfile.params.maxTrialNum) {
                            this.trialJobsMaintainer.setNoMoreTrials();
                        }
                    }
                    break;
                case NO_MORE_TRIAL_JOBS:
                    this.trialJobsMaintainer.setNoMoreTrials();
                    break;
                case KILL_TRIAL_JOB:
                    await this.trainingService.cancelTrialJob(JSON.parse(content));
                    break;
                default:
                    throw new Error(`Error: unsupported command type: [${commandType}]`);
            }
        });

        return this.trialJobsMaintainer.run();
    }

    private async run(): Promise<void> {
        await Promise.all([
            this.periodicallyUpdateExecDuration(),
            this.trainingService.run(),
            this.runInternal()]);
    }
}

export { NNIManager };
