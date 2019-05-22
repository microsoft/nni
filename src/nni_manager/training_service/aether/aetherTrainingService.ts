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

import { ChildProcess, spawn} from 'child_process';
import * as fs from 'fs';
// import * as request from 'request';
import * as path from 'path';
import * as component from '../../common/component';

import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';
import { MethodNotImplementedError } from '../../common/errors';
import { getBasePort, getExperimentId, getInitTrialSequenceId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import {
    JobApplicationForm,
    NNIManagerIpConfig,
    TrainingService,
    TrialJobMetric
} from '../../common/trainingService';
import { delay, getExperimentRootDir, uniqueString, mkDirP } from '../../common/utils';
import { TrialConfigMetadataKey } from '../../training_service/common/trialConfigMetadataKey';
import { AetherConfig, AetherTrialJobDetail } from './aetherData';
import { AetherJobRestServer } from './aetherJobRestServer';
import { AssertionError } from 'assert';

// tslint:disable-next-line:completed-docs
@component.Singleton
class AetherTrainingService implements TrainingService {

    private resolveAetherClient(): string {
        if (path.extname(__filename) == '.ts') {
            // testing mode
            return path.resolve(__dirname, '../../dist/aether/AetherClient.exe');
        } else if (path.extname(__filename) == '.js') {
            // production mode
            return path.resolve(__dirname, '../../aether/AetherClient.exe');
        } else {
            throw new Error(`${__filename} wrong file extension!`);
        }
    }

    public  readonly trialJobsMap: Map<string, AetherTrialJobDetail>;
    private aetherClientExePath: string;
    private readonly metricsEmitter: EventEmitter;
    private nextTrialSequenceId: number;
    private readonly log!: Logger;
    private nniManagerIpConfig!: NNIManagerIpConfig;
    private aetherJobConfig!: AetherConfig;
    private readonly runDeferred: Deferred<void>;
    private restServerAddress!: string;

    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, AetherTrialJobDetail>();
        this.nextTrialSequenceId = -1;
        this.runDeferred = new Deferred<void>();
        this.aetherClientExePath = this.resolveAetherClient();
        this.log.info('Aether Training Service Constructed.');
    }

    public async run(): Promise<void> {
        this.log.info('Run Aether training service.');
        const restServer: AetherJobRestServer = component.get(AetherJobRestServer);
        this.restServerAddress = `http://${this.nniManagerIpConfig.nniManagerIp}:${restServer.clusterRestServerPort}`;
        await restServer.start();

        this.log.info(`Aether Training service rest server listening on: ${restServer.endPoint}`);

        await this.runDeferred.promise;
        this.log.info('Aether Training service exit.');

        return Promise.resolve();
    }

    public listTrialJobs(): Promise<AetherTrialJobDetail[]> {
        const jobs: AetherTrialJobDetail[] = [];
        for (const [key, value] of this.trialJobsMap) {
            if (value.form.jobType === 'TRIAL') {
                jobs.push(value);
            }
        }

        return Promise.resolve(jobs);
    }

    public getTrialJob(trialJobId: string): Promise<AetherTrialJobDetail> {
        const trial: AetherTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (!trial) {
            return Promise.reject(new Error(`Trial job ${trialJobId} not found`));
        }

        return Promise.resolve(trial);
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.off('metric', listener);
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<AetherTrialJobDetail> {
        if (!this.nniManagerIpConfig) {
            throw new Error('nnimanager ip not initialized');
        }
        if (!this.aetherJobConfig) {
            throw new Error('Aether job config not initialized');
        }
        if (!this.restServerAddress) {
            throw new Error('Training Service Restful server for Aether Client not initialized');
        }

        const trialJobId: string = uniqueString(5);
        const trialSequencdId: number = this.generateSequenceId();
        const trialWorkingDirectory: string = path.join(getExperimentRootDir(), 'trials', trialJobId);
        await mkDirP(trialWorkingDirectory);

        // prepare daemon process, redirect stdout & stderr
        const clientCmdArgs: string[] = [this.restServerAddress, getExperimentId(), trialJobId];
        this.log.info(`execute command: ${this.aetherClientExePath} ${clientCmdArgs.join(' ')}`);
        const stderr = fs.createWriteStream(path.join(trialWorkingDirectory, 'stderr'), );
        const stderr_open: Deferred<void> = new Deferred<void>();
        stderr.on('open', () => {stderr_open.resolve();});
        const stdout = fs.createWriteStream(path.join(trialWorkingDirectory, 'stdout'));
        const stdout_open: Deferred<void> = new Deferred<void>();
        stdout.on('open', () => {stdout_open.resolve();});
        await Promise.all([stderr_open.promise, stdout_open.promise]);
        const clientProc: ChildProcess = spawn(this.aetherClientExePath, clientCmdArgs, {stdio: ['pipe', stdout, stderr]});
        clientProc.on('exit', (code) => {
            if (code == 1) {
                throw new Error(`Aether Client process for trial ${trialJobId} exited with error, please check ${path.join(trialWorkingDirectory, 'stderr')} to see details`);
            }
        });
        clientProc.on('error', (err) => {
                throw new Error(`Failed to start subprocess: ${err}`);
        });

        const trialDetail: AetherTrialJobDetail = new AetherTrialJobDetail(
            trialJobId,
            'WAITING',
            Date.now(),
            trialWorkingDirectory,
            form,
            trialSequencdId,
            clientProc,
            this.aetherJobConfig
        );
        this.trialJobsMap.set(trialJobId, trialDetail);
        const guid: string = await trialDetail.guid.promise;
        trialDetail.url = `aether:\\experiment\\${guid}`;
        return trialDetail;
    }

    public async updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<AetherTrialJobDetail> {
        throw new MethodNotImplementedError();
    }

    public get isMultiPhaseJobSupported(): boolean {
        return false;
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trial: AetherTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (!trial) {
            return Promise.reject(new Error(`Trial ${trialJobId} not found`));
        }

        if (!trial.clientProc.killed) {
            trial.isEarlyStopped = isEarlyStopped;
            const deferred_exit = new Deferred<void>();
            trial.clientProc.on('exit', (code) => {
                if (code == 0) {
                    trial.status = 'USER_CANCELED';  //USER or SYS CANCELLED?
                    deferred_exit.resolve();
                }
            })
            trial.clientProc.stdin.write('0');
            return deferred_exit.promise;
        }

        this.log.warning(`Daemon Process for Trial ${trialJobId} already exited.`);

        return Promise.resolve();
    }

    public setClusterMetadata(key: string, value: string): Promise<void> {

        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig> JSON.parse(value);
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                this.aetherJobConfig = <AetherConfig> JSON.parse(value);
                break;
            default:
                return Promise.reject(new Error(`Unknown key: ${key}`));
        }

        return Promise.resolve();
    }

    public async getClusterMetadata(key: string): Promise<string> {
        throw new MethodNotImplementedError();
    }

    public async cleanUp(): Promise<void> {
        this.log.info('Stopping Aether Training Service...');
        this.runDeferred.resolve();

        // cancel running jobs?
        return Promise.resolve();
    }

    public get MetricsEmitter(): EventEmitter {
        return this.metricsEmitter;
    }

    private generateSequenceId(): number {
        if (this.nextTrialSequenceId === -1) {
            this.nextTrialSequenceId = getInitTrialSequenceId();
        }

        return this.nextTrialSequenceId++;
    }
}

export { AetherTrainingService, AetherTrialJobDetail };
