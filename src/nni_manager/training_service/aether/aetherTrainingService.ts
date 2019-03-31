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

import { ChildProcess, spawn } from 'child_process';
// import * as fs from 'fs';
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
import { delay, getExperimentRootDir, uniqueString } from '../../common/utils';
import { TrialConfigMetadataKey } from '../../training_service/common/trialConfigMetadataKey';
import { AetherConfig, AetherTrialJobDetail } from './aetherData';
import { AetherJobRestServer } from './aetherJobRestServer';

// tslint:disable-next-line:completed-docs
@component.Singleton
class AetherTrainingService implements TrainingService {

    public get isMultiPhaseJobSupported(): boolean {
        return false;
    }

    public get MetricsEmitter(): EventEmitter {
        return this.metricsEmitter;
    }

    public  readonly trialJobsMap: Map<string, AetherTrialJobDetail>;
    private aetherClientExePath: string = '/fake/path/to/exe';
    private readonly metricsEmitter: EventEmitter;
    private nextTrialSequenceId: number;
    private readonly log!: Logger;
    private nniManagerIpConfig!: NNIManagerIpConfig;
    private aetherJobConfig!: AetherConfig;
    private readonly runDeferred: Deferred<void>;

    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, AetherTrialJobDetail>();
        this.nextTrialSequenceId = -1;
        this.runDeferred = new Deferred<void>();
        this.log.info('Aether Training Service Constructed.');
    }

    public async run(): Promise<void> {
        this.log.info('Run Aether training service.');
        const restServer: AetherJobRestServer = component.get(AetherJobRestServer);
        await restServer.start();

        this.log.info(`Aether Training service rest server listening on: ${restServer.endPoint}`);

        this.runDeferred.promise.then(() => {
            this.log.info('Aether Training service exit.');
        });

        return this.runDeferred.promise;
    }

    public async listTrialJobs(): Promise<AetherTrialJobDetail[]> {
        const jobs: AetherTrialJobDetail[] = [];
        for (const [key, value] of this.trialJobsMap) {
            if (value.form.jobType === 'TRIAL') {
                jobs.push(value);
            }
        }

        return Promise.resolve(jobs);
    }

    public async getTrialJob(trialJobId: string): Promise<AetherTrialJobDetail> {
        const trial: AetherTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (!trial) {
            return Promise.reject(`Trial job ${trialJobId} not found`);
        }

        return Promise.resolve(trial);
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<AetherTrialJobDetail> {
        const deferred: Deferred<AetherTrialJobDetail> = new Deferred<AetherTrialJobDetail>();
        if (!this.nniManagerIpConfig) {
            return Promise.reject('nnimanager ip not initialized');
        }
        if (!this.aetherJobConfig) {
            return Promise.reject('Aether job config not initialized');
        }

        const trialJobId: string = uniqueString(5);
        const trialSequencdId: number = this.generateSequenceId();
        const trialWorkingDirectory: string = path.join(getExperimentRootDir(), 'trials', trialJobId);
        const clientCmdArgs: string[] = [this.nniManagerIpConfig.nniManagerIp, getExperimentId(), trialJobId];
        this.log.debug(`execute command: ${this.aetherClientExePath} ${clientCmdArgs.join(' ')}`);
        const clientProc: ChildProcess = spawn(this.aetherClientExePath, clientCmdArgs);
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

        const guid: string = await trialDetail.guid.promise;
        trialDetail.url = `aether:\\experiment\\${guid}`;
        this.trialJobsMap.set(trialJobId, trialDetail);
        deferred.resolve(trialDetail);

        return deferred.promise;
    }

    public async updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<AetherTrialJobDetail> {
        throw new MethodNotImplementedError();
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.off('metric', listener);
    }
    public async cancelTrialJob(trialJobId: string): Promise<void> {
        const trial: AetherTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (!trial) {
            return Promise.reject(`Trial ${trialJobId} not found`);
        }
        trial.clientProc.kill();
        trial.status = 'SYS_CANCELED';  //USER or SYS CANCELLED?

        return Promise.resolve();
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {

        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig> JSON.parse(value);
                break;
            case TrialConfigMetadataKey.AETHER_CONFIG:
                this.aetherJobConfig = <AetherConfig> JSON.parse(value);
                break;
            default:
                throw new Error(`Unknown key: ${key}`);
        }

        return Promise.resolve();
    }

    public async getClusterMetadata(key: string): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();

        return deferred.promise;
    }

    public async cleanUp(): Promise<void> {
        this.log.info('Stopping Aether Training Service...');
        this.runDeferred.resolve();

        return Promise.resolve();
    }

    private generateSequenceId(): number {
        if (this.nextTrialSequenceId === -1) {
            this.nextTrialSequenceId = getInitTrialSequenceId();
        }

        return this.nextTrialSequenceId++;
    }
}

export { AetherTrainingService, AetherTrialJobDetail };
