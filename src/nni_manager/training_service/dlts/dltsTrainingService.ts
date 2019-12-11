// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as component from '../../common/component';
import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import {
    HyperParameters, NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric
} from '../../common/trainingService';
import { delay } from '../../common/utils';
import { DLTSJobRestServer, ParameterFileMeta } from './dltsJobRestServer';

@component.Singleton
class DLTSTrainingService implements TrainingService {
    private readonly log!: Logger;
    private readonly metricsEmitter: EventEmitter;
    //private readonly expRootDir: string;
    private readonly jobQueue: string[];
    private stopping: boolean = false;
    private readonly experimentId!: string;
    private versionCheck: boolean = true;
    // TODO: more member variables 

    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.jobQueue = [];
        this.experimentId = getExperimentId();
        this.log.info('Construct DLTS training service.');
    }

    public async run(): Promise<void> {
        this.log.info('Run DLTS training service.');
        const restServer: DLTSJobRestServer = component.get(DLTSJobRestServer);
        await restServer.start();
        restServer.setEnableVersionCheck = this.versionCheck;
        this.log.info(`DLTS Training service rest server listening on: ${restServer.endPoint}`);
        await Promise.all([
            this.statusCheckingLoop(),
            this.submitJobLoop()]);
        this.log.info('DLTS training service exit.');
    }

    private async statusCheckingLoop(): Promise<void> {
        // TODO: ...
        // this function does three things:
        // 1. update token
        // 2. update the status of submitted trial jobs
        // 3. check error msg from JobRestServer
        // you can refactor the logic if you don't like it.
        while (!this.stopping) {
            // TODO: ...
            await delay(3000);
        }
    }

    private async submitJobLoop(): Promise<void> {
        while (!this.stopping) {
            while (!this.stopping && this.jobQueue.length > 0) {
                const trialJobId: string = this.jobQueue[0];
                if (await this.submitTrialJobToDLTS(trialJobId)) {
                    // Remove trial job with trialJobId from job queue
                    this.jobQueue.shift();
                } else {
                    // Break the while loop since failed to submitJob
                    break;
                }
            }
            await delay(3000);
        }
    }

    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        // TODO: ...
        return Promise.resolve([]);
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        // TODO: ...
        return Promise.resolve(null);
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.off('metric', listener);
    }

    public get MetricsEmitter(): EventEmitter {
        return this.metricsEmitter;
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        // TODO: ...
        // possible items: prepare trialJobDetail based on the submitted `form`,
        // then enqueue it into `jobQueue`
        return Promise.resolve(null);
    }

    public cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        // TODO: ...
        return Promise.resolve();
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        // TODO: ...
        // this function is used to receive training service related configuration items
        // in experiment configuration file
        return Promise.resolve();
    }

    public getClusterMetadata(key: string): Promise<string> {
        // this function is not used by now
        return Promise.resolve(null);
    }

    public async cleanUp(): Promise<void> {
        this.log.info('Stopping DLTS training service...');
        this.stopping = true;

        const deferred: Deferred<void> = new Deferred<void>();
        const restServer: DLTSJobRestServer = component.get(DLTSJobRestServer);
        try {
            await restServer.stop();
            deferred.resolve();
            this.log.info('DLTS Training service rest server stopped successfully.');
        } catch (error) {
            // tslint:disable-next-line: no-unsafe-any
            this.log.error(`DLTS Training service rest server stopped failed, error: ${error.message}`);
            deferred.reject(error);
        }

        return deferred.promise;
    }

    private async submitTrialJobToDLTS(trialJobId: string): Promise<boolean> {
        // TODO: ...
        // possible steps:
        // Step 1. Prepare DLTS job configuration
        // Step 2. Upload code files in codeDir onto HDFS
        // Step 3. Submit DLTS job via Rest call
        return Promise.resolve(null);
    }
}

export { DLTSTrainingService };