// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as path from 'path';
import * as request from 'request';
import * as component from '../../common/component';

import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import { MethodNotImplementedError } from '../../common/errors';
import {
    NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric, LogType
} from '../../common/trainingService';
import { delay } from '../../common/utils';
import { PAIJobInfoCollector } from './paiJobInfoCollector';
import { PAIJobRestServer } from './paiJobRestServer';
import { PAIClusterConfig, PAITrialJobDetail } from './paiConfig';

/**
 * Training Service implementation for OpenPAI (Open Platform for AI)
 * Refer https://github.com/Microsoft/pai for more info about OpenPAI
 */
@component.Singleton
abstract class PAITrainingService implements TrainingService {
    protected readonly log!: Logger;
    protected readonly metricsEmitter: EventEmitter;
    protected readonly trialJobsMap: Map<string, PAITrialJobDetail>;
    protected readonly expRootDir: string;
    protected paiClusterConfig?: PAIClusterConfig;
    protected readonly jobQueue: string[];
    protected stopping: boolean = false;
    protected paiToken?: string;
    protected paiTokenUpdateTime?: number;
    protected readonly paiTokenUpdateInterval: number;
    protected readonly experimentId!: string;
    protected readonly paiJobCollector: PAIJobInfoCollector;
    protected paiRestServerPort?: number;
    protected nniManagerIpConfig?: NNIManagerIpConfig;
    protected versionCheck: boolean = true;
    protected logCollection: string;
    protected isMultiPhase: boolean = false;
    protected authFileHdfsPath: string | undefined = undefined;
    protected portList?: string | undefined;
    protected paiJobRestServer?: PAIJobRestServer;
    protected protocol: string = 'http';

    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, PAITrialJobDetail>();
        this.jobQueue = [];
        this.expRootDir = path.join('/nni-experiments', getExperimentId());
        this.experimentId = getExperimentId();
        this.paiJobCollector = new PAIJobInfoCollector(this.trialJobsMap);
        this.paiTokenUpdateInterval = 7200000; //2hours
        this.logCollection = 'none';
        this.log.info('Construct paiBase training service.');
    }

    public async run(): Promise<void> {
        this.log.info('Run PAI training service.');
        if (this.paiJobRestServer === undefined) {
            throw new Error('paiJobRestServer not initialized!');
        }
        await this.paiJobRestServer.start();
        this.paiJobRestServer.setEnableVersionCheck = this.versionCheck;
        this.log.info(`PAI Training service rest server listening on: ${this.paiJobRestServer.endPoint}`);
        await Promise.all([
            this.statusCheckingLoop(),
            this.submitJobLoop()]);
        this.log.info('PAI training service exit.');
    }

    public async submitTrialJob(_form: TrialJobApplicationForm): Promise<any> {
        throw new Error('Not implemented!');
    }

    public async updateTrialJob(_trialJobId: string, _form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        throw new Error('Not implemented!');
    }

    protected async submitTrialJobToPAI(_trialJobId: string): Promise<boolean> {
        throw new Error('Not implemented!');
    }

    protected async submitJobLoop(): Promise<void> {
        while (!this.stopping) {
            while (!this.stopping && this.jobQueue.length > 0) {
                const trialJobId: string = this.jobQueue[0];
                if (await this.submitTrialJobToPAI(trialJobId)) {
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

    public async setClusterMetadata(_key: string, _value: string): Promise<void> {
        throw new Error('Not implemented!');
    }

    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        const jobs: TrialJobDetail[] = [];

        for (const key of this.trialJobsMap.keys()) {
            jobs.push(await this.getTrialJob(key));
        }

        return jobs;
    }

    public async getTrialLog(_trialJobId: string, _logType: LogType): Promise<string> {
        throw new MethodNotImplementedError();
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        if (this.paiClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }

        const paiTrialJob: PAITrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (paiTrialJob === undefined) {
            throw new Error(`trial job ${trialJobId} not found`);
        }

        return paiTrialJob;
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

    public cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trialJobDetail: PAITrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            return Promise.reject(new Error(`cancelTrialJob: trial job id ${trialJobId} not found`));
        }

        if (this.paiClusterConfig === undefined) {
            return Promise.reject(new Error('PAI Cluster config is not initialized'));
        }
        if (this.paiToken === undefined) {
            return Promise.reject(new Error('PAI token is not initialized'));
        }

        if (trialJobDetail.status === 'UNKNOWN') {
            trialJobDetail.status = 'USER_CANCELED';
            return Promise.resolve();
        }

        const stopJobRequest: request.Options = {
            uri: `${this.protocol}://${this.paiClusterConfig.host}/rest-server/api/v2/jobs/${this.paiClusterConfig.userName}~${trialJobDetail.paiJobName}/executionType`,
            method: 'PUT',
            json: true,
            body: { value: 'STOP' },
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${this.paiToken}`
            }
        };

        // Set trialjobDetail's early stopped field, to mark the job's cancellation source
        trialJobDetail.isEarlyStopped = isEarlyStopped;
        const deferred: Deferred<void> = new Deferred<void>();

        request(stopJobRequest, (error: Error, response: request.Response, _body: any) => {
            // Status code 202 for success.
            if ((error !== undefined && error !== null) || response.statusCode >= 400) {
                this.log.error(`PAI Training service: stop trial ${trialJobId} to PAI Cluster failed!`);
                deferred.reject((error !== undefined && error !== null) ? error.message :
                    `Stop trial failed, http code: ${response.statusCode}`);
            } else {
                deferred.resolve();
            }
        });

        return deferred.promise;
    }

    public getClusterMetadata(_key: string): Promise<string> {
        throw new Error('Not implemented!');
    }

    public async cleanUp(): Promise<void> {
        this.log.info('Stopping PAI training service...');
        this.stopping = true;

        if (this.paiJobRestServer === undefined) {
            throw new Error('paiJobRestServer not initialized!');
        }

        try {
            await this.paiJobRestServer.stop();
            this.log.info('PAI Training service rest server stopped successfully.');
        } catch (error) {
            this.log.error(`PAI Training service rest server stopped failed, error: ${error.message}`);
        }
    }

    public get MetricsEmitter(): EventEmitter {
        return this.metricsEmitter;
    }

    protected formatPAIHost(host: string): string {
        // If users' host start with 'http://' or 'https://', use the original host,
        // or format to 'http//${host}'
        if (host.startsWith('http://')) {
            this.protocol = 'http';
            return host.replace('http://', '');
        } else if (host.startsWith('https://')) {
            this.protocol = 'https';
            return host.replace('https://', '');
        } else {
            return host;
        }
    }

    protected async statusCheckingLoop(): Promise<void> {
        while (!this.stopping) {
            if (this.paiClusterConfig && this.paiClusterConfig.passWord) {
                try {
                    await this.updatePaiToken();
                } catch (error) {
                    this.log.error(`${error}`);
                    //only throw error when initlize paiToken first time
                    if (this.paiToken === undefined) {
                        throw new Error(error);
                    }
                }
            }
            await this.paiJobCollector.retrieveTrialStatus(this.protocol, this.paiToken, this.paiClusterConfig);
            if (this.paiJobRestServer === undefined) {
                throw new Error('paiBaseJobRestServer not implemented!');
            }
            if (this.paiJobRestServer.getErrorMessage !== undefined) {
                throw new Error(this.paiJobRestServer.getErrorMessage);
            }
            await delay(3000);
        }
    }

    /**
     * Update pai token by the interval time or initialize the pai token
     */
    protected async updatePaiToken(): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();

        const currentTime: number = new Date().getTime();
        //If pai token initialized and not reach the interval time, do not update
        if (this.paiTokenUpdateTime !== undefined && (currentTime - this.paiTokenUpdateTime) < this.paiTokenUpdateInterval) {
            return Promise.resolve();
        }

        if (this.paiClusterConfig === undefined) {
            const paiClusterConfigError: string = `pai cluster config not initialized!`;
            this.log.error(`${paiClusterConfigError}`);
            throw Error(`${paiClusterConfigError}`);
        }

        const authenticationReq: request.Options = {
            uri: `${this.protocol}://${this.paiClusterConfig.host}/rest-server/api/v1/token`,
            method: 'POST',
            json: true,
            body: {
                username: this.paiClusterConfig.userName,
                password: this.paiClusterConfig.passWord
            }
        };

        request(authenticationReq, (error: Error, response: request.Response, body: any) => {
            if (error !== undefined && error !== null) {
                this.log.error(`Get PAI token failed: ${error.message}`);
                deferred.reject(new Error(`Get PAI token failed: ${error.message}`));
            } else {
                if (response.statusCode !== 200) {
                    this.log.error(`Get PAI token failed: get PAI Rest return code ${response.statusCode}`);
                    deferred.reject(new Error(`Get PAI token failed: ${response.body}, please check paiConfig username or password`));
                }
                this.paiToken = body.token;
                this.paiTokenUpdateTime = new Date().getTime();
                deferred.resolve();
            }
        });

        let timeoutId: NodeJS.Timer;
        const timeoutDelay: Promise<void> = new Promise<void>((_resolve: Function, reject: Function): void => {
            // Set timeout and reject the promise once reach timeout (5 seconds)
            timeoutId = setTimeout(
                () => reject(new Error('Get PAI token timeout. Please check your PAI cluster.')),
                5000);
        });

        return Promise.race([timeoutDelay, deferred.promise])
            .finally(() => { clearTimeout(timeoutId); });
    }
}

export { PAITrainingService };
