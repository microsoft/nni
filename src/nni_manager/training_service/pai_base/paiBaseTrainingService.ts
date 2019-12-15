// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as path from 'path';
import * as request from 'request';
import * as component from '../../common/component';

import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';
import { String } from 'typescript-string-operations';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import {
    HyperParameters, NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric
} from '../../common/trainingService';
import { delay, generateParamFileName,
    getExperimentRootDir, getIPV4Address, getVersion, uniqueString, unixPathJoin } from '../../common/utils';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { execMkdir, validateCodeDir } from '../common/util';
import { PAIBaseJobInfoCollector } from './paiBaseJobInfoCollector';
import { PAIBaseJobRestServer, ParameterFileMeta } from './paiBaseJobRestServer';
import { PAIBaseClusterConfig, PAIBaseTrialJobDetail } from './paiBaseConfig';

/**
 * Training Service implementation for OpenPAI (Open Platform for AI)
 * Refer https://github.com/Microsoft/pai for more info about OpenPAI
 */
@component.Singleton
class PAIBaseTrainingService implements TrainingService {
    protected readonly log!: Logger;
    protected readonly metricsEmitter: EventEmitter;
    protected readonly trialJobsMap: Map<string, PAIBaseTrialJobDetail>;
    protected readonly expRootDir: string;
    protected paiBaseClusterConfig?: PAIBaseClusterConfig;
    protected readonly jobQueue: string[];
    protected stopping: boolean = false;
    protected paiToken? : string;
    protected paiTokenUpdateTime?: number;
    protected readonly paiTokenUpdateInterval: number;
    protected readonly experimentId!: string;
    protected readonly paiJobCollector: PAIBaseJobInfoCollector;
    protected paiRestServerPort?: number;
    protected nniManagerIpConfig?: NNIManagerIpConfig;
    protected versionCheck: boolean = true;
    protected logCollection: string;
    protected isMultiPhase: boolean = false;
    protected authFileHdfsPath: string | undefined = undefined;
    protected portList?: string | undefined;

    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, PAIBaseTrialJobDetail>();
        this.jobQueue = [];
        this.expRootDir = path.join('/nni', 'experiments', getExperimentId());
        this.experimentId = getExperimentId();
        this.paiJobCollector = new PAIBaseJobInfoCollector(this.trialJobsMap);
        this.paiTokenUpdateInterval = 7200000; //2hours
        this.logCollection = 'none';
        this.log.info('Construct paiBase training service.');
    }

    public async run(): Promise<void> {
        return;
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<any> {
        return null;
    }

    public async updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const trialJobDetail: undefined | TrialJobDetail = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }

        return trialJobDetail;
    }

    protected async submitTrialJobToPAI(trialJobId: string): Promise<boolean> {
        return true;
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

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        return;
    }

    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        const jobs: TrialJobDetail[] = [];

        for (const [key, value] of this.trialJobsMap) {
            jobs.push(await this.getTrialJob(key));
        }

        return Promise.resolve(jobs);
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        if (this.paiBaseClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }

        const paiBaseTrialJob: PAIBaseTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (paiBaseTrialJob === undefined) {
            return Promise.reject(`trial job ${trialJobId} not found`);
        }

        return Promise.resolve(paiBaseTrialJob);
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
        const trialJobDetail: PAIBaseTrialJobDetail | undefined =  this.trialJobsMap.get(trialJobId);
        const deferred: Deferred<void> = new Deferred<void>();
        if (trialJobDetail === undefined) {
            this.log.error(`cancelTrialJob: trial job id ${trialJobId} not found`);

            return Promise.reject();
        }

        if (this.paiBaseClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (this.paiToken === undefined) {
            throw new Error('PAI token is not initialized');
        }

        const stopJobRequest: request.Options = {
            uri: `http://${this.paiBaseClusterConfig.host}/rest-server/api/v1/user/${this.paiBaseClusterConfig.userName}\
/jobs/${trialJobDetail.paiJobName}/executionType`, 
            method: 'PUT',
            json: true,
            body: {value: 'STOP'},
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${this.paiToken}`
            }
        };

        // Set trialjobDetail's early stopped field, to mark the job's cancellation source
        trialJobDetail.isEarlyStopped = isEarlyStopped;

        request(stopJobRequest, (error: Error, response: request.Response, body: any) => {
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

    public getClusterMetadata(key: string): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();

        deferred.resolve();

        return deferred.promise;
    }

    public async cleanUp(): Promise<void> {
        this.log.info('Stopping PAI training service...');
        this.stopping = true;

        const deferred: Deferred<void> = new Deferred<void>();
        const restServer: PAIBaseJobRestServer = component.get(PAIBaseJobRestServer);
        try {
            await restServer.stop();
            deferred.resolve();
            this.log.info('PAI Training service rest server stopped successfully.');
        } catch (error) {
            this.log.error(`PAI Training service rest server stopped failed, error: ${error.message}`);
            deferred.reject(error);
        }

        return deferred.promise;
    }

    public get MetricsEmitter(): EventEmitter {
        return this.metricsEmitter;
    }

    protected async statusCheckingLoop(): Promise<void> {
        while (!this.stopping) {
            if(this.paiBaseClusterConfig && this.paiBaseClusterConfig.passWord) {
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
            await this.paiJobCollector.retrieveTrialStatus(this.paiToken, this.paiBaseClusterConfig);
            const restServer: PAIBaseJobRestServer = component.get(PAIBaseJobRestServer);
            if (restServer.getErrorMessage !== undefined) {
                throw new Error(restServer.getErrorMessage);
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

        if (this.paiBaseClusterConfig === undefined) {
            const paiClusterConfigError: string = `pai cluster config not initialized!`;
            this.log.error(`${paiClusterConfigError}`);
            throw Error(`${paiClusterConfigError}`);
        }

        const authenticationReq: request.Options = {
            uri: `http://${this.paiBaseClusterConfig.host}/rest-server/api/v1/token`,
            method: 'POST',
            json: true,
            body: {
                username: this.paiBaseClusterConfig.userName,
                password: this.paiBaseClusterConfig.passWord
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
        const timeoutDelay: Promise<void> = new Promise<void>((resolve: Function, reject: Function): void => {
            // Set timeout and reject the promise once reach timeout (5 seconds)
            timeoutId = setTimeout(
                () => reject(new Error('Get PAI token timeout. Please check your PAI cluster.')),
                5000);
        });

        return Promise.race([timeoutDelay, deferred.promise])
            .finally(() => { clearTimeout(timeoutId); });
    }
}

export { PAIBaseTrainingService };
