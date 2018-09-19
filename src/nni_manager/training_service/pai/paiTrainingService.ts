
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

'use strict'

import * as assert from 'assert';
import * as component from '../../common/component';
import * as path from 'path';
import * as request from 'request';

import { Deferred } from 'ts-deferred';
import { EventEmitter } from 'events';
import { MethodNotImplementedError, NNIError, NNIErrorNames } from '../../common/errors';
import { getLogger, Logger } from '../../common/log';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import {
    HostJobApplicationForm, JobApplicationForm, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric, TrialJobStatus
} from '../../common/trainingService';
import { delay, getExperimentRootDir, uniqueString } from '../../common/utils';
import { ObservableTimer } from '../../common/observableTimer';
import { PAIJobRestServer } from './paiJobRestServer'
import { PAITrialJobDetail, PAI_TRIAL_COMMAND_FORMAT } from './paiData';
import { String } from 'typescript-string-operations';
import { NNIPAITrialConfig, PAIClusterConfig, PAIJobConfig, PAITaskRole } from './paiConfig';
import { HDFSClientUtility } from './hdfsClientUtility'
import { getExperimentId } from '../../common/experimentStartupInfo';


var WebHDFS = require('webhdfs');

/**
 * Training Service implementation for OpenPAI (Open Platform for AI)
 * Refer https://github.com/Microsoft/pai for more info about OpenPAI
 */
@component.Singleton
class PAITrainingService implements TrainingService {
    private readonly log!: Logger;
    private readonly metricsEmitter: EventEmitter;
    private readonly trialJobsMap: Map<string, PAITrialJobDetail>;
    private readonly expRootDir: string;
    private paiTrialConfig: NNIPAITrialConfig | undefined;
    private paiClusterConfig?: PAIClusterConfig;
    private stopping: boolean = false;
    private hdfsClient: any;
    private paiToken? : string;
    private experimentId! : string;

    constructor(@component.Inject timer: ObservableTimer) {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, PAITrialJobDetail>();
        // Root dir on HDFS
        this.expRootDir = path.join('/nni', 'experiments', getExperimentId());
        this.experimentId = getExperimentId();        
    }

    public async run(): Promise<void> {
        const restServer: PAIJobRestServer = component.get(PAIJobRestServer);
        await restServer.start();
        this.log.info(`PAI Training service rest server listening on: ${restServer.endPoint}`);
    }

    public listTrialJobs(): Promise<TrialJobDetail[]> {
        const deferred : Deferred<TrialJobDetail[]> = new Deferred<TrialJobDetail[]>();

        deferred.resolve([]);
        return deferred.promise; 
    }

    public getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        const deferred : Deferred<TrialJobDetail> = new Deferred<TrialJobDetail>();

        deferred.resolve(undefined);
        return deferred.promise; 
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void) {
        this.metricsEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void) {
        this.metricsEmitter.off('metric', listener);
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        const deferred : Deferred<PAITrialJobDetail> = new Deferred<PAITrialJobDetail>();
        if(!this.paiClusterConfig) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (!this.paiTrialConfig) {
            throw new Error('trial config is not initialized');
        }
        if (!this.paiToken) {
            throw new Error('PAI token is not initialized');
        }

        this.log.info(`submitTrialJob: form: ${JSON.stringify(form)}`);

        const trialJobId: string = uniqueString(5);
        //TODO: use HDFS working folder instead
        const trialWorkingFolder: string = path.join(this.expRootDir, 'trials', trialJobId);

        const trialJobDetail: PAITrialJobDetail = new PAITrialJobDetail(
            trialJobId,
            'WAITING',
            Date.now(),
            trialWorkingFolder,
            form);
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        // Step 1. Prepare PAI job configuration
        const paiJobName : string = `nni_exp_${this.experimentId}_trial_${trialJobId}`;
        const hdfsCodeDir : string = path.join(this.expRootDir, trialJobId);

        const nniPaiTrialCommand : string = String.Format(
            PAI_TRIAL_COMMAND_FORMAT,
            `./${trialJobId}`,
            trialJobId,
            this.paiTrialConfig.command
        ).replace(/\r\n|\n|\r/gm, '');

        console.log(`nniPAItrial command is ${nniPaiTrialCommand.trim()}`);
        const paiTaskRoles : PAITaskRole[] = [new PAITaskRole('nni_trail_' + trialJobId, 
                                    // Task role number
                                    1, 
                                    // Task CPU number
                                    this.paiTrialConfig.cpuNum, 
                                    // Task memory
                                    this.paiTrialConfig.memoryMB, 
                                    // Task GPU number
                                    this.paiTrialConfig.gpuNum, 
                                    // Task command
                                    nniPaiTrialCommand)];

        const paiJobConfig : PAIJobConfig = new PAIJobConfig(
                                    // Job name
                                    paiJobName, 
                                    // Docker image
                                    this.paiTrialConfig.image, 
                                    // dataDir
                                    this.paiTrialConfig.dataDir, 
                                    // outputDir
                                    this.paiTrialConfig.outputDir, 
                                    // codeDir
                                    `$PAI_DEFAULT_FS_URI${hdfsCodeDir}`, 
                                    // TODO: Add Virutal Cluster
                                    // PAI Task roles
                                    paiTaskRoles);
        console.log(`PAI job config is ${JSON.stringify(paiJobConfig)}`);

        // Step 2. Upload code files in codeDir onto HDFS
        try {
            await HDFSClientUtility.copyDirectoryToHdfs(this.paiTrialConfig.codeDir, hdfsCodeDir, this.hdfsClient);
        } catch (error) {
            this.log.error(`PAI Training service: copy ${this.paiTrialConfig.codeDir} to HDFS ${hdfsCodeDir} failed, error is ${error}`);
            throw new Error(error.message);
        }

        // Step 3. Submit PAI job via Rest call
        // Refer https://github.com/Microsoft/pai/blob/master/docs/rest-server/API.md for more detail about PAI Rest API
        const submitJobRequest: request.Options = {
            uri: `http://${this.paiClusterConfig.host}:9186/api/v1/jobs`,
            method: 'POST',
            json: true,
            body: paiJobConfig,
            headers: {
                "Content-Type": "application/json",
                "Authorization": 'Bearer ' + this.paiToken
            }
        };
        request(submitJobRequest, (error: Error, response: request.Response, body: any) => {
            if (error || response.statusCode >= 400) {
                this.log.error(`PAI Training service: Submit trial ${trialJobId} to PAI Cluster failed!`);
                trialJobDetail.status = 'FAILED';
                deferred.reject(error ? error.message : 'Submit trial failed, http code: ' + response.statusCode);                
            } else {
                deferred.resolve(trialJobDetail);
            }
        });

        return deferred.promise;
    }

    public updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<TrialJobDetail> {
        throw new MethodNotImplementedError();
    }

    public get isMultiPhaseJobSupported(): boolean {
        return false;
    }

    public cancelTrialJob(trialJobId: string): Promise<void> {
        this.log.info(`PAI Training service cancelTrialJob: jobId: ${trialJobId}`);
        const deferred : Deferred<void> = new Deferred<void>();

        deferred.resolve();
        return deferred.promise; 
    }

    public setClusterMetadata(key: string, value: string): Promise<void> {
        const deferred : Deferred<void> = new Deferred<void>();

        switch (key) {
            case TrialConfigMetadataKey.PAI_CLUSTER_CONFIG:
                //TODO: try catch exception when setting up HDFS client and get PAI token
                this.paiClusterConfig = <PAIClusterConfig>JSON.parse(value);
                
                this.hdfsClient = WebHDFS.createClient({
                    user: this.paiClusterConfig.userName,
                    port: 50070,
                    host: this.paiClusterConfig.host
                });

                // Get PAI authentication token
                const authentication_req: request.Options = {
                    uri: `http://${this.paiClusterConfig.host}:9186/api/v1/token`,
                    method: 'POST',
                    json: true,
                    body: {
                        username: this.paiClusterConfig.userName,
                        password: this.paiClusterConfig.passWord
                    }
                };

                request(authentication_req, (error: Error, response: request.Response, body: any) => {
                    if (error) {
                        //TODO: should me make the setClusterMetadata's return type to Promise<string>? 
                        this.log.error(`Get PAI token failed: ${error.message}`);
                        deferred.reject();
                    } else {
                        if(response.statusCode !== 200){
                            this.log.error(`Get PAI token failed: get PAI Rest return code ${response.statusCode}`);
                            deferred.reject();
                        }
                        this.paiToken = body.token;

                        console.log(`Got token ${this.paiToken} from PAI Cluster`);
                        deferred.resolve();
                    }
                });
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                if (!this.paiClusterConfig){
                    this.log.error('pai cluster config is not initialized');
                    deferred.reject();
                    break;
                }
                this.paiTrialConfig = <NNIPAITrialConfig>JSON.parse(value);
                console.log(`Set Cluster metadata: paiTrialConfig is ${JSON.stringify(this.paiTrialConfig)}`);
                deferred.resolve();
                break;
            default:
                //Reject for unknown keys
                throw new Error(`Uknown key: ${key}`);
        }

        return deferred.promise; 
    }

    public getClusterMetadata(key: string): Promise<string> {
        const deferred : Deferred<string> = new Deferred<string>();

        deferred.resolve();
        return deferred.promise; 
    }

    public cleanUp(): Promise<void> {
        const deferred : Deferred<void> = new Deferred<void>();

        deferred.resolve();
        return deferred.promise; 
    }

    
    public get MetricsEmitter() : EventEmitter {
        return this.metricsEmitter;
    }
}

export { PAITrainingService }