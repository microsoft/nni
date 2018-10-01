
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

import * as component from '../../common/component';
import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as path from 'path';
import * as request from 'request';

import { Deferred } from 'ts-deferred';
import { EventEmitter } from 'events';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { HDFSClientUtility } from './hdfsClientUtility'
import { MethodNotImplementedError } from '../../common/errors';
import { getLogger, Logger } from '../../common/log';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import {
    JobApplicationForm, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric
} from '../../common/trainingService';
import { delay, getExperimentRootDir, getIPV4Address, uniqueString } from '../../common/utils';
import { PAIJobRestServer } from './paiJobRestServer'
import { PAITrialJobDetail, PAI_INSTALL_NNI_SHELL_FORMAT, PAI_TRIAL_COMMAND_FORMAT, PAI_OUTPUT_DIR_FORMAT, PAI_LOG_PATH_FORMAT } from './paiData';
import { PAIJobInfoCollector } from './paiJobInfoCollector';
import { String } from 'typescript-string-operations';
import { NNIPAITrialConfig, PAIClusterConfig, PAIJobConfig, PAITaskRole } from './paiConfig';

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
    private readonly paiJobCollector : PAIJobInfoCollector;
    private readonly hdfsDirPattern: string;
    private hdfsBaseDir: string | undefined;
    private hdfsOutputHost: string | undefined;

    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, PAITrialJobDetail>();
        // Root dir on HDFS
        this.expRootDir = path.join('/nni', 'experiments', getExperimentId());
        this.experimentId = getExperimentId();      
        this.paiJobCollector = new PAIJobInfoCollector(this.trialJobsMap);
        this.hdfsDirPattern = 'hdfs://(?<host>([0-9]{1,3}.){3}[0-9]{1,3})(:[0-9]{2,5})?(?<baseDir>/.*)?';
    }

    public async run(): Promise<void> {
        const restServer: PAIJobRestServer = component.get(PAIJobRestServer);
        await restServer.start();
        this.log.info(`PAI Training service rest server listening on: ${restServer.endPoint}`);
        while (!this.stopping) {
            await this.paiJobCollector.updateTrialStatusFromPAI(this.paiToken, this.paiClusterConfig);
            await delay(3000);
        }
    }

    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        const jobs: TrialJobDetail[] = [];
        
        this.trialJobsMap.forEach(async (value: PAITrialJobDetail, key: string) => {
            if (value.form.jobType === 'TRIAL') {
                jobs.push(await this.getTrialJob(key));
            }
        });

        return Promise.resolve(jobs);
    }

    public getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        if(!this.paiClusterConfig) {
            throw new Error('PAI Cluster config is not initialized');
        }

        const paiTrialJob: PAITrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (!paiTrialJob) {
            return Promise.reject(`trial job ${trialJobId} not found`)
        }        

        return Promise.resolve(paiTrialJob);
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
        
        if(!this.hdfsBaseDir){
            throw new Error('hdfsBaseDir is not initialized');
        }

        if(!this.hdfsOutputHost){
            throw new Error('hdfsOutputHost is not initialized');
        }

        this.log.info(`submitTrialJob: form: ${JSON.stringify(form)}`);

        const trialJobId: string = uniqueString(5);
        //TODO: use HDFS working folder instead
        const trialWorkingFolder: string = path.join(this.expRootDir, 'trials', trialJobId);
        
        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        //create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${path.dirname(trialLocalTempFolder)}`);
        await cpp.exec(`cp -r ${this.paiTrialConfig.codeDir} ${trialLocalTempFolder}`);

        const runScriptContent : string = PAI_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });
        
        // Write file content ( parameter.cfg ) to local tmp folders
        const trialForm : TrialJobApplicationForm = (<TrialJobApplicationForm>form)
        if(trialForm) {
            await fs.promises.writeFile(path.join(trialLocalTempFolder, 'parameter.cfg'), trialForm.hyperParameters, { encoding: 'utf8' });
        }
        
        // Step 1. Prepare PAI job configuration
        const paiJobName : string = `nni_exp_${this.experimentId}_trial_${trialJobId}`;
        const hdfsCodeDir : string = path.join(this.expRootDir, trialJobId);
        
        const hdfsOutputDir : string = path.join(this.hdfsBaseDir, this.experimentId, trialJobId);
        const hdfsLogPath : string = String.Format(
            PAI_LOG_PATH_FORMAT,
            this.hdfsOutputHost,
            hdfsOutputDir);

        const trialJobDetail: PAITrialJobDetail = new PAITrialJobDetail(
            trialJobId,
            'WAITING',
            paiJobName,            
            Date.now(),
            trialWorkingFolder,
            form, 
            hdfsLogPath);
        this.trialJobsMap.set(trialJobId, trialJobDetail);

        const nniPaiTrialCommand : string = String.Format(
            PAI_TRIAL_COMMAND_FORMAT,
            // PAI will copy job's codeDir into /root directory
            `/root/${trialJobId}`,
            `/root/${trialJobId}/nnioutput`,
            trialJobId,
            this.experimentId,
            this.paiTrialConfig.command, 
            getIPV4Address(),
            hdfsOutputDir,
            this.hdfsOutputHost,
            this.paiClusterConfig.userName
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

        // Step 2. Upload code files in codeDir onto HDFS
        try {
            await HDFSClientUtility.copyDirectoryToHdfs(trialLocalTempFolder, hdfsCodeDir, this.hdfsClient);
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
                trialJobDetail.submitTime = Date.now();
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
        const trialJobDetail : PAITrialJobDetail | undefined =  this.trialJobsMap.get(trialJobId);
        const deferred : Deferred<void> = new Deferred<void>();
        if(!trialJobDetail) {
            this.log.error(`cancelTrialJob: trial job id ${trialJobId} not found`);
            return Promise.reject();
        }

        if(!this.paiClusterConfig) {
            throw new Error('PAI Cluster config is not initialized');
        }        
        if (!this.paiToken) {
            throw new Error('PAI token is not initialized');
        }

        const stopJobRequest: request.Options = {
            uri: `http://${this.paiClusterConfig.host}:9186/api/v1/jobs/${trialJobDetail.paiJobName}/executionType`,
            method: 'PUT',
            json: true,
            body: {'value' : 'STOP'},
            headers: {
                "Content-Type": "application/json",
                "Authorization": 'Bearer ' + this.paiToken
            }
        };
        request(stopJobRequest, (error: Error, response: request.Response, body: any) => {
            if (error || response.statusCode >= 400) {
                this.log.error(`PAI Training service: stop trial ${trialJobId} to PAI Cluster failed!`);
                deferred.reject(error ? error.message : 'Stop trial failed, http code: ' + response.statusCode);                
            } else {
                deferred.resolve();
            }
        });

        return deferred.promise; 
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
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
                        this.log.error(`Get PAI token failed: ${error.message}`);
                        deferred.reject(new Error(`Get PAI token failed: ${error.message}`));
                    } else {
                        if(response.statusCode !== 200){
                            this.log.error(`Get PAI token failed: get PAI Rest return code ${response.statusCode}`);
                            deferred.reject(new Error(`Get PAI token failed, please check paiConfig username or password`));
                        }
                        this.paiToken = body.token;

                        deferred.resolve();
                    }
                });
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                if (!this.paiClusterConfig){
                    this.log.error('pai cluster config is not initialized');
                    deferred.reject(new Error('pai cluster config is not initialized'));
                    break;
                }
                this.paiTrialConfig = <NNIPAITrialConfig>JSON.parse(value);
                //paiTrialConfig.outputDir could be null if it is not set in nnictl
                if(this.paiTrialConfig.outputDir === undefined || this.paiTrialConfig.outputDir === null){
                    this.paiTrialConfig.outputDir = String.Format(
                        PAI_OUTPUT_DIR_FORMAT,
                        this.paiClusterConfig.host
                    ).replace(/\r\n|\n|\r/gm, '');
                }
                
                const hdfsDirContent = this.paiTrialConfig.outputDir.match(this.hdfsDirPattern);

                if(hdfsDirContent === null) {
                    throw new Error('Trial outputDir format Error');
                }
                const groups = hdfsDirContent.groups;
                if(groups === undefined) {
                    throw new Error('Trial outputDir format Error');
                }
        
                this.hdfsOutputHost = groups['host'];
                this.hdfsBaseDir = groups['baseDir'];
                if(this.hdfsBaseDir === undefined) {
                    this.hdfsBaseDir = "/";
                }
                
                const hdfsClient = WebHDFS.createClient({
                    user: this.paiClusterConfig.userName,
                    port: 50070,
                    host: this.hdfsOutputHost
                });

                try {
                    const exist : boolean = await HDFSClientUtility.pathExists("/", hdfsClient);
                    if(!exist) {
                        deferred.reject(new Error(`Please check hdfsOutputDir host!`));
                    }
                } catch(error) {
                    deferred.reject(new Error(`HDFS encounters problem, error is ${error}. Please check hdfsOutputDir host!`));
                }

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

    public async cleanUp(): Promise<void> {
        this.stopping = true;

        const deferred : Deferred<void> = new Deferred<void>();
        const restServer: PAIJobRestServer = component.get(PAIJobRestServer);
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

    public get MetricsEmitter() : EventEmitter {
        return this.metricsEmitter;
    }
}

export { PAITrainingService }