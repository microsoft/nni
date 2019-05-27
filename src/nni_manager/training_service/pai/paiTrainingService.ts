
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

import * as cpp from 'child-process-promise';
import * as fs from 'fs';
import * as path from 'path';
import * as request from 'request';
import * as component from '../../common/component';

import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';
import { String } from 'typescript-string-operations';
import { MethodNotImplementedError } from '../../common/errors';
import { getExperimentId, getInitTrialSequenceId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import {
    JobApplicationForm, NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric
} from '../../common/trainingService';
import { delay, generateParamFileName,
    getExperimentRootDir, getIPV4Address, getVersion, uniqueString } from '../../common/utils';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { validateCodeDir, execMkdir } from '../common/util';
import { unixPathJoin } from '../../common/utils'
import { HDFSClientUtility } from './hdfsClientUtility';
import { NNIPAITrialConfig, PAIClusterConfig, PAIJobConfig, PAITaskRole } from './paiConfig';
import { PAI_LOG_PATH_FORMAT, PAI_OUTPUT_DIR_FORMAT, PAI_TRIAL_COMMAND_FORMAT, PAITrialJobDetail } from './paiData';
import { PAIJobInfoCollector } from './paiJobInfoCollector';
import { PAIJobRestServer } from './paiJobRestServer';

const WebHDFS = require('webhdfs');

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
    private jobQueue: string[];
    private stopping: boolean = false;
    private hdfsClient: any;
    private paiToken? : string;
    private paiTokenUpdateTime?: number;
    private paiTokenUpdateInterval: number;
    private experimentId! : string;
    private readonly paiJobCollector : PAIJobInfoCollector;
    private readonly hdfsDirPattern: string;
    private hdfsBaseDir: string | undefined;
    private hdfsOutputHost: string | undefined;
    private nextTrialSequenceId: number;
    private paiRestServerPort?: number;
    private nniManagerIpConfig?: NNIManagerIpConfig;
    private copyExpCodeDirPromise?: Promise<void>;
    private versionCheck: boolean = true;
    private logCollection: string;

    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, PAITrialJobDetail>();
        this.jobQueue = [];
        // Root dir on HDFS
        this.expRootDir = path.join('/nni', 'experiments', getExperimentId());
        this.experimentId = getExperimentId();
        this.paiJobCollector = new PAIJobInfoCollector(this.trialJobsMap);
        this.hdfsDirPattern = 'hdfs://(?<host>([0-9]{1,3}.){3}[0-9]{1,3})(:[0-9]{2,5})?(?<baseDir>/.*)?';
        this.nextTrialSequenceId = -1;
        this.paiTokenUpdateInterval = 7200000; //2hours
        this.logCollection = 'none';
        this.log.info('Construct OpenPAI training service.');
    }

    public async run(): Promise<void> {
        this.log.info('Run PAI training service.');
        const restServer: PAIJobRestServer = component.get(PAIJobRestServer);
        await restServer.start();
        restServer.setEnableVersionCheck = this.versionCheck;
        this.log.info(`PAI Training service rest server listening on: ${restServer.endPoint}`);
        await Promise.all([
            this.statusCheckingLoop(),
            this.submitJobLoop()]);
        this.log.info('PAI training service exit.');
    }

    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        const jobs: TrialJobDetail[] = [];

        for (const [key, value] of this.trialJobsMap) {
            if (value.form.jobType === 'TRIAL') {
                jobs.push(await this.getTrialJob(key));
            }
        }

        return Promise.resolve(jobs);
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        if (!this.paiClusterConfig) {
            throw new Error('PAI Cluster config is not initialized');
        }

        const paiTrialJob: PAITrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (!paiTrialJob) {
            return Promise.reject(`trial job ${trialJobId} not found`);
        }

        return Promise.resolve(paiTrialJob);
    }

    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.on('metric', listener);
    }

    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.off('metric', listener);
    }

    public async submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        const deferred : Deferred<PAITrialJobDetail> = new Deferred<PAITrialJobDetail>();
        if (!this.hdfsBaseDir) {
            throw new Error('hdfsBaseDir is not initialized');
        }

        this.log.info(`submitTrialJob: form: ${JSON.stringify(form)}`);

        const trialJobId: string = uniqueString(5);
        const trialSequenceId: number = this.generateSequenceId();
        //TODO: use HDFS working folder instead
        const trialWorkingFolder: string = path.join(this.expRootDir, 'trials', trialJobId);
        const paiJobName: string = `nni_exp_${this.experimentId}_trial_${trialJobId}`;

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
            trialSequenceId,
            hdfsLogPath);

        this.trialJobsMap.set(trialJobId, trialJobDetail);
        this.jobQueue.push(trialJobId);
        deferred.resolve(trialJobDetail);

        return deferred.promise;
    }

    public updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<TrialJobDetail> {
        throw new MethodNotImplementedError();
    }

    public get isMultiPhaseJobSupported(): boolean {
        return false;
    }

    public cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trialJobDetail : PAITrialJobDetail | undefined =  this.trialJobsMap.get(trialJobId);
        const deferred : Deferred<void> = new Deferred<void>();
        if (!trialJobDetail) {
            this.log.error(`cancelTrialJob: trial job id ${trialJobId} not found`);

            return Promise.reject();
        }

        if (!this.paiClusterConfig) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (!this.paiToken) {
            throw new Error('PAI token is not initialized');
        }

        const stopJobRequest: request.Options = {
            uri: `http://${this.paiClusterConfig.host}/rest-server/api/v1/user/${this.paiClusterConfig.userName}/jobs/${trialJobDetail.paiJobName}/executionType`,
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
            if (error || response.statusCode >= 400) {
                this.log.error(`PAI Training service: stop trial ${trialJobId} to PAI Cluster failed!`);
                deferred.reject(error ? error.message : `Stop trial failed, http code: ${response.statusCode}`);
            } else {
                deferred.resolve();
            }
        });

        return deferred.promise;
    }

    // tslint:disable-next-line:max-func-body-length
    public async setClusterMetadata(key: string, value: string): Promise<void> {
        const deferred : Deferred<void> = new Deferred<void>();

        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                deferred.resolve();
                break;

            case TrialConfigMetadataKey.PAI_CLUSTER_CONFIG:
                this.paiClusterConfig = <PAIClusterConfig>JSON.parse(value);

                this.hdfsClient = WebHDFS.createClient({
                    user: this.paiClusterConfig.userName,
                    // Refer PAI document for Pylon mapping https://github.com/Microsoft/pai/tree/master/docs/pylon
                    port: 80,
                    path: '/webhdfs/api/v1',
                    host: this.paiClusterConfig.host
                });

                // Get PAI authentication token
                await this.updatePaiToken();
                deferred.resolve();
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG:
                if (!this.paiClusterConfig) {
                    this.log.error('pai cluster config is not initialized');
                    deferred.reject(new Error('pai cluster config is not initialized'));
                    break;
                }
                this.paiTrialConfig = <NNIPAITrialConfig>JSON.parse(value);
                //paiTrialConfig.outputDir could be null if it is not set in nnictl
                if (this.paiTrialConfig.outputDir === undefined || this.paiTrialConfig.outputDir === null){
                    this.paiTrialConfig.outputDir = String.Format(
                        PAI_OUTPUT_DIR_FORMAT,
                        this.paiClusterConfig.host
                    ).replace(/\r\n|\n|\r/gm, '');
                }

                // Validate to make sure codeDir doesn't have too many files
                try {
                    await validateCodeDir(this.paiTrialConfig.codeDir);
                } catch(error) {
                    this.log.error(error);
                    deferred.reject(new Error(error));
                    break;
                }

                const hdfsDirContent = this.paiTrialConfig.outputDir.match(this.hdfsDirPattern);

                if (hdfsDirContent === null) {
                    throw new Error('Trial outputDir format Error');
                }
                const groups = hdfsDirContent.groups;
                if (groups === undefined) {
                    throw new Error('Trial outputDir format Error');
                }

                this.hdfsOutputHost = groups['host'];
                //TODO: choose to use /${username} as baseDir
                this.hdfsBaseDir = groups['baseDir'];
                if(this.hdfsBaseDir === undefined) {
                    this.hdfsBaseDir = '/';
                }

                let dataOutputHdfsClient;
                if (this.paiClusterConfig.host === this.hdfsOutputHost && this.hdfsClient) {
                    dataOutputHdfsClient = this.hdfsClient;
                } else {
                    dataOutputHdfsClient = WebHDFS.createClient({
                        user: this.paiClusterConfig.userName,
                        port: 50070,
                        host: this.hdfsOutputHost
                    });
                }

                try {
                    const exist : boolean = await HDFSClientUtility.pathExists('/', dataOutputHdfsClient);
                    if (!exist) {
                        deferred.reject(new Error(`Please check hdfsOutputDir host!`));
                    }
                } catch (error) {
                    deferred.reject(new Error(`HDFS encounters problem, error is ${error}. Please check hdfsOutputDir host!`));
                }

                // Copy experiment files from local folder to HDFS
                this.copyExpCodeDirPromise = HDFSClientUtility.copyDirectoryToHdfs(
                    this.paiTrialConfig.codeDir,
                    HDFSClientUtility.getHdfsExpCodeDir(this.paiClusterConfig.userName),
                    this.hdfsClient
                );

                deferred.resolve();
                break;
            case TrialConfigMetadataKey.VERSION_CHECK:
                this.versionCheck = (value === 'true' || value === 'True');
                break;
            case TrialConfigMetadataKey.LOG_COLLECTION:
                this.logCollection = value;
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
        this.log.info('Stopping PAI training service...');
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

    // tslint:disable-next-line:max-func-body-length
    private async submitTrialJobToPAI(trialJobId: string): Promise<boolean> {
        const deferred : Deferred<boolean> = new Deferred<boolean>();
        const trialJobDetail: PAITrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (!trialJobDetail) {
            throw new Error(`Failed to find PAITrialJobDetail for job ${trialJobId}`);
        }

        if (!this.paiClusterConfig) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (!this.paiTrialConfig) {
            throw new Error('trial config is not initialized');
        }
        if (!this.paiToken) {
            throw new Error('PAI token is not initialized');
        }

        if (!this.hdfsBaseDir) {
            throw new Error('hdfsBaseDir is not initialized');
        }

        if (!this.hdfsOutputHost) {
            throw new Error('hdfsOutputHost is not initialized');
        }

        if (!this.paiRestServerPort) {
            const restServer: PAIJobRestServer = component.get(PAIJobRestServer);
            this.paiRestServerPort = restServer.clusterRestServerPort;
        }

        // Make sure experiment code files is copied from local to HDFS
        if (this.copyExpCodeDirPromise) {
            await this.copyExpCodeDirPromise;
        }

        // Step 1. Prepare PAI job configuration
        const hdfsOutputDir : string = unixPathJoin(this.hdfsBaseDir, this.experimentId, trialJobId);
        const hdfsCodeDir: string = HDFSClientUtility.getHdfsTrialWorkDir(this.paiClusterConfig.userName, trialJobId);

        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        //create tmp trial working folder locally.
        await execMkdir(trialLocalTempFolder);

        const runScriptContent : string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });

        // Write file content ( parameter.cfg ) to local tmp folders
        const trialForm : TrialJobApplicationForm = (<TrialJobApplicationForm>trialJobDetail.form);
        if (trialForm) {
            await fs.promises.writeFile(
                path.join(trialLocalTempFolder, generateParamFileName(trialForm.hyperParameters)),
                trialForm.hyperParameters.value, { encoding: 'utf8' }
            );
        }

        const nniManagerIp: string = this.nniManagerIpConfig ? this.nniManagerIpConfig.nniManagerIp : getIPV4Address();
        const version: string = this.versionCheck ? await getVersion() : '';
        const nniPaiTrialCommand : string = String.Format(
            PAI_TRIAL_COMMAND_FORMAT,
            // PAI will copy job's codeDir into /root directory
            `$PWD/${trialJobId}`,
            `$PWD/${trialJobId}/nnioutput`,
            trialJobId,
            this.experimentId,
            trialJobDetail.sequenceId,
            this.paiTrialConfig.command,
            nniManagerIp,
            this.paiRestServerPort,
            hdfsOutputDir,
            this.hdfsOutputHost,
            this.paiClusterConfig.userName,
            HDFSClientUtility.getHdfsExpCodeDir(this.paiClusterConfig.userName),
            version,
            this.logCollection
        ).replace(/\r\n|\n|\r/gm, '');

        console.log(`nniPAItrial command is ${nniPaiTrialCommand.trim()}`);
        const paiTaskRoles : PAITaskRole[] = [
            new PAITaskRole(
                `nni_trail_${trialJobId}`,
                // Task role number
                1,
                // Task CPU number
                this.paiTrialConfig.cpuNum,
                // Task memory
                this.paiTrialConfig.memoryMB,
                // Task GPU number
                this.paiTrialConfig.gpuNum,
                // Task command
                nniPaiTrialCommand,
                // Task shared memory
                this.paiTrialConfig.shmMB
            )
        ];

        const paiJobConfig : PAIJobConfig = new PAIJobConfig(
            // Job name
            trialJobDetail.paiJobName,
            // Docker image
            this.paiTrialConfig.image,
            // dataDir
            this.paiTrialConfig.dataDir,
            // outputDir
            this.paiTrialConfig.outputDir,
            // codeDir
            `$PAI_DEFAULT_FS_URI${hdfsCodeDir}`,
            // PAI Task roles
            paiTaskRoles,
            // Add Virutal Cluster
            this.paiTrialConfig.virtualCluster === undefined ? 'default' : this.paiTrialConfig.virtualCluster.toString()
        );

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
            uri: `http://${this.paiClusterConfig.host}/rest-server/api/v1/user/${this.paiClusterConfig.userName}/jobs`,
            method: 'POST',
            json: true,
            body: paiJobConfig,
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${this.paiToken}`
            }
        };
        request(submitJobRequest, (error: Error, response: request.Response, body: any) => {
            if (error || response.statusCode >= 400) {
                const errorMessage : string = error ? error.message :
                    `Submit trial ${trialJobId} failed, http code:${response.statusCode}, http body: ${response.body}`;
                this.log.error(errorMessage);
                trialJobDetail.status = 'FAILED';
                deferred.reject(new Error(errorMessage));
            } else {
                trialJobDetail.submitTime = Date.now();
                deferred.resolve(true);
            }
        });

        return deferred.promise;
    }

    private generateSequenceId(): number {
        if (this.nextTrialSequenceId === -1) {
            this.nextTrialSequenceId = getInitTrialSequenceId();
        }

        return this.nextTrialSequenceId++;
    }

    private async statusCheckingLoop(): Promise<void> {
        while (!this.stopping) {
            try{
                await this.updatePaiToken();
            }catch(error){
                this.log.error(`${error}`);
                //only throw error when initlize paiToken first time
                if(!this.paiToken) {
                    throw new Error(error);
                }
            }
            await this.paiJobCollector.retrieveTrialStatus(this.paiToken, this.paiClusterConfig);
            const restServer: PAIJobRestServer = component.get(PAIJobRestServer);
            if (restServer.getErrorMessage) {
                throw new Error(restServer.getErrorMessage);
            }
            await delay(3000);
        }
    }

    private async submitJobLoop(): Promise<void> {
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

    /**
     * Update pai token by the interval time or initialize the pai token
     */
    private async updatePaiToken(): Promise<void> {
        const deferred : Deferred<void> = new Deferred<void>();

        const currentTime: number = new Date().getTime();
        //If pai token initialized and not reach the interval time, do not update
        if (this.paiTokenUpdateTime && (currentTime - this.paiTokenUpdateTime) < this.paiTokenUpdateInterval){
            return Promise.resolve();
        }

        if (!this.paiClusterConfig) {
            const paiClusterConfigError: string = `pai cluster config not initialized!`;
            this.log.error(`${paiClusterConfigError}`);
            throw Error(`${paiClusterConfigError}`);
        }

        const authentication_req: request.Options = {
            uri: `http://${this.paiClusterConfig.host}/rest-server/api/v1/token`,
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
                if (response.statusCode !== 200){
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
            .finally(() => clearTimeout(timeoutId));
    }
}

export { PAITrainingService };
