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
import { HDFSClientUtility } from './hdfsClientUtility';
import { NNIPAITrialConfig, PAIClusterConfig, PAIJobConfig, PAITaskRole } from './paiConfig';
import { PAI_LOG_PATH_FORMAT, PAI_TRIAL_COMMAND_FORMAT, PAITrialJobDetail } from './paiData';
import { PAIJobInfoCollector } from './paiJobInfoCollector';
import { PAIJobRestServer, ParameterFileMeta } from './paiJobRestServer';

import * as WebHDFS from 'webhdfs';

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
    private readonly jobQueue: string[];
    private stopping: boolean = false;
    private hdfsClient: any;
    private paiToken? : string;
    private paiTokenUpdateTime?: number;
    private readonly paiTokenUpdateInterval: number;
    private readonly experimentId!: string;
    private readonly paiJobCollector: PAIJobInfoCollector;
    private paiRestServerPort?: number;
    private nniManagerIpConfig?: NNIManagerIpConfig;
    private copyExpCodeDirPromise?: Promise<void>;
    private copyAuthFilePromise?: Promise<void>;
    private versionCheck: boolean = true;
    private logCollection: string;
    private isMultiPhase: boolean = false;
    private authFileHdfsPath: string | undefined = undefined;
    private portList?: string | undefined;

    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, PAITrialJobDetail>();
        this.jobQueue = [];
        // Root dir on HDFS
        this.expRootDir = path.join('/nni', 'experiments', getExperimentId());
        this.experimentId = getExperimentId();
        this.paiJobCollector = new PAIJobInfoCollector(this.trialJobsMap);
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
            jobs.push(await this.getTrialJob(key));
        }

        return Promise.resolve(jobs);
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        if (this.paiClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }

        const paiTrialJob: PAITrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (paiTrialJob === undefined) {
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

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        if (this.paiClusterConfig === undefined) {
            throw new Error(`paiClusterConfig not initialized!`);
        }
        const deferred: Deferred<PAITrialJobDetail> = new Deferred<PAITrialJobDetail>();

        this.log.info(`submitTrialJob: form: ${JSON.stringify(form)}`);

        const trialJobId: string = uniqueString(5);
        //TODO: use HDFS working folder instead
        const trialWorkingFolder: string = path.join(this.expRootDir, 'trials', trialJobId);
        const paiJobName: string = `nni_exp_${this.experimentId}_trial_${trialJobId}`;
        const hdfsCodeDir: string = HDFSClientUtility.getHdfsTrialWorkDir(this.paiClusterConfig.userName, trialJobId);
        const hdfsOutputDir: string = unixPathJoin(hdfsCodeDir, 'nnioutput');

        const hdfsLogPath: string = String.Format(
            PAI_LOG_PATH_FORMAT,
            this.paiClusterConfig.host,
            hdfsOutputDir
            );

        const trialJobDetail: PAITrialJobDetail = new PAITrialJobDetail(
            trialJobId,
            'WAITING',
            paiJobName,
            Date.now(),
            trialWorkingFolder,
            form,
            hdfsLogPath);

        this.trialJobsMap.set(trialJobId, trialJobDetail);
        this.jobQueue.push(trialJobId);
        deferred.resolve(trialJobDetail);

        return deferred.promise;
    }

    public async updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const trialJobDetail: undefined | TrialJobDetail = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }
        await this.writeParameterFile(trialJobId, form.hyperParameters);

        return trialJobDetail;
    }

    public get isMultiPhaseJobSupported(): boolean {
        return true;
    }

    public cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trialJobDetail: PAITrialJobDetail | undefined =  this.trialJobsMap.get(trialJobId);
        const deferred: Deferred<void> = new Deferred<void>();
        if (trialJobDetail === undefined) {
            this.log.error(`cancelTrialJob: trial job id ${trialJobId} not found`);

            return Promise.reject();
        }

        if (this.paiClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (this.paiToken === undefined) {
            throw new Error('PAI token is not initialized');
        }

        const stopJobRequest: request.Options = {
            uri: `http://${this.paiClusterConfig.host}/rest-server/api/v1/user/${this.paiClusterConfig.userName}\
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

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();

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
                if(this.paiClusterConfig.passWord) {
                    // Get PAI authentication token
                    await this.updatePaiToken();
                } else if(this.paiClusterConfig.token) {
                    this.paiToken = this.paiClusterConfig.token;
                } else {
                    deferred.reject(new Error('pai cluster config format error, please set password or token!'));
                }

                deferred.resolve();
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG:
                if (this.paiClusterConfig === undefined) {
                    this.log.error('pai cluster config is not initialized');
                    deferred.reject(new Error('pai cluster config is not initialized'));
                    break;
                }
                this.paiTrialConfig = <NNIPAITrialConfig>JSON.parse(value);

                // Validate to make sure codeDir doesn't have too many files
                try {
                    await validateCodeDir(this.paiTrialConfig.codeDir);
                } catch (error) {
                    this.log.error(error);
                    deferred.reject(new Error(error));
                    break;
                }
           
                // Copy experiment files from local folder to HDFS
                this.copyExpCodeDirPromise = HDFSClientUtility.copyDirectoryToHdfs(
                    this.paiTrialConfig.codeDir,
                    HDFSClientUtility.getHdfsExpCodeDir(this.paiClusterConfig.userName),
                    this.hdfsClient
                );
                
                // Upload authFile to hdfs
                if (this.paiTrialConfig.authFile) {
                    this.authFileHdfsPath = unixPathJoin(HDFSClientUtility.hdfsExpRootDir(this.paiClusterConfig.userName), 'authFile');
                    this.copyAuthFilePromise = HDFSClientUtility.copyFileToHdfs(this.paiTrialConfig.authFile, this.authFileHdfsPath, this.hdfsClient);
                }

                deferred.resolve();
                break;
            case TrialConfigMetadataKey.VERSION_CHECK:
                this.versionCheck = (value === 'true' || value === 'True');
                break;
            case TrialConfigMetadataKey.LOG_COLLECTION:
                this.logCollection = value;
                break;
            case TrialConfigMetadataKey.MULTI_PHASE:
                this.isMultiPhase = (value === 'true' || value === 'True');
                break;
            default:
                //Reject for unknown keys
                throw new Error(`Uknown key: ${key}`);
        }

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

    public get MetricsEmitter(): EventEmitter {
        return this.metricsEmitter;
    }

    private async submitTrialJobToPAI(trialJobId: string): Promise<boolean> {
        const deferred: Deferred<boolean> = new Deferred<boolean>();
        const trialJobDetail: PAITrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (trialJobDetail === undefined) {
            throw new Error(`Failed to find PAITrialJobDetail for job ${trialJobId}`);
        }

        if (this.paiClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (this.paiTrialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        if (this.paiToken === undefined) {
            throw new Error('PAI token is not initialized');
        }

        if (this.paiRestServerPort === undefined) {
            const restServer: PAIJobRestServer = component.get(PAIJobRestServer);
            this.paiRestServerPort = restServer.clusterRestServerPort;
        }

        // Make sure experiment code files is copied from local to HDFS
        if (this.copyExpCodeDirPromise !== undefined) {
            await this.copyExpCodeDirPromise;
        }

        //Make sure authFile is copied from local to HDFS
        if (this.paiTrialConfig.authFile) {
            await this.copyAuthFilePromise;
        }
        // Step 1. Prepare PAI job configuration

        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        //create tmp trial working folder locally.
        await execMkdir(trialLocalTempFolder);

        const runScriptContent: string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });

        // Write file content ( parameter.cfg ) to local tmp folders
        if (trialJobDetail.form !== undefined) {
            await fs.promises.writeFile(
                path.join(trialLocalTempFolder, generateParamFileName(trialJobDetail.form.hyperParameters)),
                trialJobDetail.form.hyperParameters.value, { encoding: 'utf8' }
            );
        }
        const hdfsCodeDir: string = HDFSClientUtility.getHdfsTrialWorkDir(this.paiClusterConfig.userName, trialJobId);
        const hdfsOutputDir: string = unixPathJoin(hdfsCodeDir, 'nnioutput');
        const nniManagerIp: string = this.nniManagerIpConfig ? this.nniManagerIpConfig.nniManagerIp : getIPV4Address();
        const version: string = this.versionCheck ? await getVersion() : '';
        const nniPaiTrialCommand: string = String.Format(
            PAI_TRIAL_COMMAND_FORMAT,
            // PAI will copy job's codeDir into /root directory
            `$PWD/${trialJobId}`,
            `$PWD/${trialJobId}/nnioutput`,
            trialJobId,
            this.experimentId,
            trialJobDetail.form.sequenceId,
            this.isMultiPhase,
            this.paiTrialConfig.command,
            nniManagerIp,
            this.paiRestServerPort,
            hdfsOutputDir,
            this.paiClusterConfig.host,
            this.paiClusterConfig.userName,
            HDFSClientUtility.getHdfsExpCodeDir(this.paiClusterConfig.userName),
            version,
            this.logCollection
        )
        .replace(/\r\n|\n|\r/gm, '');

        this.log.info(`nniPAItrial command is ${nniPaiTrialCommand.trim()}`);
        const paiTaskRoles: PAITaskRole[] = [
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
                this.paiTrialConfig.shmMB,
                // Task portList
                this.paiTrialConfig.portList
            )
        ];

        const paiJobConfig: PAIJobConfig = new PAIJobConfig(
            // Job name
            trialJobDetail.paiJobName,
            // Docker image
            this.paiTrialConfig.image,
            // codeDir
            `$PAI_DEFAULT_FS_URI${hdfsCodeDir}`,
            // PAI Task roles
            paiTaskRoles,
            // Add Virutal Cluster
            this.paiTrialConfig.virtualCluster === undefined ? 'default' : this.paiTrialConfig.virtualCluster.toString(),
            //Task auth File
            this.authFileHdfsPath
        );

        // Step 2. Upload code files in codeDir onto HDFS
        try {
            await HDFSClientUtility.copyDirectoryToHdfs(trialLocalTempFolder, hdfsCodeDir, this.hdfsClient);
        } catch (error) {
            this.log.error(`PAI Training service: copy ${this.paiTrialConfig.codeDir} to HDFS ${hdfsCodeDir} failed, error is ${error}`);
            trialJobDetail.status = 'FAILED'; // eslint-disable-line require-atomic-updates
            deferred.resolve(true);

            return deferred.promise;
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
            if ((error !== undefined && error !== null) || response.statusCode >= 400) {
                const errorMessage: string = (error !== undefined && error !== null) ? error.message :
                    `Submit trial ${trialJobId} failed, http code:${response.statusCode}, http body: ${response.body.message}`;
                trialJobDetail.status = 'FAILED';
                deferred.resolve(true);
            } else {
                trialJobDetail.submitTime = Date.now();
                deferred.resolve(true);
            }
        });

        return deferred.promise;
    }

    private async statusCheckingLoop(): Promise<void> {
        while (!this.stopping) {
            if(this.paiClusterConfig && this.paiClusterConfig.passWord) {
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
            await this.paiJobCollector.retrieveTrialStatus(this.paiToken, this.paiClusterConfig);
            const restServer: PAIJobRestServer = component.get(PAIJobRestServer);
            if (restServer.getErrorMessage !== undefined) {
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
            uri: `http://${this.paiClusterConfig.host}/rest-server/api/v1/token`,
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
        const timeoutDelay: Promise<void> = new Promise<void>((resolve: Function, reject: Function): void => {
            // Set timeout and reject the promise once reach timeout (5 seconds)
            timeoutId = setTimeout(
                () => reject(new Error('Get PAI token timeout. Please check your PAI cluster.')),
                5000);
        });

        return Promise.race([timeoutDelay, deferred.promise])
            .finally(() => { clearTimeout(timeoutId); });
    }

    private async writeParameterFile(trialJobId: string, hyperParameters: HyperParameters): Promise<void> {
        if (this.paiClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (this.paiTrialConfig === undefined) {
            throw new Error('PAI trial config is not initialized');
        }

        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        const hpFileName: string = generateParamFileName(hyperParameters);
        const localFilepath: string = path.join(trialLocalTempFolder, hpFileName);
        await fs.promises.writeFile(localFilepath, hyperParameters.value, { encoding: 'utf8' });
        const hdfsCodeDir: string = HDFSClientUtility.getHdfsTrialWorkDir(this.paiClusterConfig.userName, trialJobId);
        const hdfsHpFilePath: string = path.join(hdfsCodeDir, hpFileName);

        await HDFSClientUtility.copyFileToHdfs(localFilepath, hdfsHpFilePath, this.hdfsClient);

        await this.postParameterFileMeta({
            experimentId: this.experimentId,
            trialId: trialJobId,
            filePath: hdfsHpFilePath
        });
    }

    private postParameterFileMeta(parameterFileMeta: ParameterFileMeta): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();
        const restServer: PAIJobRestServer = component.get(PAIJobRestServer);
        const req: request.Options = {
            uri: `${restServer.endPoint}${restServer.apiRootUrl}/parameter-file-meta`,
            method: 'POST',
            json: true,
            body: parameterFileMeta
        };
        request(req, (err: Error, res: request.Response) => {
            if (err) {
                deferred.reject(err);
            } else {
                deferred.resolve();
            }
        });

        return deferred.promise;
    }
}

export { PAITrainingService };
