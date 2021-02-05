// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as path from 'path';
import * as request from 'request';
import * as component from '../../common/component';

import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import { MethodNotImplementedError } from '../../common/errors';
import {
    HyperParameters, NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric, LogType
} from '../../common/trainingService';
import { delay } from '../../common/utils';
import { PAIJobInfoCollector } from './paiJobInfoCollector';
import { PAIJobRestServer } from './paiJobRestServer';
import { PAIClusterConfig, PAITrialJobDetail, PAI_TRIAL_COMMAND_FORMAT, NNIPAITrialConfig } from './paiConfig';
import { String } from 'typescript-string-operations';
import {
    generateParamFileName,
    getIPV4Address, getVersion, uniqueString
} from '../../common/utils';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { execMkdir, validateCodeDir, execCopydir } from '../common/util';

const yaml = require('js-yaml');

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
    private paiClusterConfig?: PAIClusterConfig;
    private readonly jobQueue: string[];
    private stopping: boolean = false;
    private paiToken?: string;
    private paiTokenUpdateTime?: number;
    private readonly paiTokenUpdateInterval: number;
    private readonly experimentId!: string;
    private readonly paiJobCollector: PAIJobInfoCollector;
    private paiRestServerPort?: number;
    private nniManagerIpConfig?: NNIManagerIpConfig;
    private versionCheck: boolean = true;
    private logCollection: string;
    private isMultiPhase: boolean = false;
    private paiJobRestServer?: PAIJobRestServer;
    private protocol: string = 'http';
    private copyExpCodeDirPromise?: Promise<void>;
    private paiJobConfig: any;
    private nniVersion: string | undefined;
    private paiTrialConfig: NNIPAITrialConfig | undefined;

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

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;

            case TrialConfigMetadataKey.PAI_CLUSTER_CONFIG:
                this.paiJobRestServer = new PAIJobRestServer(component.get(PAITrainingService));
                this.paiClusterConfig = <PAIClusterConfig>JSON.parse(value);
                this.paiClusterConfig.host = this.formatPAIHost(this.paiClusterConfig.host);
                this.paiToken = this.paiClusterConfig.token;
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG: {
                if (this.paiClusterConfig === undefined) {
                    this.log.error('pai cluster config is not initialized');
                    break;
                }
                this.paiTrialConfig = <NNIPAITrialConfig>JSON.parse(value);
                // Validate to make sure codeDir doesn't have too many files
                await validateCodeDir(this.paiTrialConfig.codeDir);
                const nniManagerNFSExpCodeDir = path.join(this.paiTrialConfig.nniManagerNFSMountPath, this.experimentId, 'nni-code');
                await execMkdir(nniManagerNFSExpCodeDir);
                //Copy codeDir files to local working folder
                this.log.info(`Starting copy codeDir data from ${this.paiTrialConfig.codeDir} to ${nniManagerNFSExpCodeDir}`);
                this.copyExpCodeDirPromise = execCopydir(this.paiTrialConfig.codeDir, nniManagerNFSExpCodeDir);
                if (this.paiTrialConfig.paiConfigPath) {
                    this.paiJobConfig = yaml.safeLoad(fs.readFileSync(this.paiTrialConfig.paiConfigPath, 'utf8'));
                }
                break;
            }
            case TrialConfigMetadataKey.VERSION_CHECK:
                this.versionCheck = (value === 'true' || value === 'True');
                this.nniVersion = this.versionCheck ? await getVersion() : '';
                break;
            case TrialConfigMetadataKey.LOG_COLLECTION:
                this.logCollection = value;
                break;
            case TrialConfigMetadataKey.MULTI_PHASE:
                this.isMultiPhase = (value === 'true' || value === 'True');
                break;
            default:
                //Reject for unknown keys
                this.log.error(`Uknown key: ${key}`);
        }
    }

    // update trial parameters for multi-phase
    public async updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const trialJobDetail: PAITrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }
        // Write file content ( parameter.cfg ) to working folders
        await this.writeParameterFile(trialJobDetail.logPath, form.hyperParameters);

        return trialJobDetail;
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        if (this.paiClusterConfig === undefined) {
            throw new Error(`paiClusterConfig not initialized!`);
        }
        if (this.paiTrialConfig === undefined) {
            throw new Error(`paiTrialConfig not initialized!`);
        }

        this.log.info(`submitTrialJob: form: ${JSON.stringify(form)}`);

        const trialJobId: string = uniqueString(5);
        //TODO: use HDFS working folder instead
        const trialWorkingFolder: string = path.join(this.expRootDir, 'trials', trialJobId);
        const paiJobName: string = `nni_exp_${this.experimentId}_trial_${trialJobId}`;
        const logPath: string = path.join(this.paiTrialConfig.nniManagerNFSMountPath, this.experimentId, trialJobId);
        const paiJobDetailUrl: string = `${this.protocol}://${this.paiClusterConfig.host}/job-detail.html?username=${this.paiClusterConfig.userName}&jobName=${paiJobName}`;
        const trialJobDetail: PAITrialJobDetail = new PAITrialJobDetail(
            trialJobId,
            'WAITING',
            paiJobName,
            Date.now(),
            trialWorkingFolder,
            form,
            logPath,
            paiJobDetailUrl);

        this.trialJobsMap.set(trialJobId, trialJobDetail);
        this.jobQueue.push(trialJobId);

        return trialJobDetail;
    }

    private generateNNITrialCommand(trialJobDetail: PAITrialJobDetail, command: string): string {
        if (this.paiTrialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        const containerNFSExpCodeDir = `${this.paiTrialConfig.containerNFSMountPath}/${this.experimentId}/nni-code`;
        const containerWorkingDir: string = `${this.paiTrialConfig.containerNFSMountPath}/${this.experimentId}/${trialJobDetail.id}`;
        const nniManagerIp: string = this.nniManagerIpConfig ? this.nniManagerIpConfig.nniManagerIp : getIPV4Address();
        const nniPaiTrialCommand: string = String.Format(
            PAI_TRIAL_COMMAND_FORMAT,
            `${containerWorkingDir}`,
            `${containerWorkingDir}/nnioutput`,
            trialJobDetail.id,
            this.experimentId,
            trialJobDetail.form.sequenceId,
            this.isMultiPhase,
            containerNFSExpCodeDir,
            command,
            nniManagerIp,
            this.paiRestServerPort,
            this.nniVersion,
            this.logCollection
        )
            .replace(/\r\n|\n|\r/gm, '');

        return nniPaiTrialCommand;

    }

    private generateJobConfigInYamlFormat(trialJobDetail: PAITrialJobDetail): any {
        if (this.paiTrialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        const jobName = `nni_exp_${this.experimentId}_trial_${trialJobDetail.id}`

        let nniJobConfig: any = undefined;
        if (this.paiTrialConfig.paiConfigPath) {
            nniJobConfig = JSON.parse(JSON.stringify(this.paiJobConfig)); //Trick for deep clone in Typescript
            nniJobConfig.name = jobName;
            // Each taskRole will generate new command in NNI's command format
            // Each command will be formatted to NNI style
            for (const taskRoleIndex in nniJobConfig.taskRoles) {
                const commands = nniJobConfig.taskRoles[taskRoleIndex].commands
                const nniTrialCommand = this.generateNNITrialCommand(trialJobDetail, commands.join(" && ").replace(/(["'$`\\])/g, '\\$1'));
                nniJobConfig.taskRoles[taskRoleIndex].commands = [nniTrialCommand]
            }

        } else {
            nniJobConfig = {
                protocolVersion: 2,
                name: jobName,
                type: 'job',
                jobRetryCount: 0,
                prerequisites: [
                    {
                        type: 'dockerimage',
                        uri: this.paiTrialConfig.image,
                        name: 'docker_image_0'
                    }
                ],
                taskRoles: {
                    taskrole: {
                        instances: 1,
                        completion: {
                            minFailedInstances: 1,
                            minSucceededInstances: -1
                        },
                        taskRetryCount: 0,
                        dockerImage: 'docker_image_0',
                        resourcePerInstance: {
                            gpu: this.paiTrialConfig.gpuNum,
                            cpu: this.paiTrialConfig.cpuNum,
                            memoryMB: this.paiTrialConfig.memoryMB
                        },
                        commands: [
                            this.generateNNITrialCommand(trialJobDetail, this.paiTrialConfig.command)
                        ]
                    }
                },
                extras: {
                    'storages': [
                        {
                            name: this.paiTrialConfig.paiStorageConfigName
                        }
                    ],
                    submitFrom: 'submit-job-v2'
                }
            }
            if (this.paiTrialConfig.virtualCluster) {
                nniJobConfig.defaults = {
                    virtualCluster: this.paiTrialConfig.virtualCluster
                }
            }
        }
        return yaml.safeDump(nniJobConfig);
    }

    protected async submitTrialJobToPAI(trialJobId: string): Promise<boolean> {
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

        if (this.paiJobRestServer === undefined) {
            throw new Error('paiJobRestServer is not initialized');
        }

        // Make sure experiment code files is copied from local to NFS
        if (this.copyExpCodeDirPromise !== undefined) {
            await this.copyExpCodeDirPromise;
            this.log.info(`Copy codeDir data finished.`);
            // All trials share same destination NFS code folder, only copy codeDir once for an experiment.
            // After copy data finished, set copyExpCodeDirPromise be undefined to avoid log content duplicated.
            this.copyExpCodeDirPromise = undefined;
        }

        this.paiRestServerPort = this.paiJobRestServer.clusterRestServerPort;

        // Step 1. Prepare PAI job configuration
        //create trial local working folder locally.
        await execMkdir(trialJobDetail.logPath);
        // Write NNI installation file to local files
        await fs.promises.writeFile(path.join(trialJobDetail.logPath, 'install_nni.sh'), CONTAINER_INSTALL_NNI_SHELL_FORMAT, { encoding: 'utf8' });

        // Write file content ( parameter.cfg ) to local working folders
        if (trialJobDetail.form !== undefined) {
            await this.writeParameterFile(trialJobDetail.logPath, trialJobDetail.form.hyperParameters);
        }

        //Generate Job Configuration in yaml format
        const paiJobConfig = this.generateJobConfigInYamlFormat(trialJobDetail);
        this.log.debug(paiJobConfig);
        // Step 2. Submit PAI job via Rest call
        // Refer https://github.com/Microsoft/pai/blob/master/docs/rest-server/API.md for more detail about PAI Rest API
        const submitJobRequest: request.Options = {
            uri: `${this.protocol}://${this.paiClusterConfig.host}/rest-server/api/v2/jobs`,
            method: 'POST',
            body: paiJobConfig,
            followAllRedirects: true,
            headers: {
                'Content-Type': 'text/yaml',
                Authorization: `Bearer ${this.paiToken}`
            }
        };
        request(submitJobRequest, (error: Error, response: request.Response, body: any) => {
            // If submit success, will get status code 202. refer: https://github.com/microsoft/pai/blob/master/src/rest-server/docs/swagger.yaml
            if ((error !== undefined && error !== null) || response.statusCode >= 400) {
                const errorMessage: string = (error !== undefined && error !== null) ? error.message :
                    `Submit trial ${trialJobId} failed, http code:${response.statusCode}, http body: ${body}`;
                this.log.error(errorMessage);
                trialJobDetail.status = 'FAILED';
                deferred.reject(errorMessage);
            } else {
                trialJobDetail.submitTime = Date.now();
            }
            deferred.resolve(true);
        });

        return deferred.promise;
    }

    private async writeParameterFile(directory: string, hyperParameters: HyperParameters): Promise<void> {
        const filepath: string = path.join(directory, generateParamFileName(hyperParameters));
        await fs.promises.writeFile(filepath, hyperParameters.value, { encoding: 'utf8' });
    }
}

export { PAITrainingService };
