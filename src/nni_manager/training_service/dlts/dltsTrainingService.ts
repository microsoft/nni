// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as path from 'path';
import * as request from 'request';

import * as component from '../../common/component';
import { EventEmitter } from 'events';
import { String } from 'typescript-string-operations';
import { getExperimentId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import { MethodNotImplementedError } from '../../common/errors';
import {
    NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric, LogType
} from '../../common/trainingService';
import { DLTS_TRIAL_COMMAND_FORMAT } from './dltsData';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { execMkdir, validateCodeDir } from '../common/util';
import { delay, uniqueString, getIPV4Address, getExperimentRootDir, getVersion, generateParamFileName } from '../../common/utils';
import { DLTSJobRestServer } from './dltsJobRestServer';
import { TrialConfigMetadataKey } from '../../training_service/common/trialConfigMetadataKey';
import { DLTSJobConfig } from './dltsJobConfig';
import { DLTSClusterConfig } from './dltsClusterConfig';
import { DLTSTrialConfig } from './dltsTrialConfig';
import { DLTSTrialJobDetail } from './dltsTrialJobDetail';

@component.Singleton
class DLTSTrainingService implements TrainingService {
    private readonly log!: Logger;
    private readonly metricsEmitter: EventEmitter;
    //private readonly expRootDir: string;
    private readonly jobQueue: string[];
    private stopping: boolean = false;
    private readonly experimentId!: string;
    private versionCheck: boolean = true;
    private logCollection: string = 'none';
    private isMultiPhase: boolean = false;
    private dltsRestServerHost: string;
    private dltsRestServerPort?: number;
    private jobMode: boolean;

    private readonly trialJobsMap: Map<string, DLTSTrialJobDetail>;
    private nniManagerIpConfig?: NNIManagerIpConfig;
    private dltsClusterConfig?: DLTSClusterConfig;
    private dltsTrialConfig?: DLTSTrialConfig;

    constructor() {
        this.log = getLogger();
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map();
        this.jobQueue = [];
        this.experimentId = getExperimentId();
        this.dltsRestServerHost = getIPV4Address();
        this.jobMode = 'DLTS_JOB_ID' in process.env;
        this.log.info(`Construct DLTS training service in ${this.jobMode ? 'job mode' : 'local mode'}.`);
    }

    public async run(): Promise<void> {
        this.log.info('Run DLTS training service.');
        const restServer: DLTSJobRestServer = component.get(DLTSJobRestServer);
        await restServer.start();
        restServer.setEnableVersionCheck = this.versionCheck;
        this.log.info(`DLTS Training service rest server listening on: ${restServer.endPoint}`);
        if (this.jobMode) {
            await this.exposeRestServerPort(restServer.clusterRestServerPort);
        } else {
            this.dltsRestServerPort = restServer.clusterRestServerPort
        }
        await Promise.all([
            this.statusCheckingLoop(),
            this.submitJobLoop()]);
        this.log.info('DLTS training service exit.');
    }

    private async exposeRestServerPort(port: number): Promise<void> {
        if (this.dltsClusterConfig == null) {
            throw Error('Cluster config is not set');
        }
        const { dashboard, cluster, email, password } = this.dltsClusterConfig;
        const jobId = process.env['DLTS_JOB_ID'] + '';
        const uri = `${dashboard}api/clusters/${cluster}/jobs/${jobId}/endpoints`;
        const qs = { email, password };

        do {
            this.log.debug('Checking endpoints');
            const endpoints = await new Promise((resolve, reject) => {
                request.get(uri, { qs, json: true }, function (error, response, body) {
                    if (error) {
                        reject(error);
                    } else {
                        resolve(body);
                    }
                });
            });
            this.log.debug('Endpoints: %o', endpoints);
            if (Array.isArray(endpoints)) {
                const restServerEndpoint = endpoints.find(({ podPort }) => podPort === port);
                if (restServerEndpoint == null) {
                    this.log.debug('Exposing %d', port);
                    await new Promise((resolve, reject) => {
                        request.post(uri, {
                            qs,
                            json: true,
                            body: {
                                endpoints: [{
                                    name: "nni-rest-server",
                                    podPort: port
                                }]
                            }
                        }, function (error) {
                            if (error) {
                                reject(error);
                            } else {
                                resolve();
                            }
                        });
                    });
                } else if (restServerEndpoint['status'] === 'running') {
                    // We get an exposed restserver port
                    this.dltsRestServerHost = restServerEndpoint['nodeName'];
                    this.dltsRestServerPort = restServerEndpoint['port'];
                    break;
                }
            }
        } while (await new Promise(resolve => setTimeout(resolve, 1000, true)));
    }

    private async statusCheckingLoop(): Promise<void> {
        while (!this.stopping) {
            const updateDLTSTrialJobs: Promise<void>[] = [];
            for (const dltsTrialJob of this.trialJobsMap.values()) {
                updateDLTSTrialJobs.push(this.getDLTSTrialJobInfo(dltsTrialJob));
            }
    
            await Promise.all(updateDLTSTrialJobs);

            // Calcel paused dlts job
            const cancelPausedJobPromises: Promise<void>[] = [];
            for (const [trialJobId, dltsTrialJob] of this.trialJobsMap) {
                if (dltsTrialJob.dltsPaused && dltsTrialJob.status === 'RUNNING') {
                    cancelPausedJobPromises.push(this.cancelTrialJob(trialJobId));
                }
            }
            await Promise.all(cancelPausedJobPromises);

            const restServer: DLTSJobRestServer = component.get(DLTSJobRestServer);
            if (restServer.getErrorMessage !== undefined) {
                throw new Error(restServer.getErrorMessage);
            }
            await delay(3000);
        }
    }

    private async getDLTSTrialJobInfo(dltsTrialJob: DLTSTrialJobDetail): Promise<void> {
        if (this.dltsClusterConfig == null) {
            throw Error('Cluster config is not set');
        }
        const requestOptions: request.Options = {
            uri: `${this.dltsClusterConfig.dashboard}api/v2/clusters/${this.dltsClusterConfig.cluster}/jobs/${dltsTrialJob.dltsJobId}`,
            qs: {
                email: this.dltsClusterConfig.email,
                password: this.dltsClusterConfig.password
            },
            json: true
        };
        const body = await new Promise((resolve, reject) => {
            request(requestOptions, (error, response, body) => {
                if (error != null) {
                    reject(error)
                } else {
                    resolve(body)
                }
            })
        }) as any;
        void ((): void => {
            switch (body['jobStatus']) {
                case 'unapproved':
                case 'queued':
                case 'scheduling':
                    dltsTrialJob.status = "WAITING";
                    break;
                case 'running':
                    dltsTrialJob.status = "RUNNING";
                    if (dltsTrialJob.startTime === undefined) {
                        dltsTrialJob.startTime = Date.parse(body['jobStatusDetail'][0]['startedAt'])
                    }
                    if (dltsTrialJob.url === undefined) {
                        dltsTrialJob.url = `${this.dltsClusterConfig.dashboard}job/${this.dltsClusterConfig.team}/${this.dltsClusterConfig.cluster}/${dltsTrialJob.dltsJobId}`
                    }
                    break;
                case 'finished':
                    dltsTrialJob.status = "SUCCEEDED";
                    break;
                case 'failed':
                    dltsTrialJob.status = "FAILED";
                    break;
                case 'pausing':
                case 'paused':
                    dltsTrialJob.status = "RUNNING";
                    dltsTrialJob.dltsPaused = true;
                    break;
                case 'killing':
                case 'killed':
                    if (dltsTrialJob.isEarlyStopped !== undefined) {
                        dltsTrialJob.status = dltsTrialJob.isEarlyStopped === true
                            ? 'EARLY_STOPPED' : 'USER_CANCELED';
                    } else {
                        dltsTrialJob.status = 'SYS_CANCELED';
                    }
                    break;
                default:
                    dltsTrialJob.status = "UNKNOWN";
            }
        }) ();
    }

    private async submitJobLoop(): Promise<void> {
        while (!this.stopping) {
            while (!this.stopping && this.jobQueue.length > 0) {
                const trialJobId: string = this.jobQueue[0];
                this.log.info(`Got job ${trialJobId}`);
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
        return Array.from(this.trialJobsMap.values());
    }

    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        const trialJob = this.trialJobsMap.get(trialJobId);
        if (trialJob === undefined) {
            throw Error(`Trial job ${trialJobId} not found.`)
        }
        return trialJob
    }

    public async getTrialLog(_trialJobId: string, _logType: LogType): Promise<string> {
        throw new MethodNotImplementedError();
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
        const trialJobId: string = uniqueString(5);
        const trialWorkingFolder: string = path.join(
            '/nni-experiments', getExperimentId(),
            '/trials/', trialJobId);
        const trialJobDetail = new DLTSTrialJobDetail(
            trialJobId, // id
            'WAITING', // status
            Date.now(), // submitTime
            trialWorkingFolder, // workingDirectory
            form,
            `nni_exp_${this.experimentId}_trial_${trialJobId}`
        );

        this.trialJobsMap.set(trialJobId, trialJobDetail);
        this.jobQueue.push(trialJobId);

        return trialJobDetail;
    }

    public async cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        const trialJobDetail: DLTSTrialJobDetail | undefined =  this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw Error(`cancelTrialJob: trial job id ${trialJobId} not found`);
        }

        if (this.dltsClusterConfig === undefined) {
            throw Error('DLTS Cluster config is not initialized');
        }

        const options: request.Options = {
            method: 'PUT',
            uri: `${this.dltsClusterConfig.dashboard}api/clusters/${this.dltsClusterConfig.cluster}/jobs/${trialJobDetail.dltsJobId}/status`,
            qs: {
                email: this.dltsClusterConfig.email,
                password: this.dltsClusterConfig.password
            },
            body: {
                status: 'killing'
            },
            json: true
        };

        // Set trialjobDetail's early stopped field, to mark the job's cancellation source
        trialJobDetail.isEarlyStopped = isEarlyStopped;

        await new Promise((resolve, reject) => {
            request(options, (error: Error, response: request.Response, body: any) => {
                if (error) {
                    reject(error);
                } else {
                    resolve(body);
                }
            });
        });
    }

    private async getGpuType(): Promise<string> {
        if (this.dltsClusterConfig === undefined) {
            throw new Error('DLTS Cluster config is not initialized');
        }
        const gpuRequestOptions: request.Options = {
            method: 'GET',
            qs: {
                email: this.dltsClusterConfig.email,
                password: this.dltsClusterConfig.password
            },
            uri: `${this.dltsClusterConfig.dashboard}api/teams/${this.dltsClusterConfig.team}/clusters/${this.dltsClusterConfig.cluster}`,
            json: true
        };
        return new Promise<string>((resolve, reject) => {
            request(gpuRequestOptions, (error, response, data) => {
                if (error) {
                    return reject(error)
                }
                try {
                    const metadata = JSON.parse(data['metadata'])
                    resolve(Object.keys(metadata)[0])
                } catch (error) {
                    reject(error)
                }
            })
        });
    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;

            case TrialConfigMetadataKey.DLTS_CLUSTER_CONFIG:
                this.dltsClusterConfig = <DLTSClusterConfig>JSON.parse(value);
                if (!this.dltsClusterConfig.cluster) {
                    this.dltsClusterConfig.cluster = '.default'
                }
                if (!this.dltsClusterConfig.email) {
                    if (process.env['DLWS_USER_EMAIL']) {
                        this.dltsClusterConfig.email = process.env['DLWS_USER_EMAIL'] as string
                    } else {
                        throw Error('`email` field in `dltsConfig` is not configured.')
                    }
                }
                if (!this.dltsClusterConfig.password) {
                    if (process.env['DLTS_JOB_TOKEN']) {
                        this.dltsClusterConfig.password = process.env['DLTS_JOB_TOKEN'] as string
                    } else {
                        throw Error('`password` field in `dltsConfig` is not configured.')
                    }
                }
                if (!this.dltsClusterConfig.team) {
                    if (process.env['DLWS_VC_NAME']) {
                        this.dltsClusterConfig.team = process.env['DLWS_VC_NAME'] as string
                    } else {
                        throw Error('`team` field in `dltsConfig` is not configured.')
                    }
                }
                this.dltsClusterConfig.gpuType = await this.getGpuType();
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG:
                this.dltsTrialConfig = <DLTSTrialConfig>JSON.parse(value);

                // Validate to make sure codeDir doesn't have too many files
                try {
                    await validateCodeDir(this.dltsTrialConfig.codeDir);
                } catch (error) {
                    this.log.error(error);
                    throw error;
                }
           
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
    }

    public async getClusterMetadata(_key: string): Promise<string> {
        return '';
    }

    public async cleanUp(): Promise<void> {
        this.log.info('Stopping DLTS training service...');
        this.stopping = true;

        const restServer: DLTSJobRestServer = component.get(DLTSJobRestServer);
        try {
            await restServer.stop();
            this.log.info('DLTS Training service rest server stopped successfully.');
            return;
        } catch (error) {
            // tslint:disable-next-line: no-unsafe-any
            this.log.error(`DLTS Training service rest server stopped failed, error: ${error.message}`);
            throw error;
        }
    }

    private async submitTrialJobToDLTS(trialJobId: string): Promise<boolean> {
        const trialJobDetail: DLTSTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (trialJobDetail === undefined) {
            throw new Error(`Failed to find DLTSTrialJobDetail for job ${trialJobId}`);
        }

        if (this.dltsClusterConfig === undefined) {
            throw new Error('DLTS Cluster config is not initialized');
        }
        if (this.dltsTrialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }

        if (this.dltsRestServerPort === undefined) {
            const restServer: DLTSJobRestServer = component.get(DLTSJobRestServer);
            this.dltsRestServerPort = restServer.clusterRestServerPort;
        }

        // Step 1. Prepare DLTS job configuration

        const trialLocalFolder = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        //create tmp trial working folder locally.
        await execMkdir(trialLocalFolder);

        const runScriptContent: string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });

        // Write file content ( parameter.cfg ) to local tmp folders
        if (trialJobDetail.form !== undefined) {
            await fs.promises.writeFile(
                path.join(trialLocalFolder, generateParamFileName(trialJobDetail.form.hyperParameters)),
                trialJobDetail.form.hyperParameters.value, { encoding: 'utf8' }
            );
        }
        // tslint:disable-next-line: strict-boolean-expressions
        const nniManagerIp: string = this.nniManagerIpConfig ? this.nniManagerIpConfig.nniManagerIp : this.dltsRestServerHost;
        const version: string = this.versionCheck ? await getVersion() : '';
        const nniDLTSTrialCommand: string = String.Format(
            DLTS_TRIAL_COMMAND_FORMAT,
            trialLocalFolder,
            path.join(trialLocalFolder, 'nnioutput'),
            trialJobId,
            this.experimentId,
            trialJobDetail.form.sequenceId,
            false,
            this.dltsTrialConfig.codeDir,
            this.dltsTrialConfig.command,
            nniManagerIp,
            this.dltsRestServerPort,
            version,
            this.logCollection
        )
        .replace(/\r\n|\n|\r/gm, '');

        // Step 2. Submit DLTS job via Rest call
        const dltsJobConfig: DLTSJobConfig = new DLTSJobConfig(
            this.dltsClusterConfig,
            trialJobDetail.dltsJobName,
            this.dltsTrialConfig.gpuNum,
            this.dltsTrialConfig.image,
            nniDLTSTrialCommand,
            []
        );
        const submitJobRequest: request.Options = {
            method: 'POST',
            uri: `${this.dltsClusterConfig.dashboard}api/clusters/${this.dltsClusterConfig.cluster}/jobs`,
            qs: {
                email: this.dltsClusterConfig.email,
                password: this.dltsClusterConfig.password
            },
            body: dltsJobConfig,
            json: true
        }
        const responseData = await new Promise<any>((resolve, reject) => {
            request(submitJobRequest, function (error, response, data) {
                if (error) {
                    return reject(error)
                } else {
                    return resolve(data)
                }
            })
        });

        trialJobDetail.dltsJobId = responseData['jobId']

        return true;
    }

    public async updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const trialJobDetail: undefined | TrialJobDetail = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }
        if (this.dltsClusterConfig === undefined) {
            throw new Error('DLTS Cluster config is not initialized');
        }
        if (this.dltsTrialConfig === undefined) {
            throw new Error('DLTS trial config is not initialized');
        }

        const hyperParameters = form.hyperParameters;
        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        const hpFileName: string = generateParamFileName(hyperParameters);
        const localFilepath: string = path.join(trialLocalTempFolder, hpFileName);
        await fs.promises.writeFile(localFilepath, hyperParameters.value, { encoding: 'utf8' });

        const parameterFileMeta = {
            experimentId: this.experimentId,
            trialId: trialJobId
        };
        const restServer: DLTSJobRestServer = component.get(DLTSJobRestServer);
        const req: request.Options = {
            uri: `${restServer.endPoint}${restServer.apiRootUrl}/parameter-file-meta`,
            method: 'POST',
            json: true,
            body: parameterFileMeta
        };
        await new Promise((resolve, reject) => {
            request(req, (err: Error, _res: request.Response) => {
                if (err) {
                    reject(err);
                } else {
                    resolve();
                }
            });
        });

        return trialJobDetail;
    }

    public get isMultiPhaseJobSupported(): boolean {
        return false;
    }
}

export { DLTSTrainingService };
