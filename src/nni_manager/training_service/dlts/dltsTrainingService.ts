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
import { getExperimentId, getExperimentStartupInfo } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import {
    HyperParameters, NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric
} from '../../common/trainingService';
import { DLTS_TRIAL_COMMAND_FORMAT } from './dltsTemplates';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { execMkdir, validateCodeDir, execCopydir } from '../common/util';
import { delay, uniqueString, getIPV4Address, getExperimentRootDir, getVersion, generateParamFileName } from '../../common/utils';
import { DLTSJobRestServer, ParameterFileMeta } from './dltsJobRestServer';
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
    private dltsRestServerPort?: number;

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
        while (!this.stopping) {
            const updateDLTSTrialJobs: Promise<void>[] = [];
            for (const [trialJobId, dltsTrialJob] of this.trialJobsMap) {
                updateDLTSTrialJobs.push(this.getDLTSTrialJobInfo(dltsTrialJob));
            }
    
            await Promise.all(updateDLTSTrialJobs);

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
                dltsTrialJob.status = "RUNNING";
                dltsTrialJob.status = "RUNNING";
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
    }

    private async submitJobLoop(): Promise<void> {
        while (!this.stopping) {
            while (!this.stopping && this.jobQueue.length > 0) {
                const trialJobId: string = this.jobQueue[0];
                this.log.info('Got job ' + trialJobId);
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
            '/nni/experiments', getExperimentId(),
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

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;

            case TrialConfigMetadataKey.DLTS_CLUSTER_CONFIG:
                this.dltsClusterConfig = <DLTSClusterConfig>JSON.parse(value);
                if (!this.dltsClusterConfig.email && process.env['DLWS_USER_EMAIL'] !== undefined) {
                    this.dltsClusterConfig.email = process.env['DLWS_USER_EMAIL']
                }
                if (!this.dltsClusterConfig.password && process.env['DLTS_JOB_TOKEN'] !== undefined) {
                    this.dltsClusterConfig.password = process.env['DLTS_JOB_TOKEN']
                }
                if (!this.dltsClusterConfig.team && process.env['DLWS_VC_NAME'] !== undefined) {
                    this.dltsClusterConfig.team = process.env['DLWS_VC_NAME']
                }
                // TODO: move GPU here
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG:
                this.dltsTrialConfig = <DLTSTrialConfig>JSON.parse(value);

                // Validate to make sure codeDir doesn't have too many files
                try {
                    await validateCodeDir('/work/nni-code');
                } catch (error) {
                    this.log.error(error);
                    throw error;
                }

                await execCopydir(this.dltsTrialConfig.codeDir, '/work/nni-code')
           
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

    public async getClusterMetadata(key: string): Promise<string> {
        return '';
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
        const trialJobDetail: DLTSTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (trialJobDetail === undefined) {
            throw new Error(`Failed to find PAITrialJobDetail for job ${trialJobId}`);
        }

        if (this.dltsClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (this.dltsTrialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }

        if (this.dltsRestServerPort === undefined) {
            const restServer: DLTSJobRestServer = component.get(DLTSJobRestServer);
            this.dltsRestServerPort = restServer.clusterRestServerPort;
        }

        // Step 1. Prepare PAI job configuration

        const trialLocalTempFolder = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
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
        // tslint:disable-next-line: strict-boolean-expressions
        const nniManagerIp: string = this.nniManagerIpConfig ? this.nniManagerIpConfig.nniManagerIp : getIPV4Address();
        const version: string = this.versionCheck ? await getVersion() : '';
        const nniDLTSTrialCommand: string = String.Format(
            DLTS_TRIAL_COMMAND_FORMAT,
            // PAI will copy job's codeDir into /root directory
            `/work/trials/${trialJobId}`,
            `/work/trials/${trialJobId}/nnioutput`,
            trialJobId,
            this.experimentId,
            trialJobDetail.form.sequenceId,
            false,
            this.dltsTrialConfig.command,
            nniManagerIp,
            this.dltsRestServerPort,
            version,
            this.logCollection
        )
        .replace(/\r\n|\n|\r/gm, '');

        // Step 2. Upload code files in codeDir onto NFS
        
        const codeDir = path.join('/work/trials', trialJobId);
        try {
            await execCopydir(trialLocalTempFolder, codeDir)
        } catch (error) {
            this.log.error(`DLTS Training Service: failed to copy ${trialLocalTempFolder} to ${codeDir}\n${error}`)
            return true
        }

        // Step 3. Submit DLTS job via Rest call
        const gpuRequestOptions: request.Options = {
            method: 'GET',
            qs: {
                email: this.dltsClusterConfig.email,
                password: this.dltsClusterConfig.password
            },
            uri: `${this.dltsClusterConfig.dashboard}api/teams/${this.dltsClusterConfig.team}/clusters/${this.dltsClusterConfig.cluster}`,
            json: true
        };
        const gpu = await new Promise<string>((resolve, reject) => {
            request(gpuRequestOptions, (error, response, data) => {
                if (error) {
                    return reject(error)
                }
                let gpus = Object.keys(data['gpu_capacity'])
                resolve(gpus[0])
            })
        });
        const dltsJobConfig: DLTSJobConfig = new DLTSJobConfig(
            this.dltsClusterConfig,
            gpu,
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
            throw new Error('PAI Cluster config is not initialized');
        }
        if (this.dltsTrialConfig === undefined) {
            throw new Error('PAI trial config is not initialized');
        }

        const hyperParameters = form.hyperParameters;
        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        const hpFileName: string = generateParamFileName(hyperParameters);
        const localFilepath: string = path.join(trialLocalTempFolder, hpFileName);
        await fs.promises.writeFile(localFilepath, hyperParameters.value, { encoding: 'utf8' });

        const parameterFileMeta = {
            experimentId: this.experimentId,
            trialId: trialJobId,
            // filePath: hdfsHpFilePath
        };
        const restServer: DLTSJobRestServer = component.get(DLTSJobRestServer);
        const req: request.Options = {
            uri: `${restServer.endPoint}${restServer.apiRootUrl}/parameter-file-meta`,
            method: 'POST',
            json: true,
            body: parameterFileMeta
        };
        await new Promise((resolve, reject) => {
            request(req, (err: Error, res: request.Response) => {
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