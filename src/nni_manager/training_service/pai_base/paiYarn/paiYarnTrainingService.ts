// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as path from 'path';
import * as request from 'request';
import * as component from '../../../common/component';

import { EventEmitter } from 'events';
import { Deferred } from 'ts-deferred';
import { String } from 'typescript-string-operations';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../../common/log';
import {
    HyperParameters, NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric
} from '../../../common/trainingService';
import { delay, generateParamFileName,
    getExperimentRootDir, getIPV4Address, getVersion, uniqueString, unixPathJoin } from '../../../common/utils';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../../common/containerJobData';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { execMkdir, validateCodeDir } from '../../common/util';
import { HDFSClientUtility } from './hdfsClientUtility';
import { NNIPAITrialConfig, PAIJobConfig, PAITaskRole } from './paiYarnConfig';
import { PAI_LOG_PATH_FORMAT, PAI_TRIAL_COMMAND_FORMAT } from './paiYarnData';
import { PAIBaseJobInfoCollector } from '../paiBaseJobInfoCollector';
import { PAIYarnJobRestServer, ParameterFileMeta } from './paiYarnJobRestServer';
import { PAIBaseTrainingService } from '../paiBaseTrainingService';
import { PAIBaseClusterConfig, PAIBaseTrialJobDetail } from '../paiBaseConfig';

import * as WebHDFS from 'webhdfs';

/**
 * Training Service implementation for OpenPAI (Open Platform for AI)
 * Refer https://github.com/Microsoft/pai for more info about OpenPAI
 */
@component.Singleton
class PAIYarnTrainingService extends PAIBaseTrainingService {
    private hdfsClient: any;
    private copyExpCodeDirPromise?: Promise<void>;
    private copyAuthFilePromise?: Promise<void>;
    private paiTrialConfig?: NNIPAITrialConfig;

    constructor() {
        super();
    }

    public async run(): Promise<void> {
        this.log.info('Run PAIYarn training service.');
        this.paiBaseJobRestServer = component.get(PAIYarnJobRestServer);
        if (this.paiBaseJobRestServer === undefined) {
            throw new Error('paiBaseJobRestServer not initialized!');
        }
        await this.paiBaseJobRestServer.start();
        this.paiBaseJobRestServer.setEnableVersionCheck = this.versionCheck;
        this.log.info(`PAI Training service rest server listening on: ${this.paiBaseJobRestServer.endPoint}`);
        await Promise.all([
            this.statusCheckingLoop(),
            this.submitJobLoop()]);
        this.log.info('PAI training service exit.');
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        if (this.paiBaseClusterConfig === undefined) {
            throw new Error(`paiBaseClusterConfig not initialized!`);
        }
        const deferred: Deferred<PAIBaseTrialJobDetail> = new Deferred<PAIBaseTrialJobDetail>();

        this.log.info(`submitTrialJob: form: ${JSON.stringify(form)}`);

        const trialJobId: string = uniqueString(5);
        //TODO: use HDFS working folder instead
        const trialWorkingFolder: string = path.join(this.expRootDir, 'trials', trialJobId);
        const paiJobName: string = `nni_exp_${this.experimentId}_trial_${trialJobId}`;
        const hdfsCodeDir: string = HDFSClientUtility.getHdfsTrialWorkDir(this.paiBaseClusterConfig.userName, trialJobId);
        const hdfsOutputDir: string = unixPathJoin(hdfsCodeDir, 'nnioutput');

        const hdfsLogPath: string = String.Format(
            PAI_LOG_PATH_FORMAT,
            this.paiBaseClusterConfig.host,
            hdfsOutputDir
            );

        const trialJobDetail: PAIBaseTrialJobDetail = new PAIBaseTrialJobDetail(
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

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();

        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                deferred.resolve();
                break;

            case TrialConfigMetadataKey.PAI_YARN_CLUSTER_CONFIG:
                this.paiBaseClusterConfig = <PAIBaseClusterConfig>JSON.parse(value);

                this.hdfsClient = WebHDFS.createClient({
                    user: this.paiBaseClusterConfig.userName,
                    // Refer PAI document for Pylon mapping https://github.com/Microsoft/pai/tree/master/docs/pylon
                    port: 80,
                    path: '/webhdfs/api/v1',
                    host: this.paiBaseClusterConfig.host
                });
                if(this.paiBaseClusterConfig.passWord) {
                    // Get PAI authentication token
                    await this.updatePaiToken();
                } else if(this.paiBaseClusterConfig.token) {
                    this.paiToken = this.paiBaseClusterConfig.token;
                } else {
                    deferred.reject(new Error('paiBase cluster config format error, please set password or token!'));
                }

                deferred.resolve();
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG:
                if (this.paiBaseClusterConfig === undefined) {
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
                    HDFSClientUtility.getHdfsExpCodeDir(this.paiBaseClusterConfig.userName),
                    this.hdfsClient
                );
                
                // Upload authFile to hdfs
                if (this.paiTrialConfig.authFile) {
                    this.authFileHdfsPath = unixPathJoin(HDFSClientUtility.hdfsExpRootDir(this.paiBaseClusterConfig.userName), 'authFile');
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

    protected async submitTrialJobToPAI(trialJobId: string): Promise<boolean> {
        const deferred: Deferred<boolean> = new Deferred<boolean>();
        const trialJobDetail: PAIBaseTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);

        if (trialJobDetail === undefined) {
            throw new Error(`Failed to find PAITrialJobDetail for job ${trialJobId}`);
        }

        if (this.paiBaseClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (this.paiTrialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        if (this.paiToken === undefined) {
            throw new Error('PAI token is not initialized');
        }

        if (this.paiRestServerPort === undefined) {
            const restServer: PAIYarnJobRestServer = component.get(PAIYarnJobRestServer);
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
        const hdfsCodeDir: string = HDFSClientUtility.getHdfsTrialWorkDir(this.paiBaseClusterConfig.userName, trialJobId);
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
            this.paiBaseClusterConfig.host,
            this.paiBaseClusterConfig.userName,
            HDFSClientUtility.getHdfsExpCodeDir(this.paiBaseClusterConfig.userName),
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
            uri: `http://${this.paiBaseClusterConfig.host}/rest-server/api/v1/user/${this.paiBaseClusterConfig.userName}/jobs`,
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

    public async updateTrialJob(trialJobId: string, form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        const trialJobDetail: undefined | TrialJobDetail = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }
        await this.writeParameterFile(trialJobId, form.hyperParameters);

        return trialJobDetail;
    }

    protected async writeParameterFile(trialJobId: string, hyperParameters: HyperParameters): Promise<void> {
        if (this.paiBaseClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (this.paiTrialConfig === undefined) {
            throw new Error('PAI trial config is not initialized');
        }

        const trialLocalTempFolder: string = path.join(getExperimentRootDir(), 'trials-local', trialJobId);
        const hpFileName: string = generateParamFileName(hyperParameters);
        const localFilepath: string = path.join(trialLocalTempFolder, hpFileName);
        await fs.promises.writeFile(localFilepath, hyperParameters.value, { encoding: 'utf8' });
        const hdfsCodeDir: string = HDFSClientUtility.getHdfsTrialWorkDir(this.paiBaseClusterConfig.userName, trialJobId);
        const hdfsHpFilePath: string = path.join(hdfsCodeDir, hpFileName);

        await HDFSClientUtility.copyFileToHdfs(localFilepath, hdfsHpFilePath, this.hdfsClient);

        await this.postParameterFileMeta({
            experimentId: this.experimentId,
            trialId: trialJobId,
            filePath: hdfsHpFilePath
        });
    }

    protected postParameterFileMeta(parameterFileMeta: ParameterFileMeta): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();
        if (this.paiBaseJobRestServer === undefined) {
            throw new Error('paiBaseJobRestServer not implemented!');
        }
        const req: request.Options = {
            uri: `${this.paiBaseJobRestServer.endPoint}${this.paiBaseJobRestServer.apiRootUrl}/parameter-file-meta`,
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

export { PAIYarnTrainingService };
