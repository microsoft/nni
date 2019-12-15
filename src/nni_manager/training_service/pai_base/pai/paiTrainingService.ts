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
// tslint:disable-next-line:no-implicit-dependencies
import * as request from 'request';
import * as component from '../../../common/component';

import { Deferred } from 'ts-deferred';
import { String } from 'typescript-string-operations';
import {
    HyperParameters, NNIManagerIpConfig, TrainingService,
    TrialJobApplicationForm, TrialJobDetail, TrialJobMetric
} from '../../../common/trainingService';
import { delay, generateParamFileName,
    getExperimentRootDir, getIPV4Address, getVersion, uniqueString, unixPathJoin } from '../../../common/utils';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../../common/containerJobData';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { execMkdir, validateCodeDir, execCopydir } from '../../common/util';
import { PAI_LITE_TRIAL_COMMAND_FORMAT } from './paiData';
import { NNIPAITrialConfig } from './paiConfig';
import { PAIJobRestServer } from './paiJobRestServer';
import { PAIBaseTrainingService } from '../paiBaseTrainingService';
import { PAIBaseClusterConfig, PAIBaseTrialJobDetail } from '../../pai_base/paiBaseConfig';

const yaml = require('js-yaml');

/**
 * Training Service implementation for OpenPAI (Open Platform for AI)
 * Refer https://github.com/Microsoft/pai for more info about OpenPAI
 */
@component.Singleton
class PAITrainingService extends PAIBaseTrainingService {
    protected paiTrialConfig: NNIPAITrialConfig | undefined;

    constructor() {
        super();
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

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();

        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                deferred.resolve();
                break;

            case TrialConfigMetadataKey.PAI_CLUSTER_CONFIG:
                this.paiBaseClusterConfig = <PAIBaseClusterConfig>JSON.parse(value);

                if(this.paiBaseClusterConfig.passWord) {
                    // Get PAI authentication token
                    await this.updatePaiToken();
                } else if(this.paiBaseClusterConfig.token) {
                    this.paiToken = this.paiBaseClusterConfig.token;
                } else {
                    deferred.reject(new Error('pai cluster config format error, please set password or token!'));
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
                deferred.reject(new Error(`Uknown key: ${key}`));
        }

        return deferred.promise;
    }

    public async submitTrialJob(form: TrialJobApplicationForm): Promise<TrialJobDetail> {
        if (this.paiBaseClusterConfig === undefined) {
            throw new Error(`paiClusterConfig not initialized!`);
        }
        const deferred: Deferred<PAIBaseTrialJobDetail> = new Deferred<PAIBaseTrialJobDetail>();

        this.log.info(`submitTrialJob: form: ${JSON.stringify(form)}`);

        const trialJobId: string = uniqueString(5);
        //TODO: use HDFS working folder instead
        const trialWorkingFolder: string = path.join(this.expRootDir, 'trials', trialJobId);
        const paiJobName: string = `nni_exp_${this.experimentId}_trial_${trialJobId}`;
        const logPath: string = '';
        const trialJobDetail: PAIBaseTrialJobDetail = new PAIBaseTrialJobDetail(
            trialJobId,
            'WAITING',
            paiJobName,
            Date.now(),
            trialWorkingFolder,
            form,
            logPath);

        this.trialJobsMap.set(trialJobId, trialJobDetail);
        this.jobQueue.push(trialJobId);
        deferred.resolve(trialJobDetail);

        return deferred.promise;
    }

    public generateJobConfigInYamlFormat(trialJobId: string, command: string) {
        if (this.paiTrialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        const jobName = `nni_exp_${this.experimentId}_trial_${trialJobId}`
        const paiJobConfig = {
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
                        command
                    ]
                }
            },
            defaults: {
                virtualCluster: this.paiTrialConfig.virtualCluster
            },
            extras: {
                'com.microsoft.pai.runtimeplugin': [
                    {
                        plugin: this.paiTrialConfig.paiStoragePlugin
                    }
                ],
                submitFrom: 'submit-job-v2'
            }
        }
        return yaml.safeDump(paiJobConfig);
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
            const restServer: PAIJobRestServer = component.get(PAIJobRestServer);
            this.paiRestServerPort = restServer.clusterRestServerPort;
        }

        // Step 1. Prepare PAI job configuration
        const trialLocalFolder: string = path.join(this.paiTrialConfig.nniManagerNFSMountPath, this.experimentId, trialJobId);
        //create trial local working folder locally.
        await execMkdir(trialLocalFolder);

        const runScriptContent: string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local files
        await fs.promises.writeFile(path.join(trialLocalFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });

        // Write file content ( parameter.cfg ) to local working folders
        if (trialJobDetail.form !== undefined) {
            await fs.promises.writeFile(
                path.join(trialLocalFolder, generateParamFileName(trialJobDetail.form.hyperParameters)),
                trialJobDetail.form.hyperParameters.value, { encoding: 'utf8' }
            );
        }

        //Copy codeDir files to local working folder
        await execCopydir(this.paiTrialConfig.codeDir, trialLocalFolder);
        
        const nniManagerIp: string = this.nniManagerIpConfig ? this.nniManagerIpConfig.nniManagerIp : getIPV4Address();
        const version: string = this.versionCheck ? await getVersion() : '';
        const containerWorkingDir: string = `${this.paiTrialConfig.containerNFSMountPath}/${this.experimentId}/${trialJobId}`;
        const nniPaiTrialCommand: string = String.Format(
            PAI_LITE_TRIAL_COMMAND_FORMAT,
            `${containerWorkingDir}`,
            `${containerWorkingDir}/nnioutput`,
            trialJobId,
            this.experimentId,
            trialJobDetail.form.sequenceId,
            this.isMultiPhase,
            this.paiTrialConfig.command,
            nniManagerIp,
            this.paiRestServerPort,
            version,
            this.logCollection
        )
        .replace(/\r\n|\n|\r/gm, '');

        this.log.info(`nniPAItrial command is ${nniPaiTrialCommand.trim()}`);
        
        const paiJobConfig = this.generateJobConfigInYamlFormat(trialJobId, nniPaiTrialCommand);
        console.log(paiJobConfig);

        // Step 3. Submit PAI job via Rest call
        // Refer https://github.com/Microsoft/pai/blob/master/docs/rest-server/API.md for more detail about PAI Rest API
        const submitJobRequest: request.Options = {
            uri: `http://${this.paiBaseClusterConfig.host}/rest-server/api/v2/jobs`,
            method: 'POST',
            body: paiJobConfig,
            headers: {
                'Content-Type': 'text/yaml',
                Authorization: `Bearer ${this.paiToken}`
            }
        };
        request(submitJobRequest, (error: Error, response: request.Response, body: any) => {
            if ((error !== undefined && error !== null) || response.statusCode >= 400) {
                const errorMessage: string = (error !== undefined && error !== null) ? error.message :
                    `Submit trial ${trialJobId} failed, http code:${response.statusCode}, http body: ${response.body.message}`;
                this.log.error(errorMessage);
                trialJobDetail.status = 'FAILED';
            } else {
                trialJobDetail.submitTime = Date.now();
            }
            deferred.resolve(true);
        });

        return deferred.promise;
    }
}

export { PAITrainingService };
