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

import * as fs from 'fs';
import * as path from 'path';
// tslint:disable-next-line:no-implicit-dependencies
import * as request from 'request';
import * as component from '../../../common/component';

import { Deferred } from 'ts-deferred';
import { String } from 'typescript-string-operations';
import {
    HyperParameters, NNIManagerIpConfig,
    TrialJobApplicationForm, TrialJobDetail
} from '../../../common/trainingService';
import {
    generateParamFileName,
    getIPV4Address, getVersion, uniqueString
} from '../../../common/utils';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../../common/containerJobData';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { execMkdir, validateCodeDir, execCopydir } from '../../common/util';
import { PAI_K8S_TRIAL_COMMAND_FORMAT } from './paiK8SData';
import { NNIPAIK8STrialConfig } from './paiK8SConfig';
import { PAITrainingService } from '../paiTrainingService';
import { PAIClusterConfig, PAITrialJobDetail } from '../paiConfig';
import { PAIJobRestServer } from '../paiJobRestServer';

const yaml = require('js-yaml');

/**
 * Training Service implementation for OpenPAI (Open Platform for AI)
 * Refer https://github.com/Microsoft/pai for more info about OpenPAI
 */
@component.Singleton
class PAIK8STrainingService extends PAITrainingService {
    protected paiTrialConfig: NNIPAIK8STrialConfig | undefined;
    private paiJobConfig: undefined;
    private nniVersion: string | undefined;
    constructor() {
        super();

    }

    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;

            case TrialConfigMetadataKey.PAI_CLUSTER_CONFIG:
                this.paiJobRestServer = new PAIJobRestServer(component.get(PAIK8STrainingService));
                this.paiClusterConfig = <PAIClusterConfig>JSON.parse(value);
                this.paiClusterConfig.host = this.formatPAIHost(this.paiClusterConfig.host);
                if (this.paiClusterConfig.passWord) {
                    // Get PAI authentication token
                    await this.updatePaiToken();
                } else if (this.paiClusterConfig.token) {
                    this.paiToken = this.paiClusterConfig.token;
                }
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG:
                if (this.paiClusterConfig === undefined) {
                    this.log.error('pai cluster config is not initialized');
                    break;
                }
                this.paiTrialConfig = <NNIPAIK8STrialConfig>JSON.parse(value);
                // Validate to make sure codeDir doesn't have too many files
                await validateCodeDir(this.paiTrialConfig.codeDir);
                if (this.paiTrialConfig.paiConfigPath) {
                    this.paiJobConfig = yaml.safeLoad(fs.readFileSync(this.paiTrialConfig.paiConfigPath, 'utf8'));
                }
                break;
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
        const trialJobDetail: PAITrialJobDetail = new PAITrialJobDetail(
            trialJobId,
            'WAITING',
            paiJobName,
            Date.now(),
            trialWorkingFolder,
            form,
            logPath);

        this.trialJobsMap.set(trialJobId, trialJobDetail);
        this.jobQueue.push(trialJobId);

        return trialJobDetail;
    }

    private generateNNITrialCommand(trialJobDetail: PAITrialJobDetail, command: string): string {
        if (this.paiTrialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        const containerWorkingDir: string = `${this.paiTrialConfig.containerNFSMountPath}/${this.experimentId}/${trialJobDetail.id}`;
        const nniManagerIp: string = this.nniManagerIpConfig ? this.nniManagerIpConfig.nniManagerIp : getIPV4Address();
        const nniPaiTrialCommand: string = String.Format(
            PAI_K8S_TRIAL_COMMAND_FORMAT,
            `${containerWorkingDir}`,
            `${containerWorkingDir}/nnioutput`,
            trialJobDetail.id,
            this.experimentId,
            trialJobDetail.form.sequenceId,
            this.isMultiPhase,
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
            nniJobConfig = this.paiJobConfig;
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
                    'com.microsoft.pai.runtimeplugin': [
                        {
                            plugin: this.paiTrialConfig.paiStoragePlugin
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

        this.paiRestServerPort = this.paiJobRestServer.clusterRestServerPort;

        // Step 1. Prepare PAI job configuration
        //create trial local working folder locally.
        await execMkdir(trialJobDetail.logPath);

        const runScriptContent: string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local files
        await fs.promises.writeFile(path.join(trialJobDetail.logPath, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });

        // Write file content ( parameter.cfg ) to local working folders
        if (trialJobDetail.form !== undefined) {
            await this.writeParameterFile(trialJobDetail.logPath, trialJobDetail.form.hyperParameters);
        }

        //Copy codeDir files to local working folder
        await execCopydir(this.paiTrialConfig.codeDir, trialJobDetail.logPath);
        //Generate Job Configuration in yaml format
        const paiJobConfig = this.generateJobConfigInYamlFormat(trialJobDetail);
        this.log.debug(paiJobConfig);
        // Step 3. Submit PAI job via Rest call
        // Refer https://github.com/Microsoft/pai/blob/master/docs/rest-server/API.md for more detail about PAI Rest API
        const submitJobRequest: request.Options = {
            uri: `${this.protocol}://${this.paiClusterConfig.host}/rest-server/api/v2/jobs`,
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
                    `Submit trial ${trialJobId} failed, http code:${response.statusCode}, http body: ${body}`;

                this.log.error(errorMessage);
                trialJobDetail.status = 'FAILED';
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

export { PAIK8STrainingService };
