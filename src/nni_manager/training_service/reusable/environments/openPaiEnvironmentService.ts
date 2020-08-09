// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as yaml from 'js-yaml';
import * as request from 'request';
import { Deferred } from 'ts-deferred';
import * as component from '../../../common/component';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../../common/log';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { PAIClusterConfig } from '../../pai/paiConfig';
import { NNIPAIK8STrialConfig } from '../../pai/paiK8S/paiK8SConfig';
import { EnvironmentInformation, EnvironmentService } from '../environment';
import { StorageService } from '../storageService';


/**
 * Collector PAI jobs info from PAI cluster, and update pai job status locally
 */
@component.Singleton
export class OpenPaiEnvironmentService extends EnvironmentService {

    private readonly log: Logger = getLogger();
    private paiClusterConfig: PAIClusterConfig | undefined;
    private paiTrialConfig: NNIPAIK8STrialConfig | undefined;
    private paiJobConfig: any;
    private paiToken?: string;
    private protocol: string = 'http';

    private experimentId: string;

    constructor() {
        super();
        this.experimentId = getExperimentId();
    }

    public get environmentMaintenceLoopInterval(): number {
        return 5000;
    }

    public get hasStorageService(): boolean {
        return true;
    }

    public async config(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.PAI_CLUSTER_CONFIG:
                this.paiClusterConfig = <PAIClusterConfig>JSON.parse(value);
                this.paiClusterConfig.host = this.formatPAIHost(this.paiClusterConfig.host);
                this.paiToken = this.paiClusterConfig.token;
                break;

            case TrialConfigMetadataKey.TRIAL_CONFIG: {
                if (this.paiClusterConfig === undefined) {
                    this.log.error('pai cluster config is not initialized');
                    break;
                }
                this.paiTrialConfig = <NNIPAIK8STrialConfig>JSON.parse(value);
                // Validate to make sure codeDir doesn't have too many files

                const storageService = component.get<StorageService>(StorageService);
                const remoteRoot = storageService.joinPath(this.paiTrialConfig.nniManagerNFSMountPath, this.experimentId);
                storageService.initialize(this.paiTrialConfig.nniManagerNFSMountPath, remoteRoot);

                if (this.paiTrialConfig.paiConfigPath) {
                    this.paiJobConfig = yaml.safeLoad(fs.readFileSync(this.paiTrialConfig.paiConfigPath, 'utf8'));
                }

                if (this.paiClusterConfig.gpuNum === undefined) {
                    this.paiClusterConfig.gpuNum = this.paiTrialConfig.gpuNum;
                }
                if (this.paiClusterConfig.cpuNum === undefined) {
                    this.paiClusterConfig.cpuNum = this.paiTrialConfig.cpuNum;
                }
                if (this.paiClusterConfig.memoryMB === undefined) {
                    this.paiClusterConfig.memoryMB = this.paiTrialConfig.memoryMB;
                }
                break;
            }
            default:
                this.log.debug(`OpenPAI not proccessed metadata key: '${key}', value: '${value}'`);
        }
    }

    public async refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();

        if (this.paiClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (this.paiToken === undefined) {
            throw new Error('PAI token is not initialized');
        }

        const getJobInfoRequest: request.Options = {
            uri: `${this.protocol}://${this.paiClusterConfig.host}/rest-server/api/v2/jobs?username=${this.paiClusterConfig.userName}`,
            method: 'GET',
            json: true,
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${this.paiToken}`
            }
        };

        request(getJobInfoRequest, async (error: any, response: request.Response, body: any) => {
            // Status code 200 for success
            if ((error !== undefined && error !== null) || response.statusCode >= 400) {
                const errorMessage: string = (error !== undefined && error !== null) ? error.message :
                    `OpenPAI: get environment list from PAI Cluster failed!, http code:${response.statusCode}, http body: ${JSON.stringify(body)}`;
                this.log.error(`${errorMessage}`);
                deferred.reject(errorMessage);
            } else {
                const jobInfos = new Map<string, any>();
                body.forEach((jobInfo: any) => {
                    jobInfos.set(jobInfo.name, jobInfo);
                });

                environments.forEach((environment) => {
                    if (jobInfos.has(environment.envId)) {
                        const jobResponse = jobInfos.get(environment.envId);
                        if (jobResponse && jobResponse.state) {
                            const oldEnvironmentStatus = environment.status;
                            switch (jobResponse.state) {
                                case 'RUNNING':
                                case 'WAITING':
                                case 'SUCCEEDED':
                                    environment.setStatus(jobResponse.state);
                                    break;
                                case 'FAILED':
                                    environment.setStatus(jobResponse.state);
                                    deferred.reject(`OpenPAI: job ${environment.envId} is failed!`);
                                    break;
                                case 'STOPPED':
                                case 'STOPPING':
                                    environment.setStatus('USER_CANCELED');
                                    break;
                                default:
                                    this.log.error(`OpenPAI: job ${environment.envId} returns unknown state ${jobResponse.state}.`);
                                    environment.setStatus('UNKNOWN');
                            }
                            if (oldEnvironmentStatus !== environment.status) {
                                this.log.debug(`OpenPAI: job ${environment.envId} change status ${oldEnvironmentStatus} to ${environment.status} due to job is ${jobResponse.state}.`)
                            }
                        } else {
                            this.log.error(`OpenPAI: job ${environment.envId} has no state returned. body:${JSON.stringify(jobResponse)}`);
                            // some error happens, and mark this environment
                            environment.status = 'FAILED';
                        }
                    } else {
                        this.log.error(`OpenPAI job ${environment.envId} is not found in job list.`);
                        environment.status = 'UNKNOWN';
                    }
                });
                deferred.resolve();
            }
        });
        return deferred.promise;
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();

        if (this.paiClusterConfig === undefined) {
            throw new Error('PAI Cluster config is not initialized');
        }
        if (this.paiToken === undefined) {
            throw new Error('PAI token is not initialized');
        }
        if (this.paiTrialConfig === undefined) {
            throw new Error('PAI trial config is not initialized');
        }

        // Step 1. Prepare PAI job configuration
        const environmentRoot = `${this.paiTrialConfig.containerNFSMountPath}/${this.experimentId}`;
        environment.runnerWorkingFolder = `${environmentRoot}/envs/${environment.id}`;
        environment.command = `cd ${environmentRoot} && ${environment.command}`;
        environment.trackingUrl = `${this.protocol}://${this.paiClusterConfig.host}/job-detail.html?username=${this.paiClusterConfig.userName}&jobName=${environment.envId}`;
        environment.useActiveGpu = this.paiClusterConfig.useActiveGpu;
        environment.maxTrialNumberPerGpu = this.paiClusterConfig.maxTrialNumPerGpu;

        // Step 2. Generate Job Configuration in yaml format
        const paiJobConfig = this.generateJobConfigInYamlFormat(environment);
        this.log.debug(`generated paiJobConfig: ${paiJobConfig}`);

        // Step 3. Submit PAI job via Rest call
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
        request(submitJobRequest, (error, response, body) => {
            // Status code 202 for success, refer https://github.com/microsoft/pai/blob/master/src/rest-server/docs/swagger.yaml
            if ((error !== undefined && error !== null) || response.statusCode >= 400) {
                const errorMessage: string = (error !== undefined && error !== null) ? error.message :
                    `start environment ${environment.envId} failed, http code:${response.statusCode}, http body: ${body}`;

                this.log.error(errorMessage);
                environment.status = 'FAILED';
                deferred.reject(errorMessage);
            }
            deferred.resolve();
        });

        return deferred.promise;
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();

        if (this.paiClusterConfig === undefined) {
            return Promise.reject(new Error('PAI Cluster config is not initialized'));
        }
        if (this.paiToken === undefined) {
            return Promise.reject(Error('PAI token is not initialized'));
        }

        const stopJobRequest: request.Options = {
            uri: `${this.protocol}://${this.paiClusterConfig.host}/rest-server/api/v2/jobs/${this.paiClusterConfig.userName}~${environment.envId}/executionType`,
            method: 'PUT',
            json: true,
            body: { value: 'STOP' },
            time: true,
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${this.paiToken}`
            }
        };

        this.log.debug(`stopping OpenPAI environment ${environment.envId}, ${stopJobRequest.uri}`);

        try {
            request(stopJobRequest, (error, response, _body) => {
                try {
                    // Status code 202 for success.
                    if ((error !== undefined && error !== null) || (response && response.statusCode >= 400)) {
                        const errorMessage: string = (error !== undefined && error !== null) ? error.message :
                            `OpenPAI: stop job ${environment.envId} failed, http code:${response.statusCode}, http body: ${_body}`;
                        this.log.error(`${errorMessage}`);
                        deferred.reject((error !== undefined && error !== null) ? error :
                            `Stop trial failed, http code: ${response.statusCode}`);
                    } else {
                        this.log.info(`OpenPAI job ${environment.envId} stopped.`);
                    }
                    deferred.resolve();
                } catch (error) {
                    this.log.error(`OpenPAI error when inner stopping environment ${error}`);
                    deferred.reject(error);
                }
            });
        } catch (error) {
            this.log.error(`OpenPAI error when stopping environment ${error}`);
            return Promise.reject(error);
        }

        return deferred.promise;
    }

    private generateJobConfigInYamlFormat(environment: EnvironmentInformation): any {
        if (this.paiTrialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        const jobName = environment.envId;

        let nniJobConfig: any = undefined;
        if (this.paiTrialConfig.paiConfigPath) {
            nniJobConfig = JSON.parse(JSON.stringify(this.paiJobConfig)); //Trick for deep clone in Typescript
            nniJobConfig.name = jobName;
            if (nniJobConfig.taskRoles) {

                environment.nodeCount = 0;
                // count instance
                for (const taskRoleName in nniJobConfig.taskRoles) {
                    const taskRole = nniJobConfig.taskRoles[taskRoleName];
                    let instanceCount = 1;
                    if (taskRole.instances) {
                        instanceCount = taskRole.instances;
                    }
                    environment.nodeCount += instanceCount;
                }

                // Each taskRole will generate new command in NNI's command format
                // Each command will be formatted to NNI style
                for (const taskRoleName in nniJobConfig.taskRoles) {
                    const taskRole = nniJobConfig.taskRoles[taskRoleName];
                    // replace ' to '\''
                    const joinedCommand = taskRole.commands.join(" && ").replace("'", "'\\''").trim();
                    const nniTrialCommand = `${environment.command} --node_count ${environment.nodeCount} --trial_command '${joinedCommand}'`;
                    this.log.debug(`replace command ${taskRole.commands} to ${[nniTrialCommand]}`);
                    taskRole.commands = [nniTrialCommand];
                }
            }

        } else {
            if (this.paiClusterConfig === undefined) {
                throw new Error('PAI Cluster config is not initialized');
            }
            if (this.paiClusterConfig.gpuNum === undefined) {
                throw new Error('PAI Cluster gpuNum is not initialized');
            }
            if (this.paiClusterConfig.cpuNum === undefined) {
                throw new Error('PAI Cluster cpuNum is not initialized');
            }
            if (this.paiClusterConfig.memoryMB === undefined) {
                throw new Error('PAI Cluster memoryMB is not initialized');
            }

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
                            gpu: this.paiClusterConfig.gpuNum,
                            cpu: this.paiClusterConfig.cpuNum,
                            memoryMB: this.paiClusterConfig.memoryMB
                        },
                        commands: [
                            environment.command
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
}
