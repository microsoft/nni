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
import { EnvironmentInformation, EnvironmentService } from '../environment';
import { StorageService } from '../storageService';
import { NNIError, NNIErrorNames, MethodNotImplementedError } from '../../../common/errors';
import { ObservableTimer } from '../../../common/observableTimer';
import {
    HyperParameters, NNIManagerIpConfig, TrainingService, TrialJobApplicationForm,
    TrialJobDetail, TrialJobMetric, LogType
} from '../../../common/trainingService';
import {
    delay, generateParamFileName, getExperimentRootDir, getIPV4Address, getJobCancelStatus,
    getVersion, uniqueString
} from '../../../common/utils';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../../common/containerJobData';
import { GPUSummary, ScheduleResultType } from '../../common/gpuData';
import { TrialConfig } from '../../common/trialConfig';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { execMkdir, validateCodeDir } from '../../common/util';
import { GPUScheduler } from '../../remote_machine/gpuScheduler';
import {
    ExecutorManager, RemoteMachineMeta,
    RemoteMachineScheduleInfo, RemoteMachineScheduleResult, RemoteMachineTrialJobDetail
} from '../../remote_machine/remoteMachineData';
import { RemoteMachineJobRestServer } from '../../remote_machine/remoteMachineJobRestServer';


@component.Singleton
export class RemoteEnvironmentService extends EnvironmentService {

    private readonly initExecutorId = "initConnection";
    private readonly machineExecutorManagerMap: Map<RemoteMachineMeta, ExecutorManager>; //machine excutor map
    private readonly machineCopyExpCodeDirPromiseMap: Map<RemoteMachineMeta, Promise<void>>;
    private readonly trialExecutorManagerMap: Map<string, ExecutorManager>; //trial excutor map
    private readonly trialJobsMap: Map<string, RemoteMachineTrialJobDetail>;
    private readonly expRootDir: string;
    private trialConfig: TrialConfig | undefined;
    private gpuScheduler?: GPUScheduler;
    private readonly jobQueue: string[];
    private readonly timer: ObservableTimer;
    private stopping: boolean = false;
    private readonly metricsEmitter: EventEmitter;
    private readonly log: Logger;
    private isMultiPhase: boolean = false;
    private remoteRestServerPort?: number;
    private nniManagerIpConfig?: NNIManagerIpConfig;
    private versionCheck: boolean = true;
    private logCollection: string;
    private sshConnectionPromises: any[];

    private readonly log: Logger = getLogger();

    private experimentId: string;

    constructor() {
        super();
        this.experimentId = getExperimentId();
    }

    public get environmentMaintenceLoopInterval(): number {
        return 5000;
    }

    public get hasStorageService(): boolean {
        return false;
    }

    /**
     * Set culster metadata
     * @param key metadata key
     * //1. MACHINE_LIST -- create executor of machine list
     * //2. TRIAL_CONFIG -- trial configuration
     * @param value metadata value
     */
    public async config(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;
            case TrialConfigMetadataKey.MACHINE_LIST:
                await this.setupConnections(value);
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG: {
                const remoteMachineTrailConfig: TrialConfig = <TrialConfig>JSON.parse(value);
                // Parse trial config failed, throw Error
                if (remoteMachineTrailConfig === undefined) {
                    throw new Error('trial config parsed failed');
                }
                // codeDir is not a valid directory, throw Error
                if (!fs.lstatSync(remoteMachineTrailConfig.codeDir)
                    .isDirectory()) {
                    throw new Error(`codeDir ${remoteMachineTrailConfig.codeDir} is not a directory`);
                }

                try {
                    // Validate to make sure codeDir doesn't have too many files
                    await validateCodeDir(remoteMachineTrailConfig.codeDir);
                } catch (error) {
                    this.log.error(error);
                    return Promise.reject(new Error(error));
                }

                this.trialConfig = remoteMachineTrailConfig;
                break;
            }
            case TrialConfigMetadataKey.MULTI_PHASE:
                this.isMultiPhase = (value === 'true' || value === 'True');
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
    }

    private async setupConnections(machineList: string): Promise<void> {
        this.log.debug(`Connecting to remote machines: ${machineList}`);
        //TO DO: verify if value's format is wrong, and json parse failed, how to handle error
        const rmMetaList: RemoteMachineMeta[] = <RemoteMachineMeta[]>JSON.parse(machineList);

        for (const rmMeta of rmMetaList) {
            this.sshConnectionPromises.push(this.initRemoteMachineOnConnected(rmMeta));
        }
    }

    private async initRemoteMachineOnConnected(rmMeta: RemoteMachineMeta): Promise<void> {
        rmMeta.occupiedGpuIndexMap = new Map<number, number>();
        const executorManager: ExecutorManager = new ExecutorManager(rmMeta);
        this.log.info(`connecting to ${rmMeta.username}@${rmMeta.ip}:${rmMeta.port}`);
        const executor: ShellExecutor = await executorManager.getExecutor(this.initExecutorId);
        this.log.debug(`reached ${executor.name}`);
        this.machineExecutorManagerMap.set(rmMeta, executorManager);
        this.log.debug(`initializing ${executor.name}`);

        // Create root working directory after executor is ready
        const nniRootDir: string = executor.joinPath(executor.getTempPath(), 'nni');
        await executor.createFolder(executor.getRemoteExperimentRootDir(getExperimentId()));

        // the directory to store temp scripts in remote machine
        const remoteGpuScriptCollectorDir: string = executor.getRemoteScriptsPath(getExperimentId());

        // clean up previous result.
        await executor.createFolder(remoteGpuScriptCollectorDir, true);
        await executor.allowPermission(true, nniRootDir);

        //Begin to execute gpu_metrics_collection scripts
        const script = executor.generateGpuStatsScript(getExperimentId());
        executor.executeScript(script, false, true);
        // the timer is trigger in 1 second, it causes multiple runs on server.
        // So reduce it's freqeunce, only allow one of it run.
        const collectingCount: boolean[] = [];

        const disposable: Rx.IDisposable = this.timer.subscribe(
            async () => {
                if (collectingCount.length == 0) {
                    collectingCount.push(true);
                    const cmdresult = await executor.readLastLines(executor.joinPath(remoteGpuScriptCollectorDir, 'gpu_metrics'));
                    if (cmdresult !== "") {
                        rmMeta.gpuSummary = <GPUSummary>JSON.parse(cmdresult);
                        if (rmMeta.gpuSummary.gpuCount === 0) {
                            this.log.warning(`No GPU found on remote machine ${rmMeta.ip}`);
                            this.timer.unsubscribe(disposable);
                        }
                    }
                    if (this.stopping) {
                        this.timer.unsubscribe(disposable);
                        this.log.debug(`Stopped GPU collector on ${rmMeta.ip}, since experiment is exiting.`);
                    }
                    collectingCount.pop();
                }
            }
        );
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
