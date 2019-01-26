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

import * as assert from 'assert';
import * as cpp from 'child-process-promise';
import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Client, ConnectConfig } from 'ssh2';
import { Deferred } from 'ts-deferred';
import { String } from 'typescript-string-operations';
import * as component from '../../common/component';
import { NNIError, NNIErrorNames } from '../../common/errors';
import { getExperimentId, getInitTrialSequenceId } from '../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../common/log';
import { ObservableTimer } from '../../common/observableTimer';
import {
    HostJobApplicationForm, HyperParameters, JobApplicationForm, TrainingService, TrialJobApplicationForm, TrialJobDetail, TrialJobMetric, NNIManagerIpConfig
} from '../../common/trainingService';
import { delay, generateParamFileName, getExperimentRootDir, uniqueString, getJobCancelStatus, getRemoteTmpDir,getIPV4Address  } from '../../common/utils';
import { GPUSummary } from '../common/gpuData';
import { TrialConfig } from '../common/trialConfig';
import { TrialConfigMetadataKey } from '../common/trialConfigMetadataKey';
import { GPUScheduler } from './gpuScheduler';
import {
    HOST_JOB_SHELL_FORMAT, RemoteCommandResult, RemoteMachineMeta,
    RemoteMachineScheduleInfo, RemoteMachineScheduleResult,
    RemoteMachineTrialJobDetail, ScheduleResultType, REMOTEMACHINE_TRIAL_COMMAND_FORMAT, 
    GPU_COLLECTOR_FORMAT
} from './remoteMachineData';
import { SSHClientUtility } from './sshClientUtility';
import { validateCodeDir} from '../common/util';
import { RemoteMachineJobRestServer } from './remoteMachineJobRestServer';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { mkDirP } from '../../common/utils';

/**
 * Training Service implementation for Remote Machine (Linux)
 */
@component.Singleton
class RemoteMachineTrainingService implements TrainingService {
    private machineSSHClientMap: Map<RemoteMachineMeta, Client>;
    private trialJobsMap: Map<string, RemoteMachineTrialJobDetail>;
    private expRootDir: string;
    private remoteExpRootDir: string;
    private trialConfig: TrialConfig | undefined;
    private gpuScheduler: GPUScheduler;
    private jobQueue: string[];
    private timer: ObservableTimer;
    private stopping: boolean = false;
    private metricsEmitter: EventEmitter;
    private log: Logger;
    private isMultiPhase: boolean = false;
    private disableLog: boolean = false;
    private trialSequenceId: number;
    private remoteRestServerPort?: number;
    private readonly remoteOS: string;
    private nniManagerIpConfig?: NNIManagerIpConfig;

    constructor(@component.Inject timer: ObservableTimer) {
        this.remoteOS = 'linux';
        this.metricsEmitter = new EventEmitter();
        this.trialJobsMap = new Map<string, RemoteMachineTrialJobDetail>();
        this.machineSSHClientMap = new Map<RemoteMachineMeta, Client>();
        this.gpuScheduler = new GPUScheduler(this.machineSSHClientMap);
        this.jobQueue = [];
        this.expRootDir = getExperimentRootDir();
        this.remoteExpRootDir = this.getRemoteExperimentRootDir();
        this.timer = timer;
        this.log = getLogger();
        this.trialSequenceId = -1;
    }

    /**
     * Loop to launch trial jobs and collect trial metrics
     */
    public async run(): Promise<void> {
        const restServer: RemoteMachineJobRestServer = component.get(RemoteMachineJobRestServer);
        await restServer.start();
        while (!this.stopping) {
            while (this.jobQueue.length > 0) {
                this.updateGpuReversion();
                const trialJobId: string = this.jobQueue[0];
                const prepareResult : boolean = await this.prepareTrialJob(trialJobId);
                if (prepareResult) {
                    // Remove trial job with trialJobId from job queue
                    this.jobQueue.shift();
                } else {
                    // Break the while loop since no GPU resource is available right now,
                    // Wait to schedule job in next time iteration
                    break;
                }     
            }
            await delay(3000);
        }
    }

    /**
     * List submitted trial jobs
     */
    public async listTrialJobs(): Promise<TrialJobDetail[]> {
        const jobs: TrialJobDetail[] = [];
        const deferred: Deferred<TrialJobDetail[]> = new Deferred<TrialJobDetail[]>();

        for (const [key, value] of this.trialJobsMap) { 
            if (value.form.jobType === 'TRIAL') {
                jobs.push(await this.getTrialJob(key));
            }
        };
        deferred.resolve(jobs);

        return deferred.promise;
    }

    /**
     * Get trial job detail information
     * @param trialJobId ID of trial job
     */
    public async getTrialJob(trialJobId: string): Promise<TrialJobDetail> {
        const trialJob: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (!trialJob) {
            throw new NNIError(NNIErrorNames.NOT_FOUND, `trial job id ${trialJobId} not found`);
        }
        //TO DO: add another job status, and design new job status change logic
        if (trialJob.status === 'RUNNING' || trialJob.status === 'UNKNOWN') {
            // Get ssh client where the job is running
            if (trialJob.rmMeta === undefined) {
                throw new Error(`rmMeta not set for submitted job ${trialJobId}`);
            }
            const sshClient: Client | undefined = this.machineSSHClientMap.get(trialJob.rmMeta);
            if (!sshClient) {
                throw new Error(`Invalid job id: ${trialJobId}, cannot find ssh client`);
            }

            return this.updateTrialJobStatus(trialJob, sshClient);
        } else {
            return trialJob;
        }
    }

    /**
     * Add job metrics listener
     * @param listener callback listener
     */
    public addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.on('metric', listener);
    }

    /**
     * Remove job metrics listener
     * @param listener callback listener
     */
    public removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void {
        this.metricsEmitter.off('metric', listener);
    }

    /**
     * Submit trial job
     * @param form trial job description form
     */
    public submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail> {
        this.log.info(`submitTrialJob: form: ${JSON.stringify(form)}`);

        if (!this.trialConfig) {
            throw new Error('trial config is not initialized');
        }

        if (form.jobType === 'HOST') {
            return this.runHostJob(<HostJobApplicationForm>form);
        } else if (form.jobType === 'TRIAL') {
            // Generate trial job id(random)
            const trialJobId: string = uniqueString(5);
            const trialWorkingFolder: string = path.join(this.remoteExpRootDir, 'trials', trialJobId);

            const trialJobDetail: RemoteMachineTrialJobDetail = new RemoteMachineTrialJobDetail(
                trialJobId,
                'WAITING',
                Date.now(),
                trialWorkingFolder,
                form,
                this.generateSequenceId()
            );
            this.jobQueue.push(trialJobId);
            this.trialJobsMap.set(trialJobId, trialJobDetail);

            return Promise.resolve(trialJobDetail);
        } else {
            return Promise.reject(new Error(`Job form not supported: ${JSON.stringify(form)}, jobType should be HOST or TRIAL.`));
        }
    }

    /**
     * Update trial job for multi-phase
     * @param trialJobId trial job id
     * @param form job application form
     */
    public async updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<TrialJobDetail> {
        this.log.info(`updateTrialJob: form: ${JSON.stringify(form)}`);
        const trialJobDetail: undefined | TrialJobDetail = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }
        if (form.jobType === 'TRIAL') {
            const rmMeta: RemoteMachineMeta | undefined = (<RemoteMachineTrialJobDetail>trialJobDetail).rmMeta;
            if (rmMeta !== undefined) {
                await this.writeParameterFile(trialJobId, (<TrialJobApplicationForm>form).hyperParameters, rmMeta);
            } else {
                throw new Error(`updateTrialJob failed: ${trialJobId} rmMeta not found`);
            }
        } else {
            throw new Error(`updateTrialJob failed: jobType ${form.jobType} not supported.`);
        }

        return trialJobDetail;
    }
    
    /**
     * remove gpu reversion when job is not runningj
     */
    private updateGpuReversion() {
        for (const [key, value] of this.trialJobsMap) { 
            if(!['WAITING', 'RUNNING'].includes(value.status)) {
                this.gpuScheduler.removeGpuReservation(value.id, value.rmMeta);
            }
        };
    }

    /**
     * Is multiphase job supported in current training service
     */
    public get isMultiPhaseJobSupported(): boolean {
        return true;
    }

    /**
     * Cancel trial job
     * @param trialJobId ID of trial job
     */
    public async cancelTrialJob(trialJobId: string, isEarlyStopped: boolean = false): Promise<void> {
        this.log.info(`cancelTrialJob: jobId: ${trialJobId}`);
        const deferred: Deferred<void> = new Deferred<void>();
        const trialJob: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (!trialJob) {
            deferred.reject();
            throw new Error(`trial job id ${trialJobId} not found`);
        }

        // Remove the job with trialJobId from job queue
        const index : number = this.jobQueue.indexOf(trialJobId);
        if (index >= 0) {
            this.jobQueue.splice(index, 1);
        }

        // Get ssh client where the job is running
        if (trialJob.rmMeta !== undefined) {
            // If the trial job is already scheduled, check its status and kill the trial process in remote machine
            const sshClient: Client | undefined = this.machineSSHClientMap.get(trialJob.rmMeta);
            if (!sshClient) {
                deferred.reject();
                throw new Error(`Invalid job id ${trialJobId}, cannot find ssh client`);
            }

            const jobpidPath: string = this.getJobPidPath(trialJob.id);
            try {
                await SSHClientUtility.remoteExeCommand(`pkill -P \`cat ${jobpidPath}\``, sshClient);
                trialJob.status = getJobCancelStatus(isEarlyStopped);
            } catch (error) {
                // Not handle the error since pkill failed will not impact trial job's current status
                this.log.error(`remoteTrainingService.cancelTrialJob: ${error.message}`);
            }
        } else {
            // Job is not scheduled yet, set status to 'USER_CANCELLED' directly
            assert(isEarlyStopped === false, 'isEarlyStopped is not supposed to be true here.');
            trialJob.status = getJobCancelStatus(isEarlyStopped);
        }
    }

    /**
     * Set culster metadata
     * @param key metadata key
     * //1. MACHINE_LIST -- create ssh client connect of machine list
     * //2. TRIAL_CONFIG -- trial configuration
     * @param value metadata value
     */
    public async setClusterMetadata(key: string, value: string): Promise<void> {
        switch (key) {
            case TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = <NNIManagerIpConfig>JSON.parse(value);
                break;
            case TrialConfigMetadataKey.MACHINE_LIST:
                let localGpuMetricFolder: string = await this.generateGpuScript();
                await this.setupConnections(value, localGpuMetricFolder);
                //remove local temp files
                // await cpp.exec(`rm -rf ${localGpuMetricFolder}`);
                break;
            case TrialConfigMetadataKey.TRIAL_CONFIG:
                const remoteMachineTrailConfig: TrialConfig = <TrialConfig>JSON.parse(value);
                // Parse trial config failed, throw Error
                if (!remoteMachineTrailConfig) {
                    throw new Error('trial config parsed failed');
                }
                // codeDir is not a valid directory, throw Error
                if (!fs.lstatSync(remoteMachineTrailConfig.codeDir).isDirectory()) {
                    throw new Error(`codeDir ${remoteMachineTrailConfig.codeDir} is not a directory`);
                }

                // Validate to make sure codeDir doesn't have too many files
                try {
                    await validateCodeDir(remoteMachineTrailConfig.codeDir);
                } catch(error) {
                    this.log.error(error);
                    return Promise.reject(new Error(error));                    
                }

                this.trialConfig = remoteMachineTrailConfig;
                break;
            case TrialConfigMetadataKey.MULTI_PHASE:
                this.isMultiPhase = (value === 'true' || value === 'True');
                break;
            case TrialConfigMetadataKey.DISABLE_LOG:
                this.disableLog = (value === 'true' || value === 'True');
                break;
            default:
                //Reject for unknown keys
                throw new Error(`Uknown key: ${key}`);
        }
    }

    /**
     * Get culster metadata
     * @param key metadata key
     */
    public getClusterMetadata(key: string): Promise<string> {
        const deferred: Deferred<string> = new Deferred<string>();

        return deferred.promise;
    }

    public async cleanUp(): Promise<void> {
        this.stopping = true;
        await this.cleanupConnections();
        return Promise.resolve();
    }
    
    /**
     * stop gpu_metric_collector process in remote machine and remove unused scripts
     */
    private async cleanupConnections(): Promise<void> {
        for (const [rmMeta, client] of this.machineSSHClientMap.entries()) {
            let jobpidPath: string = path.join(this.getRemoteScriptsPath(), 'pid');
            await SSHClientUtility.remoteExeCommand(`pkill -P \`cat ${jobpidPath}\``, client);
            await SSHClientUtility.remoteExeCommand(`rm -rf ${this.getRemoteScriptsPath()}`, client);
        }
        return Promise.resolve();
    } 
    
    private async generateGpuScript(): Promise<string> {
        let localGpuMetricFolder: string = `/tmp/nni/scripts/${uniqueString(5)}`;
        await cpp.exec(`mkdir -p ${localGpuMetricFolder}`);
        //generate gpu_metrics_collector.sh
        let gpuMetricFilePath: string = path.join(localGpuMetricFolder, 'gpu_metrics_collector.sh');

        const remoteScriptsDir: string = this.getRemoteScriptsPath();
        const gpuCollectorContent: string = String.Format(
            GPU_COLLECTOR_FORMAT, 
            remoteScriptsDir, 
            path.join(remoteScriptsDir, 'pid'), 
        );
        await fs.promises.writeFile(gpuMetricFilePath, gpuCollectorContent, { encoding: 'utf8' });
        return Promise.resolve(localGpuMetricFolder);
    }

    private async setupConnections(machineList: string, localGpuMetricFolder: string): Promise<void> {
        const deferred: Deferred<void> = new Deferred<void>();
        //TO DO: verify if value's format is wrong, and json parse failed, how to handle error
        const rmMetaList: RemoteMachineMeta[] = <RemoteMachineMeta[]>JSON.parse(machineList);
        let connectedRMNum: number = 0;

        rmMetaList.forEach((rmMeta: RemoteMachineMeta) => {
            const conn: Client = new Client();
            let connectConfig: ConnectConfig = {
                host: rmMeta.ip,
                port: rmMeta.port,
                username: rmMeta.username };
            if (rmMeta.passwd) {
                connectConfig.password = rmMeta.passwd;                
            } else if(rmMeta.sshKeyPath) {
                if(!fs.existsSync(rmMeta.sshKeyPath)) {
                    //SSh key path is not a valid file, reject
                    deferred.reject(new Error(`${rmMeta.sshKeyPath} does not exist.`));
                }
                const privateKey: string = fs.readFileSync(rmMeta.sshKeyPath, 'utf8');

                connectConfig.privateKey = privateKey;
                connectConfig.passphrase = rmMeta.passphrase;
            } else {
                deferred.reject(new Error(`No valid passwd or sshKeyPath is configed.`));
            }
            this.machineSSHClientMap.set(rmMeta, conn);
            conn.on('ready', async () => {
                this.machineSSHClientMap.set(rmMeta, conn);
                let gpuMetricFilePath: string = path.join(localGpuMetricFolder, 'gpu_metrics_collector.sh');
                await this.initRemoteMachineOnConnected(gpuMetricFilePath, rmMeta, conn);
                if (++connectedRMNum === rmMetaList.length) {
                    deferred.resolve();
                }
            }).on('error', (err: Error) => {
                // SSH connection error, reject with error message
                deferred.reject(new Error(err.message));
            }).connect(connectConfig);
        });
        return deferred.promise;
    }

    private async initRemoteMachineOnConnected(gpuMetricFilePath: string, rmMeta: RemoteMachineMeta, conn: Client): Promise<void> {
        // Create root working directory after ssh connection is ready
        //TO DO: Should we mk experiments rootDir here?
        const nniRootDir: string = '/tmp/nni';
        await SSHClientUtility.remoteExeCommand(`mkdir -p ${this.remoteExpRootDir}`, conn);

        // Copy NNI scripts to remote expeirment working directory
        const remoteScriptsDir: string = this.getRemoteScriptsPath();
        await SSHClientUtility.remoteExeCommand(`mkdir -p ${remoteScriptsDir}`, conn);
        await SSHClientUtility.remoteExeCommand(`chmod 777 ${nniRootDir} ${nniRootDir}/* ${nniRootDir}/scripts/*`, conn);
        //copy gpu_metrics_collector.sh to remote
        await SSHClientUtility.copyFileToRemote(gpuMetricFilePath, path.join(remoteScriptsDir, 'gpu_metrics_collector.sh'), conn);
        
        //Begin to execute gpu_metrics_collection scripts
        SSHClientUtility.remoteExeCommand(`bash ${path.join(remoteScriptsDir, 'gpu_metrics_collector.sh')}`, conn);

        this.timer.subscribe(
            async (tick: number) => {
                const cmdresult: RemoteCommandResult = await SSHClientUtility.remoteExeCommand(
                    `tail -n 1 ${path.join(remoteScriptsDir, 'gpu_metrics')}`, conn);
                if (cmdresult && cmdresult.stdout) {
                    rmMeta.gpuSummary = <GPUSummary>JSON.parse(cmdresult.stdout);
                }
            }
        );
    }

    private async prepareTrialJob(trialJobId: string): Promise<boolean> {
        const deferred : Deferred<boolean> = new Deferred<boolean>();

        if (!this.trialConfig) {
            throw new Error('trial config is not initialized');
        }
        const trialJobDetail: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new NNIError(NNIErrorNames.INVALID_JOB_DETAIL, `Invalid job detail information for trial job ${trialJobId}`);
        }
        // get an ssh client from scheduler
        const rmScheduleResult: RemoteMachineScheduleResult = this.gpuScheduler.scheduleMachine(this.trialConfig.gpuNum, trialJobId);
        if (rmScheduleResult.resultType === ScheduleResultType.REQUIRE_EXCEED_TOTAL) {
            const errorMessage : string = `Required GPU number ${this.trialConfig.gpuNum} is too large, no machine can meet`;
            this.log.error(errorMessage);
            deferred.reject();
            throw new NNIError(NNIErrorNames.RESOURCE_NOT_AVAILABLE, errorMessage);
        } else if (rmScheduleResult.resultType === ScheduleResultType.SUCCEED
            && rmScheduleResult.scheduleInfo !== undefined) {
            const rmScheduleInfo : RemoteMachineScheduleInfo = rmScheduleResult.scheduleInfo;
            const trialWorkingFolder: string = path.join(this.remoteExpRootDir, 'trials', trialJobId);
            await this.launchTrialOnScheduledMachine(
                trialJobId, trialWorkingFolder, <TrialJobApplicationForm>trialJobDetail.form, rmScheduleInfo);

            trialJobDetail.status = 'RUNNING';
            trialJobDetail.url = `file://${rmScheduleInfo.rmMeta.ip}:${trialWorkingFolder}`;
            trialJobDetail.startTime = Date.now();
            trialJobDetail.rmMeta = rmScheduleInfo.rmMeta;

            deferred.resolve(true);
        } else if (rmScheduleResult.resultType === ScheduleResultType.TMP_NO_AVAILABLE_GPU) {
            this.log.info(`Right now no available GPU can be allocated for trial ${trialJobId}, will try to schedule later`);
            deferred.resolve(false);
        } else {
            deferred.reject(`Invalid schedule resutl type: ${rmScheduleResult.resultType}`);
        }

        return deferred.promise;
    }

    private async launchTrialOnScheduledMachine(trialJobId: string, trialWorkingFolder: string, form: TrialJobApplicationForm,
                                                rmScheduleInfo: RemoteMachineScheduleInfo): Promise<void> {
        if (!this.trialConfig) {
            throw new Error('trial config is not initialized');
        }
        const cuda_visible_device: string = rmScheduleInfo.cuda_visible_device;
        const sshClient: Client | undefined = this.machineSSHClientMap.get(rmScheduleInfo.rmMeta);
        if (sshClient === undefined) {
            assert(false, 'sshClient is undefined.');

            // for lint
            return;
        }
        const trialJobDetail: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`Can not get trial job detail for job: ${trialJobId}`);
        }

        const trialLocalTempFolder: string = path.join(this.expRootDir, 'trials-local', trialJobId);

        await SSHClientUtility.remoteExeCommand(`mkdir -p ${trialWorkingFolder}`, sshClient);
        await SSHClientUtility.remoteExeCommand(`mkdir -p ${path.join(trialWorkingFolder, '.nni')}`, sshClient);

        await SSHClientUtility.remoteExeCommand(`touch ${path.join(trialWorkingFolder, '.nni', 'metrics')}`, sshClient);
        // RemoteMachineRunShellFormat is the run shell format string,
        // See definition in remoteMachineData.ts

        let command: string;
        // Set CUDA_VISIBLE_DEVICES environment variable based on cuda_visible_device
        // If no valid cuda_visible_device is defined, set CUDA_VISIBLE_DEVICES to empty string to hide GPU device
        if(typeof cuda_visible_device === 'string' && cuda_visible_device.length > 0) {
            command = `CUDA_VISIBLE_DEVICES=${cuda_visible_device} ${this.trialConfig.command}`;
        } else {
            command = `CUDA_VISIBLE_DEVICES=" " ${this.trialConfig.command}`;
        }
        
        const nniManagerIp = this.nniManagerIpConfig?this.nniManagerIpConfig.nniManagerIp:getIPV4Address();
        if(!this.remoteRestServerPort) {
            const restServer: RemoteMachineJobRestServer = component.get(RemoteMachineJobRestServer);
            this.remoteRestServerPort = restServer.clusterRestServerPort;
        }
        const runScriptTrialContent: string = String.Format(
            REMOTEMACHINE_TRIAL_COMMAND_FORMAT,
            trialWorkingFolder,
            trialWorkingFolder,
            trialJobId,
            getExperimentId(),
            trialJobDetail.sequenceId.toString(),
            this.isMultiPhase,
            path.join(trialWorkingFolder, '.nni', 'jobpid'),
            command,
            nniManagerIp,
            this.remoteRestServerPort,
            this.disableLog? '--disable_log' : '',
            path.join(trialWorkingFolder, '.nni', 'code')
        )

        //create tmp trial working folder locally.
        await cpp.exec(`mkdir -p ${path.join(trialLocalTempFolder, '.nni')}`);

        //create tmp trial working folder locally.
        await cpp.exec(`cp -r ${this.trialConfig.codeDir}/* ${trialLocalTempFolder}`);
        const installScriptContent : string = CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        // Write NNI installation file to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'install_nni.sh'), installScriptContent, { encoding: 'utf8' });
        // Write file content ( run.sh and parameter.cfg ) to local tmp files
        await fs.promises.writeFile(path.join(trialLocalTempFolder, 'run.sh'), runScriptTrialContent, { encoding: 'utf8' });
        await this.writeParameterFile(trialJobId, form.hyperParameters, rmScheduleInfo.rmMeta);
        // Copy files in codeDir to remote working directory
        await SSHClientUtility.copyDirectoryToRemote(trialLocalTempFolder, trialWorkingFolder, sshClient, this.remoteOS);
        // Execute command in remote machine
        SSHClientUtility.remoteExeCommand(`bash ${path.join(trialWorkingFolder, 'run.sh')}`, sshClient);
    }

    private async runHostJob(form: HostJobApplicationForm): Promise<TrialJobDetail> {
        const rmMeta: RemoteMachineMeta = this.getRmMetaByHost(form.host);
        const sshClient: Client | undefined = this.machineSSHClientMap.get(rmMeta);
        if (sshClient === undefined) {
            throw new Error('sshClient not found.');
        }
        const jobId: string = uniqueString(5);
        const localDir: string = path.join(this.expRootDir, 'hostjobs-local', jobId);
        const remoteDir: string = this.getHostJobRemoteDir(jobId);
        await cpp.exec(`mkdir -p ${localDir}`);
        await SSHClientUtility.remoteExeCommand(`mkdir -p ${remoteDir}`, sshClient);
        const runScriptContent: string = String.Format(
            HOST_JOB_SHELL_FORMAT, remoteDir, path.join(remoteDir, 'jobpid'), form.cmd, path.join(remoteDir, 'code')
        );
        await fs.promises.writeFile(path.join(localDir, 'run.sh'), runScriptContent, { encoding: 'utf8' });
        await SSHClientUtility.copyFileToRemote(
            path.join(localDir, 'run.sh'), path.join(remoteDir, 'run.sh'), sshClient);
        SSHClientUtility.remoteExeCommand(`bash ${path.join(remoteDir, 'run.sh')}`, sshClient);

        const jobDetail: RemoteMachineTrialJobDetail =  new RemoteMachineTrialJobDetail(
            jobId, 'RUNNING', Date.now(), remoteDir, form, this.generateSequenceId()
        );
        jobDetail.rmMeta = rmMeta;
        jobDetail.startTime = Date.now();
        this.trialJobsMap.set(jobId, jobDetail);
        this.log.debug(`runHostJob: return: ${JSON.stringify(jobDetail)} `);

        return jobDetail;
    }

    private getRmMetaByHost(host: string): RemoteMachineMeta {
        for (const [rmMeta, client] of this.machineSSHClientMap.entries()) {
            if (rmMeta.ip === host) {
                return rmMeta;
            }
        }
        throw new Error(`Host not found: ${host}`);
    }

    private async updateTrialJobStatus(trialJob: RemoteMachineTrialJobDetail, sshClient: Client): Promise<TrialJobDetail> {
        const deferred: Deferred<TrialJobDetail> = new Deferred<TrialJobDetail>();
        const jobpidPath: string = this.getJobPidPath(trialJob.id);
        const trialReturnCodeFilePath: string = path.join(this.remoteExpRootDir, 'trials', trialJob.id, '.nni', 'code');
        try {
            const killResult: number = (await SSHClientUtility.remoteExeCommand(`kill -0 \`cat ${jobpidPath}\``, sshClient)).exitCode;
            // if the process of jobpid is not alive any more
            if (killResult !== 0) {
                const trailReturnCode: string = await SSHClientUtility.getRemoteFileContent(trialReturnCodeFilePath, sshClient);
                this.log.debug(`trailjob ${trialJob.id} return code: ${trailReturnCode}`);
                const match: RegExpMatchArray | null = trailReturnCode.trim().match(/^(\d+)\s+(\d+)$/);
                if (match) {
                    const { 1: code, 2: timestamp } = match;
                    // Update trial job's status based on result code
                    if (parseInt(code, 10) === 0) {
                        trialJob.status = 'SUCCEEDED';
                    } else {
                        trialJob.status = 'FAILED';
                    }
                    trialJob.endTime = parseInt(timestamp, 10);
                }
                this.log.info(`trailJob status update: ${trialJob.id}, ${trialJob.status}`);
            }
            deferred.resolve(trialJob);
        } catch (error) {
            this.log.error(`Update job status exception, error is ${error.message}`);
            if (error instanceof NNIError && error.name === NNIErrorNames.NOT_FOUND) {
                deferred.resolve(trialJob);
            } else {
                trialJob.status = 'UNKNOWN';
                deferred.resolve(trialJob);
            }
        }
        return deferred.promise;
    }

    private getRemoteScriptsPath(): string {
        return path.join(getRemoteTmpDir(this.remoteOS), 'nni', 'scripts');
    }

    private getHostJobRemoteDir(jobId: string): string {
        return path.join(this.remoteExpRootDir, 'hostjobs', jobId);
    }

    private getRemoteExperimentRootDir(): string{
        return path.join(getRemoteTmpDir(this.remoteOS), 'nni', 'experiments', getExperimentId());
    }

    public get MetricsEmitter() : EventEmitter {
        return this.metricsEmitter;
    }

    private getJobPidPath(jobId: string): string {
        const trialJobDetail: RemoteMachineTrialJobDetail | undefined = this.trialJobsMap.get(jobId);
        if (trialJobDetail === undefined) {
            throw new NNIError(NNIErrorNames.INVALID_JOB_DETAIL, `Invalid job detail information for trial job ${jobId}`);
        }

        let jobpidPath: string;
        if (trialJobDetail.form.jobType === 'TRIAL') {
            jobpidPath = path.join(trialJobDetail.workingDirectory, '.nni', 'jobpid');
        } else if (trialJobDetail.form.jobType === 'HOST') {
            jobpidPath = path.join(this.getHostJobRemoteDir(jobId), 'jobpid');
        } else {
            throw new Error(`Job type not supported: ${trialJobDetail.form.jobType}`);
        }

        return jobpidPath;
    }

    private async writeParameterFile(trialJobId: string, hyperParameters: HyperParameters, rmMeta: RemoteMachineMeta): Promise<void> {
        const sshClient: Client | undefined = this.machineSSHClientMap.get(rmMeta);
        if (sshClient === undefined) {
            throw new Error('sshClient is undefined.');
        }

        const trialWorkingFolder: string = path.join(this.remoteExpRootDir, 'trials', trialJobId);
        const trialLocalTempFolder: string = path.join(this.expRootDir, 'trials-local', trialJobId);

        const fileName: string = generateParamFileName(hyperParameters);
        const localFilepath: string = path.join(trialLocalTempFolder, fileName);
        await fs.promises.writeFile(localFilepath, hyperParameters.value, { encoding: 'utf8' });

        await SSHClientUtility.copyFileToRemote(localFilepath, path.join(trialWorkingFolder, fileName), sshClient);
    }

    private generateSequenceId(): number {
        if (this.trialSequenceId === -1) {
            this.trialSequenceId = getInitTrialSequenceId();
        }

        return this.trialSequenceId++;
    }
}

export { RemoteMachineTrainingService };
