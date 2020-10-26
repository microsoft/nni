// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as path from 'path';
import * as component from '../../../common/component';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../../common/log';
import { EnvironmentInformation, EnvironmentService } from '../environment';
import {
    getExperimentRootDir,
} from '../../../common/utils';
import { TrialConfig } from '../../common/trialConfig';
import { TrialConfigMetadataKey } from '../../common/trialConfigMetadataKey';
import { execMkdir, validateCodeDir } from '../../common/util';
import {
    ExecutorManager, RemoteMachineMeta,
} from '../../remote_machine/remoteMachineData';
import { ShellExecutor } from 'training_service/remote_machine/shellExecutor';
import { RemoteMachineEnvironmentInformation } from '../remote/remoteConfig';


@component.Singleton
export class RemoteEnvironmentService extends EnvironmentService {

    private readonly initExecutorId = "initConnection";
    private readonly machineExecutorManagerMap: Map<RemoteMachineMeta, ExecutorManager>;
    private readonly environmentExecutorManagerMap: Map<string, ExecutorManager>;
    private readonly remoteMachineMetaOccupiedMap: Map<RemoteMachineMeta, boolean>;
    private trialConfig: TrialConfig | undefined;
    private readonly log: Logger;
    private sshConnectionPromises: any[];
    private experimentRootDir: string;
    private experimentId: string;

    constructor() {
        super();
        this.experimentId = getExperimentId();
        this.environmentExecutorManagerMap = new Map<string, ExecutorManager>();
        this.machineExecutorManagerMap = new Map<RemoteMachineMeta, ExecutorManager>();
        this.remoteMachineMetaOccupiedMap = new Map<RemoteMachineMeta, boolean>();
        this.sshConnectionPromises = [];
        this.experimentRootDir = getExperimentRootDir();
        this.experimentId = getExperimentId();
        this.log = getLogger();
    }

    public get prefetchedEnvironmentCount(): number {
        return this.machineExecutorManagerMap.size;
    }

    public get environmentMaintenceLoopInterval(): number {
        return 5000;
    }

    public get hasMoreEnvironments(): boolean {
        return false;
    }

    public get hasStorageService(): boolean {
        return false;
    }

    public async config(key: string, value: string): Promise<void> {
        switch (key) {
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
            default:
                this.log.debug(`Remote not support metadata key: '${key}', value: '${value}'`);
        }
    }

    private scheduleMachine(): RemoteMachineMeta | undefined {
        for (const [rmMeta, occupied] of this.remoteMachineMetaOccupiedMap) {
            if (!occupied) {
                this.remoteMachineMetaOccupiedMap.set(rmMeta, true);
                return rmMeta;
            }
        }
        return undefined;
    }

    private async setupConnections(machineList: string): Promise<void> {
        this.log.debug(`Connecting to remote machines: ${machineList}`);
        //TO DO: verify if value's format is wrong, and json parse failed, how to handle error
        const rmMetaList: RemoteMachineMeta[] = <RemoteMachineMeta[]>JSON.parse(machineList);

        for (const rmMeta of rmMetaList) {
            this.sshConnectionPromises.push(await this.initRemoteMachineOnConnected(rmMeta));
        }
    }

    private async initRemoteMachineOnConnected(rmMeta: RemoteMachineMeta): Promise<void> {
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
    }
    
    private async refreshEnvironment(environment: EnvironmentInformation): Promise<void> {
        const executor = await this.getExecutor(environment.id);
        const jobpidPath: string = `${environment.runnerWorkingFolder}/pid`;
        const runnerReturnCodeFilePath: string = `${environment.runnerWorkingFolder}/code`;
        /* eslint-disable require-atomic-updates */
        try {
            // check if pid file exist
            const pidExist = await executor.fileExist(jobpidPath);
            if (!pidExist) {
                return;
            }
            const isAlive = await executor.isProcessAlive(jobpidPath);
            environment.status = 'RUNNING';
            // if the process of jobpid is not alive any more
            if (!isAlive) {
                const remoteEnvironment: RemoteMachineEnvironmentInformation = environment as RemoteMachineEnvironmentInformation;
                if (remoteEnvironment.rmMachineMeta === undefined) {
                    throw new Error(`${remoteEnvironment.id} machine meta not initialized!`);
                }
                this.log.info(`pid in ${remoteEnvironment.rmMachineMeta.ip}:${jobpidPath} is not alive!`);
                if (fs.existsSync(runnerReturnCodeFilePath)) {
                    const runnerReturnCode: string = await executor.getRemoteFileContent(runnerReturnCodeFilePath);
                    const match: RegExpMatchArray | null = runnerReturnCode.trim()
                        .match(/^-?(\d+)\s+(\d+)$/);
                    if (match !== null) {
                        const { 1: code } = match;
                        // Update trial job's status based on result code
                        if (parseInt(code, 10) === 0) {
                            environment.setStatus('SUCCEEDED');
                        } else {
                            environment.setStatus('FAILED');
                        }
                        this.releaseEnvironmentResource(environment);
                    }
                }
            }
        } catch (error) {
            this.log.error(`Update job status exception, error is ${error.message}`);
        }
    }

    public async refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void> {
        const tasks: Promise<void>[] = [];
        environments.forEach(async (environment) => {
            tasks.push(this.refreshEnvironment(environment));
        });
        await Promise.all(tasks);
    }

    /**
     * If a environment is finished, release the connection resource
     * @param environment remote machine environment job detail
     */
    private releaseEnvironmentResource(environment: EnvironmentInformation): void {
        const executorManager = this.environmentExecutorManagerMap.get(environment.id);
        if (executorManager === undefined) {
            throw new Error(`ExecutorManager is not assigned for environment ${environment.id}`);
        }

        // Note, it still keep reference in trialExecutorManagerMap, as there may be following requests from nni manager.
        executorManager.releaseExecutor(environment.id);
        const remoteEnvironment: RemoteMachineEnvironmentInformation = environment as RemoteMachineEnvironmentInformation;
        if (remoteEnvironment.rmMachineMeta === undefined) {
            throw new Error(`${remoteEnvironment.id} rmMachineMeta not initialized!`);
        }
        this.remoteMachineMetaOccupiedMap.set(remoteEnvironment.rmMachineMeta, false);
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
        if (this.sshConnectionPromises.length > 0) {
            await Promise.all(this.sshConnectionPromises);
            this.log.info('ssh connection initialized!');
            // set sshConnectionPromises to [] to avoid log information duplicated
            this.sshConnectionPromises = [];
            if (this.trialConfig ===  undefined) {
                throw new Error("trial config not initialized!");
            }
            Array.from(this.machineExecutorManagerMap.keys()).forEach(rmMeta => {
                // initialize remoteMachineMetaOccupiedMap, false means not occupied
                this.remoteMachineMetaOccupiedMap.set(rmMeta, false);
            });
        }
        const remoteEnvironment: RemoteMachineEnvironmentInformation = environment as RemoteMachineEnvironmentInformation;
        remoteEnvironment.status = 'WAITING';
        // schedule machine for environment, generate command
        await this.prepareEnvironment(remoteEnvironment);
        // launch runner process in machine
        await this.launchRunner(environment);
    }

    private async prepareEnvironment(environment: RemoteMachineEnvironmentInformation): Promise<boolean> {
        if (this.trialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }

        // get an executor from scheduler
        const rmMachineMeta: RemoteMachineMeta | undefined = this.scheduleMachine();
        if (rmMachineMeta === undefined) {
            this.log.warning(`No available machine!`);
            return Promise.resolve(false);
        } else {
            environment.rmMachineMeta = rmMachineMeta;
            const executorManager: ExecutorManager | undefined = this.machineExecutorManagerMap.get(environment.rmMachineMeta);
            if (executorManager === undefined) {
                throw new Error(`executorManager not initialized`);
            }
            this.environmentExecutorManagerMap.set(environment.id, executorManager);
            const executor = await this.getExecutor(environment.id);
            environment.runnerWorkingFolder = 
                executor.joinPath(executor.getRemoteExperimentRootDir(getExperimentId()), 
                'envs', environment.id)
            environment.command = `cd ${environment.runnerWorkingFolder} && \
${environment.command} --job_pid_file ${environment.runnerWorkingFolder}/pid \
1>${environment.runnerWorkingFolder}/trialrunner_stdout 2>${environment.runnerWorkingFolder}/trialrunner_stderr \
&& echo $? \`date +%s%3N\` >${environment.runnerWorkingFolder}/code`;
            return Promise.resolve(true);
        }
    }

    private async launchRunner(environment: RemoteMachineEnvironmentInformation): Promise<void> {
        if (this.trialConfig === undefined) {
            throw new Error('trial config is not initialized');
        }
        const executor = await this.getExecutor(environment.id);
        const environmentLocalTempFolder: string =  
            path.join(this.experimentRootDir, this.experimentId, "environment-temp")
        await executor.createFolder(environment.runnerWorkingFolder);
        await execMkdir(environmentLocalTempFolder);
        await fs.promises.writeFile(path.join(environmentLocalTempFolder, executor.getScriptName("run")),
        environment.command, { encoding: 'utf8' });
        // Copy files in codeDir to remote working directory
        await executor.copyDirectoryToRemote(environmentLocalTempFolder, environment.runnerWorkingFolder);
        // Execute command in remote machine
        executor.executeScript(executor.joinPath(environment.runnerWorkingFolder,
            executor.getScriptName("run")), true, false);
        if (environment.rmMachineMeta === undefined) {
            throw new Error(`${environment.id} rmMachineMeta not initialized!`);
        }
        environment.trackingUrl = `file://${environment.rmMachineMeta.ip}:${environment.runnerWorkingFolder}`;
    }

    private async getExecutor(environmentId: string): Promise<ShellExecutor> {
        const executorManager = this.environmentExecutorManagerMap.get(environmentId);
        if (executorManager === undefined) {
            throw new Error(`ExecutorManager is not assigned for environment ${environmentId}`);
        }
        return await executorManager.getExecutor(environmentId);
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        const executor = await this.getExecutor(environment.id);

        if (environment.status === 'UNKNOWN') {
            environment.status = 'USER_CANCELED';
            this.releaseEnvironmentResource(environment);
            return
        }

        const jobpidPath: string = `${environment.runnerWorkingFolder}/pid`;
        try {
            await executor.killChildProcesses(jobpidPath);
            this.releaseEnvironmentResource(environment);
        } catch (error) {
            this.log.error(`stopEnvironment: ${error}`);
        }
    }
}
