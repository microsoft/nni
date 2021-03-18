// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import * as path from 'path';
import * as component from '../../../common/component';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import { getLogger, Logger } from '../../../common/log';
import { EnvironmentInformation, EnvironmentService } from '../environment';
import { getExperimentRootDir } from '../../../common/utils';
import { ExperimentConfig, RemoteConfig, RemoteMachineConfig, flattenConfig } from '../../../common/experimentConfig';
import { execMkdir, validateCodeDir } from '../../common/util';
import { ExecutorManager } from '../../remote_machine/remoteMachineData';
import { ShellExecutor } from 'training_service/remote_machine/shellExecutor';
import { RemoteMachineEnvironmentInformation } from '../remote/remoteConfig';
import { SharedStorageService } from '../sharedStorage'

interface FlattenRemoteConfig extends ExperimentConfig, RemoteConfig { }

@component.Singleton
export class RemoteEnvironmentService extends EnvironmentService {

    private readonly initExecutorId = "initConnection";
    private readonly machineExecutorManagerMap: Map<RemoteMachineConfig, ExecutorManager>;
    private readonly environmentExecutorManagerMap: Map<string, ExecutorManager>;
    private readonly remoteMachineMetaOccupiedMap: Map<RemoteMachineConfig, boolean>;
    private readonly log: Logger;
    private sshConnectionPromises: any[];
    private experimentRootDir: string;
    private experimentId: string;
    private config: FlattenRemoteConfig;

    constructor(config: ExperimentConfig) {
        super();
        this.experimentId = getExperimentId();
        this.environmentExecutorManagerMap = new Map<string, ExecutorManager>();
        this.machineExecutorManagerMap = new Map<RemoteMachineConfig, ExecutorManager>();
        this.remoteMachineMetaOccupiedMap = new Map<RemoteMachineConfig, boolean>();
        this.sshConnectionPromises = [];
        this.experimentRootDir = getExperimentRootDir();
        this.experimentId = getExperimentId();
        this.log = getLogger();
        this.config = flattenConfig(config, 'remote');

        // codeDir is not a valid directory, throw Error
        if (!fs.lstatSync(this.config.trialCodeDirectory).isDirectory()) {
            throw new Error(`codeDir ${this.config.trialCodeDirectory} is not a directory`);
        }

        this.sshConnectionPromises = this.config.machineList.map(
            machine => this.initRemoteMachineOnConnected(machine)
        );
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

    public get getName(): string {
        return 'remote';
    }

    private scheduleMachine(): RemoteMachineConfig | undefined {
        for (const [rmMeta, occupied] of this.remoteMachineMetaOccupiedMap) {
            if (!occupied) {
                this.remoteMachineMetaOccupiedMap.set(rmMeta, true);
                return rmMeta;
            }
        }
        return undefined;
    }

    private async initRemoteMachineOnConnected(rmMeta: RemoteMachineConfig): Promise<void> {
        const executorManager: ExecutorManager = new ExecutorManager(rmMeta);
        this.log.info(`connecting to ${rmMeta.user}@${rmMeta.host}:${rmMeta.port}`);
        const executor: ShellExecutor = await executorManager.getExecutor(this.initExecutorId);
        this.log.debug(`reached ${executor.name}`);
        this.machineExecutorManagerMap.set(rmMeta, executorManager);
        this.log.debug(`initializing ${executor.name}`);

        // Create root working directory after executor is ready
        const nniRootDir: string = executor.joinPath(executor.getTempPath(), 'nni-experiments');
        await executor.createFolder(executor.getRemoteExperimentRootDir(getExperimentId()));

        // the directory to store temp scripts in remote machine
        const remoteGpuScriptCollectorDir: string = executor.getRemoteScriptsPath(getExperimentId());

        // clean up previous result.
        await executor.createFolder(remoteGpuScriptCollectorDir, true);
        await executor.allowPermission(true, nniRootDir);
    }

    public async refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void> {	
        const tasks = environments.map(environment => this.refreshEnvironment(environment));
        await Promise.all(tasks);	
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
                this.log.info(`pid in ${remoteEnvironment.rmMachineMeta.host}:${jobpidPath} is not alive!`);
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
        // get an executor from scheduler
        const rmMachineMeta: RemoteMachineConfig | undefined = this.scheduleMachine();
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
            if (environment.useSharedStorage) {
                const environmentRoot = component.get<SharedStorageService>(SharedStorageService).remoteWorkingRoot;
                environment.runnerWorkingFolder = executor.joinPath(environmentRoot, 'envs', environment.id)
                const remoteMountCommand = component.get<SharedStorageService>(SharedStorageService).remoteMountCommand;
                await executor.executeScript(remoteMountCommand, false, false);
            } else {
                environment.runnerWorkingFolder = 
                    executor.joinPath(executor.getRemoteExperimentRootDir(getExperimentId()), 
                    'envs', environment.id)
            }
            environment.command = `cd ${environment.runnerWorkingFolder} && \
                ${environment.command} --job_pid_file ${environment.runnerWorkingFolder}/pid \
                1>${environment.runnerWorkingFolder}/trialrunner_stdout 2>${environment.runnerWorkingFolder}/trialrunner_stderr \
                && echo $? \`date +%s%3N\` >${environment.runnerWorkingFolder}/code`;
            return Promise.resolve(true);
        }
    }

    private async launchRunner(environment: RemoteMachineEnvironmentInformation): Promise<void> {
        const executor = await this.getExecutor(environment.id);
        const environmentLocalTempFolder: string =  
            path.join(this.experimentRootDir, "environment-temp")
        await executor.createFolder(environment.runnerWorkingFolder);
        await execMkdir(environmentLocalTempFolder);
        await fs.promises.writeFile(path.join(environmentLocalTempFolder, executor.getScriptName("run")),
        environment.command, { encoding: 'utf8' });
        // Copy files in codeDir to remote working directory
        await executor.copyDirectoryToRemote(environmentLocalTempFolder, environment.runnerWorkingFolder);
        // Execute command in remote machine, set isInteractive=true to run script in conda environment
        executor.executeScript(executor.joinPath(environment.runnerWorkingFolder,
            executor.getScriptName("run")), true, true);
        if (environment.rmMachineMeta === undefined) {
            throw new Error(`${environment.id} rmMachineMeta not initialized!`);
        }
        environment.trackingUrl = `file://${environment.rmMachineMeta.host}:${environment.runnerWorkingFolder}`;
    }

    private async getExecutor(environmentId: string): Promise<ShellExecutor> {
        const executorManager = this.environmentExecutorManagerMap.get(environmentId);
        if (executorManager === undefined) {
            throw new Error(`ExecutorManager is not assigned for environment ${environmentId}`);
        }
        return await executorManager.getExecutor(environmentId);
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        if (environment.isAlive === false) {
            return;
        }

        const executor = await this.getExecutor(environment.id);

        if (environment.status === 'UNKNOWN') {
            environment.status = 'USER_CANCELED';
            this.releaseEnvironmentResource(environment);
            return;
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
