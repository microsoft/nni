// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import fs from 'fs';
import path from 'path';
import { IocShim } from 'common/ioc_shim';
import { getLogger, Logger } from 'common/log';
import { EnvironmentInformation, EnvironmentService } from '../environment';
import { getLogLevel } from 'common/utils';
import { RemoteConfig, RemoteMachineConfig } from 'common/experimentConfig';
import { ExperimentStartupInfo } from 'common/experimentStartupInfo';
import { execMkdir } from 'training_service/common/util';
import { ExecutorManager } from 'training_service/remote_machine/remoteMachineData';
import { ShellExecutor } from 'training_service/remote_machine/shellExecutor';
import { RemoteMachineEnvironmentInformation } from '../remote/remoteConfig';
import { SharedStorageService } from '../sharedStorage';
import { createScriptFile } from 'common/shellUtils';

export class RemoteEnvironmentService extends EnvironmentService {

    private readonly initExecutorId = "initConnection";
    private readonly machineExecutorManagerMap: Map<RemoteMachineConfig, ExecutorManager>;
    private readonly environmentExecutorManagerMap: Map<string, ExecutorManager>;
    private readonly remoteMachineMetaOccupiedMap: Map<RemoteMachineConfig, boolean>;
    private readonly log: Logger;
    private sshConnectionPromises: Promise<void[]>;
    private experimentRootDir: string;
    private remoteExperimentRootDir: string = "";
    private experimentId: string;
    private config: RemoteConfig;

    constructor(config: RemoteConfig, info: ExperimentStartupInfo) {
        super();
        this.experimentId = info.experimentId;
        this.environmentExecutorManagerMap = new Map<string, ExecutorManager>();
        this.machineExecutorManagerMap = new Map<RemoteMachineConfig, ExecutorManager>();
        this.remoteMachineMetaOccupiedMap = new Map<RemoteMachineConfig, boolean>();
        this.experimentRootDir = info.logDir;
        this.log = getLogger('RemoteEnvironmentService');
        this.config = config;

        // codeDir is not a valid directory, throw Error
        if (!fs.lstatSync(this.config.trialCodeDirectory).isDirectory()) {
            throw new Error(`codeDir ${this.config.trialCodeDirectory} is not a directory`);
        }

        this.sshConnectionPromises = Promise.all(this.config.machineList.map(
            machine => this.initRemoteMachineOnConnected(machine)
        ));
    }

    public async init(): Promise<void> {
        await this.sshConnectionPromises;
        this.log.info('ssh connection initialized!');
        Array.from(this.machineExecutorManagerMap.keys()).forEach(rmMeta => {
            // initialize remoteMachineMetaOccupiedMap, false means not occupied
            this.remoteMachineMetaOccupiedMap.set(rmMeta, false);
        });
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
        await executor.createFolder(executor.getRemoteExperimentRootDir(this.experimentId));

        // the directory to store temp scripts in remote machine
        const remoteGpuScriptCollectorDir: string = executor.getRemoteScriptsPath(this.experimentId);

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
                        await this.releaseEnvironmentResource(environment);
                    }
                }
            }
        } catch (error) {
            this.log.error(`Update job status exception, error is ${(error as any).message}`);
        }
    }

    /**
     * If a environment is finished, release the connection resource
     * @param environment remote machine environment job detail
     */
    private async releaseEnvironmentResource(environment: EnvironmentInformation): Promise<void> {
        if (environment.useSharedStorage) {
            const executor = await this.getExecutor(environment.id);
            const remoteUmountCommand = IocShim.get<SharedStorageService>(SharedStorageService).remoteUmountCommand;
            const result = await executor.executeScript(remoteUmountCommand, false, false);
            if (result.exitCode !== 0) {
                this.log.error(`Umount shared storage on remote machine failed.\n ERROR: ${result.stderr}`);
            }
        }

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

    private async getScript(environment: EnvironmentInformation): Promise<string> {
        const executor = await this.getExecutor(environment.id);
        const isDebug = getLogLevel() == "debug";
        let script: string = environment.command;
        environment.runnerWorkingFolder = executor.joinPath(this.remoteExperimentRootDir, 'envs', environment.id);

        let codeScript = `echo $? \`date +%s%3N\` >${environment.runnerWorkingFolder}/code`;
        if (executor.isWindows) {
            const prepare = `mkdir envs\\${environment.id} 2>NUL & cd envs\\${environment.id}`;
            const startrun = `powershell ..\\install_nni.ps1 && python -m nni.tools.trial_tool.trial_runner`;
            const developingScript = "IF EXIST nni_trial_tool (ECHO \"nni_trial_tool exists already\") ELSE (mkdir nni_trial_tool && tar -xof ../nni_trial_tool.tar.gz -C ./nni_trial_tool) && pip3 install websockets";

            script = isDebug ? `${prepare} && ${developingScript} && ${startrun}` : `${prepare} && ${startrun}`;
            codeScript = `powershell -command "Write $? " " (((New-TimeSpan -Start (Get-Date "01/01/1970") -End (Get-Date).ToUniversalTime()).TotalMilliseconds).ToString("0")) | Out-file ${path.join(environment.runnerWorkingFolder, 'code')} -Append -NoNewline -encoding utf8"`;
        }

        script = `cd ${this.remoteExperimentRootDir} && \
            ${script} --job_pid_file ${environment.runnerWorkingFolder}/pid \
            1>${environment.runnerWorkingFolder}/trialrunner_stdout 2>${environment.runnerWorkingFolder}/trialrunner_stderr \
            && ${codeScript}`;

        return script;
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
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
                this.remoteExperimentRootDir = IocShim.get<SharedStorageService>(SharedStorageService).remoteWorkingRoot;
                if (!this.remoteExperimentRootDir.startsWith('/')) {
                    this.remoteExperimentRootDir = executor.joinPath((await executor.getCurrentPath()).trim(), this.remoteExperimentRootDir);
                }
                const remoteMountCommand = IocShim.get<SharedStorageService>(SharedStorageService).remoteMountCommand.replace(/echo -e /g, `echo `).replace(/echo /g, `echo -e `).replace(/\\\$/g, `\\\\\\$`);
                const result = await executor.executeScript(remoteMountCommand, false, false);
                if (result.exitCode !== 0) {
                    throw new Error(`Mount shared storage on remote machine failed.\n ERROR: ${result.stderr}`);
                }
            } else {
                this.remoteExperimentRootDir = executor.getRemoteExperimentRootDir(this.experimentId);
            }

            environment.command = await this.getScript(environment);
            environment.useActiveGpu = rmMachineMeta.useActiveGpu;
            return Promise.resolve(true);
        }
    }

    private async launchRunner(environment: RemoteMachineEnvironmentInformation): Promise<void> {
        const executor = await this.getExecutor(environment.id);
        const environmentLocalTempFolder: string =  
            path.join(this.experimentRootDir, "environment-temp")
        await executor.createFolder(environment.runnerWorkingFolder);
        await execMkdir(environmentLocalTempFolder);
        await createScriptFile(path.join(environmentLocalTempFolder, executor.getScriptName("run")),
                environment.command);
        // Copy files in codeDir to remote working directory
        await executor.copyDirectoryToRemote(environmentLocalTempFolder, this.remoteExperimentRootDir);
        // Execute command in remote machine, set isInteractive=true to run script in conda environment
        executor.executeScript(executor.joinPath(this.remoteExperimentRootDir,
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
            await this.releaseEnvironmentResource(environment);
            return;
        }

        const jobpidPath: string = `${environment.runnerWorkingFolder}/pid`;
        try {
            await executor.killChildProcesses(jobpidPath);
            await this.releaseEnvironmentResource(environment);
        } catch (error) {
            this.log.error(`stopEnvironment: ${error}`);
        }
    }
}
