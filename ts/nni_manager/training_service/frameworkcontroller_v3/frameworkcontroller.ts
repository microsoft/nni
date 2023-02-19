// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import { Logger, getLogger } from 'common/log';
import type { FrameworkControllerConfig, TrainingServiceConfig } from 'common/experimentConfig';
import type { EnvironmentInfo, Metric, Parameter, TrainingServiceV3 } from 'common/training_service_v3';
import { ExperimentStartupInfo, getExperimentId, getBasePort } from 'common/experimentStartupInfo';
import { delay, getExperimentRootDir, uniqueString, getIPV4Address, getVersion } from 'common/utils';
import { GPU_INFO, INITIALIZED, KILL_TRIAL_JOB, NEW_TRIAL_JOB, REPORT_METRIC_DATA, STDOUT, TRIAL_END, VERSION_CHECK } from 'core/commands';

import { validateCodeDir, tarAdd, tarExtract } from '../common/util';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT } from '../common/containerJobData';
import { CONTAINER_INSTALL_NNI_SHELL_FORMAT_FOR_WIN } from '../common/containerJobData';
import { FrameworkControllerJobStatus, FrameworkControllerJobCompleteStatus } from '../kubernetes/frameworkcontroller/frameworkcontrollerConfig';

import type { Command, CommandChannel } from './commandChannel';
import { Channel, EnvironmentInformation, RunnerSettings } from './environment';
import { FrameworkControllerEnvironmentService } from './environmentService';

import { assert } from 'console';

export class FrameworkControllerTSV3 implements TrainingServiceV3 {
    private config: FrameworkControllerConfig;
    private log: Logger;
    private commandEmitter: EventEmitter;
    private commandChannel: CommandChannel | undefined;
    private envService: FrameworkControllerEnvironmentService | undefined;
    // TODO: use EnvironmentInfo instead of EnvironmentInformation
    private envs: Map<string, EnvironmentInformation>;
    private stopping: boolean = false;
    private envManageLoopPromise: Promise<void> | undefined;
    private trialToEnv: Map<string, EnvironmentInformation>;
    private parameters: [string, Parameter][];

    constructor(trainingServiceId: string, config: TrainingServiceConfig) {
        this.log = getLogger(`FrameworkControllerV3.${trainingServiceId}`);
        this.log.info('Training sevice config:', config);

        this.config = config as FrameworkControllerConfig;
        this.envs = new Map<string, EnvironmentInformation>();
        this.trialToEnv = new Map<string, EnvironmentInformation>();
        this.parameters = [];
        this.commandEmitter = new EventEmitter();
    }

    /**
     *  Invoked during experiment initialization.
     *
     *  It should verify the config and raise error if the specified training resource is not available.
     *
     *  It should not start daemon on worker machine.
     *  If another component of the experiment failed to initialize,
     *  the process will abort without invoking any clean up function.
     **/
    public async init(): Promise<void> {
        // validate storage config
        assert(this.config.storage !== undefined, 'Storage config should be set');
        assert(this.config.storage.storageType === 'nfs',
            'FrameworkController training service v3 only supports NFS storage for now');
        assert(this.config.storage.server !== undefined, 'NFS server should be set');
        assert(this.config.storage.path !== undefined, 'NFS path should be set');
        assert(this.config.storage.localMountPath !== undefined, 'Local mount path should be set');

        await validateCodeDir(this.config.trialCodeDirectory);

        const startupInfo = ExperimentStartupInfo.getInstance();
        this.envService = new FrameworkControllerEnvironmentService(this.config, startupInfo);
        this.commandEmitter.on("command", (command: Command): void => {
            this.handleCommand(command).catch((error: Error) => {
                this.log.error(`Error on handling env ${command.environment.id} command: ${command.command},
                                data: ${command.data}, error: ${error}`);
            })
        });
        this.envService.initCommandChannel(this.commandEmitter);
        this.commandChannel = this.envService.getCommandChannel;
        return;
    }

    /**
     * Copy user's trial code directory to <local-exp-path>/<exp-id>/code-copy/
     */
    private async copyTrialCodeToTempDir(codeCopyDir: string): Promise<void> {
        fs.mkdirSync(codeCopyDir);
        const codeDir = path.resolve(this.config.trialCodeDirectory);
        await tarAdd(path.join(codeCopyDir, "trialcode.tar.gz"), codeDir);
        await fs.promises.writeFile(path.join(codeCopyDir, "install_nni.sh"), CONTAINER_INSTALL_NNI_SHELL_FORMAT);
        await fs.promises.writeFile(path.join(codeCopyDir, "install_nni.ps1"), CONTAINER_INSTALL_NNI_SHELL_FORMAT_FOR_WIN);
        return Promise.resolve();
    }

    /**
     * Copy files from <local-exp-path>/<exp-id>/code-copy/ to <shared-storage-local-mount-path>/<exp-id>/
     */
    private async copyTrialCodeToSharedStorage(sourceDir: string, sharedExpPath: string): Promise<void> {
        fs.mkdirSync(sharedExpPath, { recursive: true });
        fs.copyFileSync(path.join(sourceDir, 'trialcode.tar.gz'), path.join(sharedExpPath, 'trialcode.tar.gz'),
                        fs.constants.COPYFILE_EXCL);
        fs.copyFileSync(path.join(sourceDir, 'install_nni.sh'), path.join(sharedExpPath, 'install_nni.sh'),
                        fs.constants.COPYFILE_EXCL);
        fs.copyFileSync(path.join(sourceDir, 'install_nni.ps1'), path.join(sharedExpPath, 'install_nni.ps1'),
                        fs.constants.COPYFILE_EXCL);
        return Promise.resolve();
    }

    /**
     * Untar trial code to <container mount path>/nni/<exp-id>/envs/<env-id>/code/
     * Write settings.json to <container mount path>/nni/<exp-id>/envs/<env-id>/
     * 
     * Here, the files are prepared to the <local mount path>/..., corresponding to the <container mount path> in container.
     */
    private async prepareFilesToEnvironment(envId: string, sharedExpPath: string, envSettings: RunnerSettings): Promise<void> {
        // untar trial code
        const envPath = path.join(sharedExpPath, 'envs', envId);
        fs.mkdirSync(envPath, { recursive: true });
        // chmode to 777 because otherwise job container cannot create files in the folder
        fs.chmodSync(envPath, 0o777);
        const targetPath = path.join(envPath, 'code');
        fs.mkdirSync(targetPath);
        await tarExtract(path.join(sharedExpPath, 'trialcode.tar.gz'), targetPath);
        // chmod to 0o777 because this is trial's working directory, trial may download data to this folder
        fs.chmodSync(targetPath, 0o777);
        // write settings.json
        await fs.promises.writeFile(path.join(envPath, "settings.json"), JSON.stringify(envSettings));
        return Promise.resolve();
    }

    /**
     * @param trialId trial's id
     * @param data data is a string of json format, it is passed through to tuner/strategy
     */
    private async handleMetricData(trialId: string, data: any): Promise<void> {
        if (Array.isArray(data)) {
            for (const subItem of data) {
                this.commandEmitter.emit('metric', trialId, subItem);
            }
        } else {
            this.commandEmitter.emit('metric', trialId, data);
        }
    }

    private async handleStdout(commandData: any): Promise<void> {
        this.log.debug(`Trial stdout: ${commandData["msg"]}`);
        const metricPattern: RegExp = /NNISDK_MEb'(?<metrics>.*a?)'$/gm;
        try {
            if (commandData["tag"] === 'trial' && commandData["msg"] !== undefined) {
                const message: string = commandData["msg"];
                let metricsContent = metricPattern.exec(message);
                while (metricsContent && metricsContent.groups) {
                    const key: string = 'metrics';
                    const data = metricsContent.groups[key];
                    await this.handleMetricData(commandData["trial"], data);
                    metricsContent = metricPattern.exec(message);
                }
            }
        } catch (err) {
            this.log.error(`TrialDispatcher: handleStdout error: ${err}`);
        }
    }

    private async handleCommand(command: Command): Promise<void> {
        this.log.debug(`TrialDispatcher: env ${command.environment.id} received command ${command.command}.`);
        switch (command.command) {
            case REPORT_METRIC_DATA:
                // TODO: refactor this part, in the current implementation metric data is passed through STDOUT
                this.log.warning(`Unexpected command: ${REPORT_METRIC_DATA}.`);
                break;
            case STDOUT:
                await this.handleStdout(command.data);
                break;
            case INITIALIZED:
                this.log.debug(`Init command: ${INITIALIZED}.`);
                const env = this.envs.get(command.environment.id);
                if (env === undefined) {
                    throw new Error(`Environment ${command.environment.id} not found.`);
                }
                env.isRunnerReady = true;
                // this environment is ready
                this.commandEmitter.emit('env_status_change', command.environment.id);
                break;
            case VERSION_CHECK:
                this.log.debug(`Version command: ${VERSION_CHECK}.`);
                break;
            case GPU_INFO:
                this.log.debug(`Gpu command: ${GPU_INFO}.`);
                break;
            case TRIAL_END:
                // command.data: { "code": retCode, "time": end_time, "trial": self.id }
                this.log.debug(`Trial end command: ${TRIAL_END}.`);
                this.commandEmitter.emit('trial_end', command.environment.id,
                                         command.data['trial'], command.data['code'], command.data['end_time']);
                break;
        }
    }

    private async environmentSetting(channelName: Channel, platform: string): Promise<RunnerSettings> {
        const envSettings: RunnerSettings = new RunnerSettings();
        envSettings.nniManagerIP = this.config.nniManagerIp === undefined? await getIPV4Address() : this.config.nniManagerIp;
        // FIXME: why we need another port?
        envSettings.nniManagerPort = getBasePort() + 1;
        envSettings.commandChannel = channelName;
        // trialCommand might be empty, in which case trialCommand is specified in taskRoles
        // FIXME: rethink the relation between config.trialCommand and the command(s) in trainingservice config,
        // maybe we should not provide such flexibility.
        envSettings.command = this.config.taskRoles.map((taskRole) => taskRole.command);
        envSettings.nniManagerVersion = await getVersion();
        // FIXME: check if it is a valid log level
        envSettings.logCollection = 'none';
        envSettings.platform = platform;
        envSettings.experimentId = getExperimentId();
        envSettings.enableGpuCollector = this.config.taskRoles.map((taskRole) => taskRole.gpuNumber > 0)
                                                              .reduce((res, useGpu) => res || useGpu, false);
        return Promise.resolve(envSettings);
    }

    private async environmentManagementLoop(): Promise<void> {
        if (this.envService === undefined) {
            throw new Error('Environment service is not initialized, please call init() first');
        }
        while (!this.stopping) {
            for (const env of this.envs.values()) {
                const k8sJobInfo = await (this.envService as any).getK8sJobInfo(env);
                if (k8sJobInfo.status && k8sJobInfo.status.state) {
                    this.log.debug(`k8sJobInfo.status.state: ${k8sJobInfo.status.state}`);
                    const frameworkJobType: FrameworkControllerJobStatus = <FrameworkControllerJobStatus>k8sJobInfo.status.state;
                    // The status is defined here:
                    // https://github.com/microsoft/frameworkcontroller/blob/master/pkg/apis/frameworkcontroller/v1/types.go#L490
                    switch (frameworkJobType) {
                        case 'AttemptCreationPending':
                        case 'AttemptCreationRequested':
                        case 'AttemptPreparing':
                            env.setStatus('WAITING');
                            break;
                        case 'AttemptRunning':
                            if (env.status !== 'RUNNING') {
                                env.setStatus('RUNNING');
                                // Still need to emit here, because this status change may happen after runner initialized is ready,
                                // due to the interval of status check.
                                this.commandEmitter.emit('env_status_change', env.id);
                            }
                            break;
                        case 'AttemptDeletionPending':
                        case 'AttemptDeletionRequested':
                        case 'AttemptDeleting':
                        case 'AttemptCompleted':
                            this.log.info(`Environment ${env.id} is in ${frameworkJobType} state`);
                            break;
                        case  'Completed': {
                            const completedJobType: FrameworkControllerJobCompleteStatus =
                            <FrameworkControllerJobCompleteStatus>k8sJobInfo.status.attemptStatus.completionStatus.type.name;
                            switch (completedJobType) {
                                case 'Succeeded':
                                    env.setStatus('SUCCEEDED');
                                    break;
                                case 'Failed':
                                    env.setStatus('FAILED');
                                    break;
                                default:
                                    this.log.warning(`Environment ${env.id} is in ${completedJobType} state`);
                                    env.setStatus('UNKNOWN');
                            }
                            this.commandEmitter.emit('env_status_change', env.id);
                            break;
                        }
                        default:
                            this.log.warning(`Environment ${env.id} is in ${frameworkJobType} state`);
                            env.setStatus('UNKNOWN');
                    }
                }
            }
            // TODO: remove dead environments
            await delay(2000); // 2s
        }
        this.log.info('EnvironmentManagementLoop done');
        return Promise.resolve();
    }

    /**
     * The files that should be provided to an environment are:
     *      1. install_nni.sh and install_nni.ps1
     *      2. trial code
     *      3. settings.json: the settings for trial runner
     * These files are placed in the provided storage (i.e., nfs, ...),
     * specifically, the uncompressed trial code and settings.json are placed 
     * under <container's storage mount path>/nni/<exp-id>/envs/<env-id>/,
     * install_nni.sh and install_nni.ps1 are placed under <container's storage mount path>/nni/<exp-id>/.
     */
    private async createEnvironment(sharedExpPath: string): Promise<EnvironmentInformation> {
        // TODO: do I need to check experiment stopped?
        if (this.commandChannel === undefined|| this.envService === undefined) {
            throw new Error('Environment service is not initialized, please call init() first');
        }
        const envId = uniqueString(5);
        this.log.info(`Creating a new environment ${envId}...`);
        const envName = `nni_exp_${getExperimentId()}_env_${envId}`;
        const env = this.envService.createEnvironmentInformation(envId, envName);
        const envSettings = await this.environmentSetting(this.commandChannel.channelName, this.envService.getName);
        // Upload the related files from local to the nfs folder of the to-be-created environment
        await this.prepareFilesToEnvironment(envId, sharedExpPath, envSettings);

        // The job command (i.e., trial_runner) that replaces user trial command.
        const expWorkingDir = `${this.envService.getContainerMountPath}/nni/${getExperimentId()}`;
        // The while clause in the command is for dealing with container auto restart
        // FIXME: does env need the field 'command'?
        env.command = `cd ${expWorkingDir} && sh ./install_nni.sh `
                    + `&& cd envs/${envId} `
                    + `&& i=0; while [ -e "trialrunner_stdout$\{i\}" -o -e "trialrunner_stderr$\{i\}" ]; `
                    + `do i=$((i+1)); done; `
                    + `python3 -m nni.tools.trial_tool.trial_runnerv3 `
                    + `1>${expWorkingDir}/envs/${envId}/trialrunner_stdout$\{i\} `
                    + `2>${expWorkingDir}/envs/${envId}/trialrunner_stderr$\{i\}`;
        const fullCommand = (this.envService as any).generatePortAndCommand(env.command);
        const envPath = path.join(sharedExpPath, 'envs', envId);
        await fs.promises.writeFile(path.join(envPath, `${envId}_run.sh`), fullCommand, { encoding: 'utf8' });

        const fcJobConfig = await this.envService.startEnvironment(env);
        // Dump the k8s job config for debug
        await fs.promises.writeFile(path.join(envPath, 'job_k8s_config.yaml'), JSON.stringify(fcJobConfig), { encoding: 'utf8' });
        this.commandChannel.open(env);

        return env;
    }

    /**
     *  Invoked after init().
     *  It is suggested to resolve the promise after all daemons initialized.
     *  
     *  Return an empty list of environment because the environment is not ready at this point.
     **/
    public async start(): Promise<EnvironmentInformation[]> {
        if (this.envService === undefined || this.commandChannel === undefined) {
            throw new Error('Environment service or command channel is not initialized, please call init() first');
        }
        this.envService.init(); // does nothing
        await this.commandChannel.start();
        await this.commandChannel.run(); // does nothing
        this.log.info(`TrialDispatcher: started channel: ${this.commandChannel.constructor.name}`);

        // Copy files (install_nni.sh/ps1 and trial code) to local experiment folder
        const codeCopyDir = path.join(getExperimentRootDir(), "code-copy");
        await this.copyTrialCodeToTempDir(codeCopyDir);

        // Copy the .tar.gz trial code from the local experiment folder to the shared storage.
        // Copy to <shared-storage-local-mount-path>/nni/exp-id/xxx.tar.gz
        if (this.config.storage.localMountPath === undefined) {
            throw new Error('localMountPath is not set in storage config');
        }
        const sharedExpPath = path.join(this.config.storage.localMountPath, 'nni', getExperimentId());
        await this.copyTrialCodeToSharedStorage(codeCopyDir, sharedExpPath);

        // Create one environment
        // FIXME: create new environment when requested
        const env: EnvironmentInformation = await this.createEnvironment(sharedExpPath);
        this.envs.set(env.id, env);
        this.log.info('FrameworkController training service started.');
        this.envManageLoopPromise = this.environmentManagementLoop();
        // Return an empty list because the first environment has not been in running state.
        // The available environments will be updated through onEnvironmentUpdate callback.
        return [ ];
    }

    public async stop(): Promise<void> {
        if (this.envService === undefined) {
            throw new Error('Environment service is not initialized');
        }
        this.stopping = true;
        await this.envManageLoopPromise;
        // Stop environments
        this.log.debug('Stopping environments...');
        const promises: Promise<void>[] = [];
        for (const env of this.envs.values()) {
            promises.push(this.envService.stopEnvironment(env));
        }
        await Promise.all(promises);
        this.log.info('FrameworkController training service stopped.');
        return Promise.resolve();
    }

    /* Following methods are guaranteed to be invoked after start() and before stop(). */

    public async uploadDirectory(_directoryName: string, _path: string): Promise<void> {
        // not in use, because uploading directory has been done in preparing environment.
        return;
    }

    /**
     *  By design _trialCommand and _directoryName should be used instead of the command in this.config,
     *  for now we just use the commands in this.config.
     * 
     *  Return trial ID on success.
     *  Return null if the environment is not available.
     **/
    public async createTrial(environmentId: string, _trialCommand: string, _directoryName: string): Promise<string | null> {
        if (this.commandChannel === undefined) {
            throw new Error('Command channel is not initialized.');
        }
        this.log.info(`Create trial in environment ${environmentId}.`);
        const env = this.envs.get(environmentId);
        if (env === undefined) {
            throw new Error(`Environment ${environmentId} not found`);
        } else if (env.status === 'RUNNING' && env.runningTrialCount === 0) {
            // FIXME: use existing trialId when resume
            const trialId: string = uniqueString(5);
            // The logic of obtaining parameter is a little complex
            this.commandEmitter.emit('request_parameter', trialId);
            const latestParam = this.parameters[this.parameters.length - 1];
            this.log.info(`The obtained parameter: ${latestParam}}`);
            assert(latestParam[0] === trialId);
            const settings = {
                trialId: trialId,
                gpuIndices: '',
                sequenceId: 0,
                parameter: latestParam[1],
            }
            // FIXME: sendCommand only needs envId
            await this.commandChannel.sendCommand(env, NEW_TRIAL_JOB, settings);
            env.runningTrialCount = 1;
            this.trialToEnv.set(trialId, env);
            this.commandEmitter.emit('trial_start', trialId);
            return Promise.resolve(trialId);
        } else {
            this.log.warning(`Environment ${environmentId} is in ${env.status} state, is running ${env.runningTrialCount} trials.`);
            return Promise.resolve(null);
        }
    }

    /**
     *  Kill a trial.
     *  The trial ID is guaranteed to exist, but the trial is not guaranteed to be running.
     **/
    public async stopTrial(trialId: string): Promise<void> {
        this.log.info(`Stop trial ${trialId}.`);
        if (this.commandChannel === undefined) {
            throw new Error('Command channel is not initialized.');
        }
        const env = this.trialToEnv.get(trialId);
        if (env === undefined) {
            this.log.warning(`Trial ${trialId} not found, stop trial ignored.`);
            return Promise.resolve();
        }
        await this.commandChannel.sendCommand(env, KILL_TRIAL_JOB, trialId);
        return Promise.resolve();
    }

    // TODO: resume trial

    /**
     *  Send a hyperparameter configuration to a trial.
     *  Will only be invoked after onRequestParameter().
     **/
    public async sendParameter(trialId: string, parameter: Parameter): Promise<void> {
        if (this.commandChannel === undefined) {
            throw new Error('Command channel is not initialized, please call init() first');
        }
        this.log.debug(`Send parameter ${parameter} to trial ${trialId}.`);
        this.parameters.push([trialId, parameter]);
        return Promise.resolve();
    }

    /* Following methods are guaranteed to be invoked after init() and before start(). */

    /**
     *  Invoke the callback when a trial invokes nni.get_next_parameter().
     **/
    public onRequestParameter(callback: (trialId: string) => Promise<void>): void {
        this.log.debug('onRequestParameter callback called.');
        this.commandEmitter.on('request_parameter', (trialId) => {
            callback(trialId);
        });
    }

    /**
     *  Invoke the callback when a trial invokes nni.report_final_result() and nni.report_intermediate_result().
     **/
    public onMetric(callback: (trialId: string, metric: Metric) => Promise<void>): void {
        this.log.debug('onMetric callback called.');
        this.commandEmitter.on('metric', (trialId, metricStr) => {
            this.log.debug(`onMetric callback called with data from trial ${trialId}: ${metricStr}.`);
            callback(trialId, metricStr);
        });
    }

    /**
     *  Invoke the callback when a trial process is launched.
     *
     *  If there are multiple listeners, `timestamp` should be consistent.
     *
     *  If the training platform automatically retries failed jobs, the callback should only be invoked on first start.
     **/
    public onTrialStart(callback: (trialId: string, timestamp: number) => Promise<void>): void {
        this.log.debug('onTrialStart callback called.');
        this.commandEmitter.on('trial_start', (trialId) => {
            callback(trialId, Date.now());
        });
    }

    /**
     *  Invoke the callback when a trial stops.
     *
     *  If the trial stops on its own, provide the exit code.
     *  If the trial is killed for any reason, set `exitCode` to null.
     *
     *  If there are multiple listeners, `timestamp` should be consistent.
     *
     *  If the training platform automatically retries failed jobs, the callback should only be invoked on last end.
     **/
    public onTrialEnd(callback: (trialId: string, timestamp: number, exitCode: number | null) => Promise<void>): void {
        this.log.debug('onTrialEnd callback called.');
        this.commandEmitter.on('trial_end', (envId, trialId, exitCode, endTime) => {
            const env = this.envs.get(envId)
            if (env === undefined) {
                throw new Error(`The finished trial's environment ${envId} not found`);
            }
            env.runningTrialCount = 0;
            callback(trialId, endTime, exitCode);
        });
    }

    /**
     *  Invoke the callback when any environment's status changes.
     *
     *  Note that `environments` object should be immutable.
     **/
    public onEnvironmentUpdate(callback: (environments: EnvironmentInformation[]) => Promise<void>): void {
        this.log.debug('onEnvironmentUpdate callback called.');
        // The callback is invoked when (1) environment is started or exits, (2) trial job is started or finished.
        this.commandEmitter.on('env_status_change', () => {
            callback(Array.from(this.envs.values())
            .filter((env) => env.status === 'RUNNING' && env.isRunnerReady && env.runningTrialCount === 0));
        });
    }
}
