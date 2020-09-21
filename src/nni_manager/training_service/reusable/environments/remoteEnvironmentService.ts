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
import { ShellExecutor } from 'training_service/remote_machine/shellExecutor';


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
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<void> {
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
    }
}
