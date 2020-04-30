// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { TrialJobApplicationForm, TrialJobDetail, TrialJobStatus } from '../../common/trainingService';
import { GPUInfo, GPUSummary } from '../common/gpuData';
import { ShellExecutor } from './shellExecutor';

/**
 * Metadata of remote machine for configuration and statuc query
 */
export class RemoteMachineMeta {
    public readonly ip: string = '';
    public readonly port: number = 22;
    public readonly username: string = '';
    public readonly passwd: string = '';
    public readonly sshKeyPath?: string;
    public readonly passphrase?: string;
    public gpuSummary: GPUSummary | undefined;
    public readonly gpuIndices?: string;
    public readonly maxTrialNumPerGpu?: number;
    //TODO: initialize varialbe in constructor
    public occupiedGpuIndexMap?: Map<number, number>;
    public readonly useActiveGpu?: boolean = false;
}

export function parseGpuIndices(gpuIndices?: string): Set<number> | undefined {
    if (gpuIndices !== undefined) {
        const indices: number[] = gpuIndices.split(',')
            .map((x: string) => parseInt(x, 10));
        if (indices.length > 0) {
            return new Set(indices);
        } else {
            throw new Error('gpuIndices can not be empty if specified.');
        }
    }
}

/**
 * The execution result for command executed on remote machine
 */
export class RemoteCommandResult {
    public readonly stdout: string;
    public readonly stderr: string;
    public readonly exitCode: number;

    constructor(stdout: string, stderr: string, exitCode: number) {
        this.stdout = stdout;
        this.stderr = stderr;
        this.exitCode = exitCode;
    }
}

/**
 * RemoteMachineTrialJobDetail
 */
export class RemoteMachineTrialJobDetail implements TrialJobDetail {
    public id: string;
    public status: TrialJobStatus;
    public submitTime: number;
    public startTime?: number;
    public endTime?: number;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: TrialJobApplicationForm;
    public rmMeta?: RemoteMachineMeta;
    public isEarlyStopped?: boolean;
    public gpuIndices: GPUInfo[];

    constructor(id: string, status: TrialJobStatus, submitTime: number,
        workingDirectory: string, form: TrialJobApplicationForm) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.tags = [];
        this.gpuIndices = [];
    }
}

/**
 * The remote machine executor manager
 */
export class ExecutorManager {
    private readonly executorArray: ShellExecutor[];
    private readonly maxTrialNumberPerConnection: number;
    private readonly rmMeta: RemoteMachineMeta;
    constructor(executorArray: ShellExecutor[], maxTrialNumberPerConnection: number, rmMeta: RemoteMachineMeta) {
        this.rmMeta = rmMeta;
        this.executorArray = executorArray;
        this.maxTrialNumberPerConnection = maxTrialNumberPerConnection;
    }

    /**
     * find a available executor, if no executor available, return a new one
     */
    public async getAvailableExecutor(): Promise<ShellExecutor> {
        for (const index of this.executorArray.keys()) {
            const connectionNumber: number = this.executorArray[index].getUsedConnectionNumber;
            if (connectionNumber < this.maxTrialNumberPerConnection) {
                this.executorArray[index].addUsedConnectionNumber();

                return this.executorArray[index];
            }
        }

        //init a new executor if could not get an available one
        return await this.initNewShellExecutor();
    }

    /**
     * add a new executor to executorArray
     * @param executor ShellExecutor
     */
    public addNewShellExecutor(executor: ShellExecutor): void {
        this.executorArray.push(executor);
    }

    /**
     * first executor instance is used for gpu collector and host job
     */
    public getFirstExecutor(): ShellExecutor {
        return this.executorArray[0];
    }

    /**
     * close all of executor
     */
    public closeAllExecutor(): void {
        for (const executor of this.executorArray) {
            executor.close();
        }
    }

    /**
     * retrieve resource, minus a number for given executor
     * @param executor executor
     */
    public releaseConnection(executor: ShellExecutor | undefined): void {
        if (executor === undefined) {
            throw new Error(`could not release a undefined executor`);
        }
        for (const index of this.executorArray.keys()) {
            if (this.executorArray[index] === executor) {
                this.executorArray[index].minusUsedConnectionNumber();
                break;
            }
        }
    }

    /**
     * Create a new connection executor and initialize it
     */
    private async initNewShellExecutor(): Promise<ShellExecutor> {
        const executor = new ShellExecutor();
        await executor.initialize(this.rmMeta);
        return executor;
    }
}

export type RemoteMachineScheduleResult = { scheduleInfo: RemoteMachineScheduleInfo | undefined; resultType: ScheduleResultType };

export type RemoteMachineScheduleInfo = { rmMeta: RemoteMachineMeta; cudaVisibleDevice: string };

export enum ScheduleResultType {
    // Schedule succeeded
    SUCCEED,

    // Temporarily, no enough available GPU right now
    TMP_NO_AVAILABLE_GPU,

    // Cannot match requirement even if all GPU are a
    REQUIRE_EXCEED_TOTAL
}

export const REMOTEMACHINE_TRIAL_COMMAND_FORMAT: string =
    `#!/bin/bash
export NNI_PLATFORM=remote NNI_SYS_DIR={0} NNI_OUTPUT_DIR={1} NNI_TRIAL_JOB_ID={2} NNI_EXP_ID={3} \
NNI_TRIAL_SEQ_ID={4} export MULTI_PHASE={5}
cd $NNI_SYS_DIR
sh install_nni.sh
echo $$ >{6}
python3 -m nni_trial_tool.trial_keeper --trial_command '{7}' --nnimanager_ip '{8}' --nnimanager_port '{9}' \
--nni_manager_version '{10}' --log_collection '{11}' 1>$NNI_OUTPUT_DIR/trialkeeper_stdout 2>$NNI_OUTPUT_DIR/trialkeeper_stderr
echo $? \`date +%s%3N\` >{12}`;

export const HOST_JOB_SHELL_FORMAT: string =
    `#!/bin/bash
cd {0}
echo $$ >{1}
eval {2} >stdout 2>stderr
echo $? \`date +%s%3N\` >{3}`;
