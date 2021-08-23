// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { TrialJobApplicationForm, TrialJobDetail, TrialJobStatus } from '../../common/trainingService';
import { RemoteMachineConfig } from '../../common/experimentConfig';
import { GPUInfo, GPUSummary, ScheduleResultType } from '../common/gpuData';
import { ShellExecutor } from './shellExecutor';

/**
 * Metadata of remote machine for configuration and statuc query
 */
export class RemoteMachineMeta {
    public readonly config: RemoteMachineConfig;
    public gpuSummary: GPUSummary | undefined;
    public occupiedGpuIndexMap: Map<number, number>;

    constructor(config: RemoteMachineConfig) {
        this.config = config;
        this.occupiedGpuIndexMap = new Map<number, number>();
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
    public readonly rmMeta: RemoteMachineMeta;
    private readonly executorMap: Map<string, ShellExecutor> = new Map<string, ShellExecutor>();

    private executors: ShellExecutor[] = [];

    constructor(config: RemoteMachineConfig) {
        this.rmMeta = new RemoteMachineMeta(config);
    }

    public async getExecutor(id: string): Promise<ShellExecutor> {
        let isFound = false;
        let executor: ShellExecutor | undefined;

        // already assigned
        if (this.executorMap.has(id)) {
            executor = this.executorMap.get(id);
            if (executor === undefined) {
                throw new Error("executor shouldn't be undefined before return!");
            }
            return executor;
        }

        for (const candidateExecutor of this.executors) {
            if (candidateExecutor.addUsage()) {
                isFound = true;
                executor = candidateExecutor;
                break;
            }
        }
        // init a new executor if no free one.
        if (!isFound) {
            executor = await this.createShellExecutor();
        }

        if (executor === undefined) {
            throw new Error("executor shouldn't be undefined before set!");
        }
        this.executorMap.set(id, executor);

        return executor;
    }

    /**
     * close all of executor
     */
    public releaseAllExecutor(): void {
        this.executorMap.clear();
        for (const executor of this.executors) {
            executor.close();
        }
        this.executors = [];
    }

    /**
     * retrieve resource, minus a number for given executor
     * @param executor executor
     */
    public releaseExecutor(id: string): void {
        const executor = this.executorMap.get(id);
        if (executor === undefined) {
            throw new Error(`executor for ${id} is not found`);
        }
        executor.releaseUsage();
        this.executorMap.delete(id);
    }

    /**
     * Create a new connection executor and initialize it
     */
    private async createShellExecutor(): Promise<ShellExecutor> {
        const executor = new ShellExecutor();
        await executor.initialize(this.rmMeta);
        if (!executor.addUsage()) {
            throw new Error("failed to add usage on new created Executor! It's a wired bug!");
        }
        this.executors.push(executor);
        return executor;
    }
}

export type RemoteMachineScheduleResult = { scheduleInfo: RemoteMachineScheduleInfo | undefined; resultType: ScheduleResultType };

export type RemoteMachineScheduleInfo = { rmMeta: RemoteMachineMeta; cudaVisibleDevice: string };
