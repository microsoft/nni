// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { EventEmitter } from "events";
import { getLogger, Logger } from "../../common/log";
import { TrialJobStatus } from "../../common/trainingService";
import { GPUInfo } from "../../training_service/common/gpuData";
import { WebCommandChannel } from "./channels/webCommandChannel";
import { CommandChannel } from "./commandChannel";


export type EnvironmentStatus = 'UNKNOWN' | 'WAITING' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'USER_CANCELED';
export type Channel = "web" | "file" | "aml" | "ut";


export class TrialGpuSummary {
    // GPU count on the machine
    public gpuCount: number;
    // The timestamp when GPU summary data queried
    public timestamp: string;
    // The array of GPU information for each GPU card
    public gpuInfos: GPUInfo[];
    // GPU assigned status
    public assignedGpuIndexMap: Map<number, number> = new Map<number, number>();

    constructor(gpuCount: number, timestamp: string, gpuInfos: GPUInfo[]) {
        this.gpuCount = gpuCount;
        this.timestamp = timestamp;
        this.gpuInfos = gpuInfos;
    }
}

export class EnvironmentInformation {
    // node id is 5 chars, so won't conflict.
    private readonly defaultNodeId = "default";
    private log: Logger;
    private isNoGpuWarned: boolean = false;

    // key states
    // true: environment is running, waiting, or unknown.
    public isAlive: boolean = true;
    // true: Runner is initialized, and can receive trials.
    public isRunnerReady: boolean = false;
    // don't set status in environment directly, use setFinalState function to set a final state.
    public status: EnvironmentStatus = "UNKNOWN";

    // true: environment is ready to run trial.
    public runningTrialCount: number = 0;
    // uses to count how many trial runs on this environment.
    // it can be used in many scenarios, but for now, it uses for reusable.
    public assignedTrialCount: number = 0;

    // NNI environment ID
    public id: string;
    // training platform unique job ID.
    public envId: string;
    // training platform job friendly name, in case it's different with job ID.
    public name: string;
    public trackingUrl: string = "";
    public workingFolder: string = "";
    public runnerWorkingFolder: string = "";
    public command: string = "";
    public nodeCount: number = 1;

    // it's used to aggregate node status for multiple node trial
    public nodes: Map<string, NodeInformation>;
    public gpuSummaries: Map<string, TrialGpuSummary> = new Map<string, TrialGpuSummary>();

    // use can specify which gpus can be used by NNI.
    // it's usable for sharable environment like remote machine.
    public usableGpus?: number[];
    // user can specify how to use GPU resource for an environment, like local and remote.
    public maxTrialNumberPerGpu?: number;
    public useActiveGpu?: boolean;

    constructor(id: string, name: string, envId?: string) {
        this.log = getLogger();
        this.id = id;
        this.name = name;
        this.envId = envId ? envId : name;
        this.nodes = new Map<string, NodeInformation>();
    }

    public setStatus(status: EnvironmentStatus): void {
        if (this.status !== status) {
            this.log.info(`EnvironmentInformation: ${this.envId} change status from ${this.status} to ${status}.`)
            this.status = status;
        }
    }

    public setGpuSummary(nodeId: string, newGpuSummary: TrialGpuSummary): void {
        if (nodeId === null || nodeId === undefined) {
            nodeId = this.defaultNodeId;
        }

        const originalGpuSummary = this.gpuSummaries.get(nodeId);
        if (undefined === originalGpuSummary) {
            newGpuSummary.assignedGpuIndexMap = new Map<number, number>();
            this.gpuSummaries.set(nodeId, newGpuSummary);
        } else {
            originalGpuSummary.gpuCount = newGpuSummary.gpuCount;
            originalGpuSummary.timestamp = newGpuSummary.timestamp;
            originalGpuSummary.gpuInfos = newGpuSummary.gpuInfos;
        }
    }

    public get defaultGpuSummary(): TrialGpuSummary | undefined {
        const gpuSummary = this.gpuSummaries.get(this.defaultNodeId);
        if (gpuSummary === undefined) {
            if (false === this.isNoGpuWarned) {
                this.log.warning(`EnvironmentInformation: ${this.envId} no default gpu found. current gpu info ${JSON.stringify(this.gpuSummaries)}`);
                this.isNoGpuWarned = true;
            }
        } else {
            this.isNoGpuWarned = false;
        }
        return gpuSummary;
    }
}

export abstract class EnvironmentService {

    public abstract get hasStorageService(): boolean;
    public abstract config(key: string, value: string): Promise<void>;
    public abstract refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void>;
    public abstract stopEnvironment(environment: EnvironmentInformation): Promise<void>;
    public abstract startEnvironment(environment: EnvironmentInformation): Promise<void>;
    
    // It is used to set prefetched environment count, default value is 0 for OpenPAI and AML mode,
    // in remote mode, this value is set to the length of machine list.
    public get prefetchedEnvironmentCount(): number {
        return 0;
    }

    // It depends on environment pressure and settings
    // for example, OpenPAI relies on API calls, and there is an limitation for frequence, so it need to be bigger.
    public get environmentMaintenceLoopInterval(): number {
        return 5000;
    }

    // it's needed in two scenario
    // 1. remote machine has fixed number, so it can return false, when all environment are assigned.
    // 2. If there are consistent error on requested environments, for example, authentication failure on platform.
    public get hasMoreEnvironments(): boolean {
        return true;
    }

    public createCommandChannel(commandEmitter: EventEmitter): CommandChannel {
        return new WebCommandChannel(commandEmitter);
    }

    public createEnvironmentInformation(envId: string, envName: string): EnvironmentInformation {
        return new EnvironmentInformation(envId, envName);
    }
}

export class NodeInformation {
    public id: string;
    public status: TrialJobStatus = "UNKNOWN";
    public endTime?: number;

    constructor(id: string) {
        this.id = id;
    }
}

export class RunnerSettings {
    public experimentId: string = "";
    public platform: string = "";
    public nniManagerIP: string = "";
    public nniManagerPort: number = 8081;
    public nniManagerVersion: string = "";
    public logCollection: string = "none";
    public command: string = "";
    public enableGpuCollector: boolean = true;

    // specify which communication channel is used by runner.
    // supported channel includes: rest, storage, aml
    public commandChannel: Channel = "file";
}
