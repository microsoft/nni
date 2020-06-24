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

import { GPUSummary } from "training_service/common/gpuData";
import { getLogger, Logger } from "../../common/log";
import { TrialJobStatus } from "../../common/trainingService";


export type EnvironmentStatus = 'UNKNOWN' | 'WAITING' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'USER_CANCELED';
export type Channel = "rest" | "file" | "aml"

export abstract class EnvironmentService {

    public abstract get hasStorageService(): boolean;

    public abstract config(key: string, value: string): Promise<void>;
    public abstract refreshEnvironmentsStatus(environments: EnvironmentInformation[]): Promise<void>;
    public abstract startEnvironment(environment: EnvironmentInformation): Promise<void>;
    public abstract stopEnvironment(environment: EnvironmentInformation): Promise<void>;
}

export class NodeInfomation {
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
    public enableGpuCollector: boolean = false;

    // specify which communication channel is used by runner.
    // supported channel includes: rest, storage, aml
    public commandChannel: Channel = "file";
}

export class EnvironmentInformation {
    private log: Logger;

    // NNI environment ID
    public id: string;
    // training platform unique job ID.
    public jobId: string;
    // training platform job friendly name, in case it's different with job ID.
    public jobName: string;

    // key states
    // true: environment is ready to run trial.
    public isIdle: boolean = false;
    // true: environment is running, waiting, or unknown.
    public isAlive: boolean = true;
    // don't set status in environment directly, use setFinalState function to set a final state.
    public status: EnvironmentStatus = "UNKNOWN";

    public trackingUrl: string = "";
    public workingFolder: string = "";
    public runnerWorkingFolder: string = "";
    public command: string = "";
    public nodeCount: number = 1;

    // it's used to aggregate node status for multiple node trial
    public nodes: Map<string, NodeInfomation>;
    public gpuSummary: Map<string, GPUSummary> = new Map<string, GPUSummary>();

    constructor(id: string, jobName: string, jobId?: string) {
        this.log = getLogger();
        this.id = id;
        this.jobName = jobName;
        this.jobId = jobId ? jobId : jobName;
        this.nodes = new Map<string, NodeInfomation>();
    }

    public setFinalStatus(status: EnvironmentStatus): void {
        switch (status) {
            case 'WAITING':
            case 'SUCCEEDED':
            case 'FAILED':
            case 'USER_CANCELED':
                this.status = status;
                break;
            default:
                this.log.error(`Environment: job ${this.jobId} set an invalid final state ${status}.`);
                break;
        }
    }
}
