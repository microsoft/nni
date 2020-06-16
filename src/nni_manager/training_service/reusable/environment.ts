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


export type EnvironmentStatus = 'UNKNOWN' | 'WAITING' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'USER_CANCELED';

export abstract class EnvironmentService {

    public abstract get hasStorageService(): boolean;

    public abstract config(key: string, value: string): Promise<void>;
    public abstract refreshEnvironmentsStatus(environment: EnvironmentInformation[]): Promise<void>;
    public abstract startEnvironment(environment: EnvironmentInformation): Promise<void>;
    public abstract stopEnvironment(environment: EnvironmentInformation): Promise<void>;
}

export class RunnerSettings {
    public experimentId: string = "";
    public platform: string = "";
    public nniManagerIP: string = "";
    public nniManagerPort: number = 8081;
    public nniManagerVersion: string = "";
    public logCollection: string = "none";
    public command: string = "";
}

export class EnvironmentInformation {
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
    public status: EnvironmentStatus = "UNKNOWN";

    public trackingUrl: string = "";
    public workingFolder: string = "";
    public runnerWorkingFolder: string = "";
    public command: string = "";
    public nodeCount: number = 1;

    constructor(id: string, jobName: string, jobId?: string) {
        this.id = id;
        this.jobName = jobName;
        this.jobId = jobId ? jobId : jobName;
    }
}
