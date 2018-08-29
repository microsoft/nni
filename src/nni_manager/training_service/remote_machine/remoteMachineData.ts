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

import { Client } from 'ssh2';
import { JobApplicationForm, TrialJobDetail, TrialJobStatus  } from '../../common/trainingService';
import { GPUSummary } from '../common/gpuData';


/**
 * Metadata of remote machine for configuration and statuc query
 */
export class RemoteMachineMeta {
    public readonly ip : string;
    public readonly port : number;
    public readonly username : string;
    public readonly passwd: string;
    public gpuSummary : GPUSummary | undefined;
    /* GPU Reservation info, the key is GPU index, the value is the job id which reserves this GPU*/
    public gpuReservation : Map<number, string>;

    constructor(ip : string, port : number, username : string, passwd : string) {
        this.ip = ip;
        this.port = port;
        this.username = username;
        this.passwd = passwd;
        this.gpuReservation = new Map<number, string>();
    }
}

/**
 * The execution result for command executed on remote machine
 */
export class RemoteCommandResult {
    public readonly stdout : string;
    public readonly stderr : string;
    public readonly exitCode : number;

    constructor(stdout : string, stderr : string, exitCode : number) {
        this.stdout = stdout;
        this.stderr = stderr;
        this.exitCode = exitCode;
    }
}

// tslint:disable-next-line:max-classes-per-file
export class JobMetrics {
    public readonly jobId: string;
    public readonly metrics: string[];
    public readonly jobStatus: TrialJobStatus;
    public readonly endTimestamp: number;

    constructor(jobId : string, metrics : string[], jobStatus : TrialJobStatus, endTimestamp : number) {
        this.jobId = jobId;
        this.metrics = metrics;
        this.jobStatus = jobStatus;
        this.endTimestamp = endTimestamp;
    }
}

/**
 * RemoteMachineTrialJobDetail
 */
// tslint:disable-next-line:max-classes-per-file
export class RemoteMachineTrialJobDetail implements TrialJobDetail {
    public id: string;
    public status: TrialJobStatus;
    public submitTime: Date;
    public startTime?: Date;
    public endTime?: Date;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: JobApplicationForm;
    public rmMeta?: RemoteMachineMeta;

    constructor(id: string, status: TrialJobStatus, submitTime: Date, workingDirectory: string, form: JobApplicationForm) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.tags = [];
    }
}

export type RemoteMachineScheduleResult = { scheduleInfo : RemoteMachineScheduleInfo | undefined, resultType : ScheduleResultType};

export type RemoteMachineScheduleInfo = { client: Client; rmMeta : RemoteMachineMeta; cuda_visible_device : string};

export enum ScheduleResultType {
    /* Schedule succeeded*/
    SUCCEED,

    /* Temporarily, no enough available GPU right now */    
    TMP_NO_AVAILABLE_GPU,

    /* Cannot match requirement even if all GPU are a*/
    REQUIRE_EXCEED_TOTAL
}

export const REMOTEMACHINERUNSHELLFORMAT: string =
`#!/bin/bash
export NNI_PLATFORM=remote NNI_SYS_DIR={0} NNI_TRIAL_JOB_ID={1} NNI_OUTPUT_DIR={0}
cd $NNI_SYS_DIR
echo $$ >{2}
eval {3}{4} 2>{5}
echo $? \`date +%s%3N\` >{6}`;

export const HOSTJOBSHELLFORMAT: string =
`#!/bin/bash
cd {0}
echo $$ >{1}
eval {2} >stdout 2>stderr
echo $? \`date +%s%3N\` >{3}`;
