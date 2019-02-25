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

import { JobApplicationForm, TrialJobDetail, TrialJobStatus  } from '../../common/trainingService';
import { GPUSummary } from '../common/gpuData';


/**
 * Metadata of remote machine for configuration and statuc query
 */
export class RemoteMachineMeta {
    public readonly ip : string;
    public readonly port : number;
    public readonly username : string;
    public readonly passwd?: string;
    public readonly sshKeyPath?: string;
    public readonly passphrase?: string;
    public gpuSummary : GPUSummary | undefined;
    /* GPU Reservation info, the key is GPU index, the value is the job id which reserves this GPU*/
    public gpuReservation : Map<number, string>;

    constructor(ip : string, port : number, username : string, passwd : string, 
        sshKeyPath : string, passphrase : string) {
        this.ip = ip;
        this.port = port;
        this.username = username;
        this.passwd = passwd;
        this.sshKeyPath = sshKeyPath;
        this.passphrase = passphrase;
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

/**
 * RemoteMachineTrialJobDetail
 */
// tslint:disable-next-line:max-classes-per-file
export class RemoteMachineTrialJobDetail implements TrialJobDetail {
    public id: string;
    public status: TrialJobStatus;
    public submitTime: number;
    public startTime?: number;
    public endTime?: number;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: JobApplicationForm;
    public sequenceId: number;
    public rmMeta?: RemoteMachineMeta;
    public isEarlyStopped?: boolean;

    constructor(id: string, status: TrialJobStatus, submitTime: number,
                workingDirectory: string, form: JobApplicationForm, sequenceId: number) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.sequenceId = sequenceId;
        this.tags = [];
    }
}

export type RemoteMachineScheduleResult = { scheduleInfo : RemoteMachineScheduleInfo | undefined; resultType : ScheduleResultType};

export type RemoteMachineScheduleInfo = { rmMeta : RemoteMachineMeta; cuda_visible_device : string};

export enum ScheduleResultType {
    /* Schedule succeeded*/
    SUCCEED,

    /* Temporarily, no enough available GPU right now */    
    TMP_NO_AVAILABLE_GPU,

    /* Cannot match requirement even if all GPU are a*/
    REQUIRE_EXCEED_TOTAL
}

export const REMOTEMACHINE_TRIAL_COMMAND_FORMAT: string =
`#!/bin/bash
export NNI_PLATFORM=remote NNI_SYS_DIR={0} NNI_OUTPUT_DIR={1} NNI_TRIAL_JOB_ID={2} NNI_EXP_ID={3} NNI_TRIAL_SEQ_ID={4} export MULTI_PHASE={5}
cd $NNI_SYS_DIR
sh install_nni.sh
echo $$ >{6}
python3 -m nni_trial_tool.trial_keeper --trial_command '{7}' --nnimanager_ip '{8}' --nnimanager_port '{9}' 1>$NNI_OUTPUT_DIR/trialkeeper_stdout 2>$NNI_OUTPUT_DIR/trialkeeper_stderr
echo $? \`date +%s%3N\` >{10}`;

export const HOST_JOB_SHELL_FORMAT: string =
`#!/bin/bash
cd {0}
echo $$ >{1}
eval {2} >stdout 2>stderr
echo $? \`date +%s%3N\` >{3}`;

export const GPU_COLLECTOR_FORMAT: string = 
`
#!/bin/bash
export METRIC_OUTPUT_DIR={0}
echo $$ >{1}
python3 -m nni_gpu_tool.gpu_metrics_collector
`
