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

import * as fs from 'fs';
import { Client, ConnectConfig } from 'ssh2';
import { Deferred } from 'ts-deferred';
import { JobApplicationForm, TrialJobDetail, TrialJobStatus  } from '../../common/trainingService';
import { GPUSummary, GPUInfo } from '../common/gpuData';

/**
 * Metadata of remote machine for configuration and statuc query
 */
export class RemoteMachineMeta {
    public readonly ip : string = '';
    public readonly port : number = 22;
    public readonly username : string = '';
    public readonly passwd: string = '';
    public readonly sshKeyPath?: string;
    public readonly passphrase?: string;
    public gpuSummary : GPUSummary | undefined;
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
    public gpuIndices: GPUInfo[];

    constructor(id: string, status: TrialJobStatus, submitTime: number,
                workingDirectory: string, form: JobApplicationForm, sequenceId: number) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.sequenceId = sequenceId;
        this.tags = [];
        this.gpuIndices = []
    }
}

/**
 * The remote machine ssh client used for trial and gpu detector
 */
export class SSHClient {
    private readonly sshClient: Client;
    private usedConnectionNumber: number; //count the connection number of every client
    constructor(sshClient: Client, usedConnectionNumber: number) {
        this.sshClient = sshClient;
        this.usedConnectionNumber = usedConnectionNumber;
    }
    
    public get getSSHClientInstance(): Client {
        return this.sshClient;
    }

    public get getUsedConnectionNumber(): number {
        return this.usedConnectionNumber;
    }

    public addUsedConnectionNumber() {
        this.usedConnectionNumber += 1;
    }

    public minusUsedConnectionNumber() {
        this.usedConnectionNumber -= 1;
    }
}

export class SSHClientManager {
    private sshClientArray: SSHClient[];
    private readonly maxTrialNumberPerConnection: number;
    private readonly rmMeta: RemoteMachineMeta;
    constructor(sshClientArray: SSHClient[], maxTrialNumberPerConnection: number, rmMeta: RemoteMachineMeta) {
        this.rmMeta = rmMeta;
        this.sshClientArray = sshClientArray;
        this.maxTrialNumberPerConnection = maxTrialNumberPerConnection;
    }

    /**
     * Create a new ssh connection client and initialize it
     */
    private initNewSSHClient(): Promise<Client> {
        const deferred: Deferred<Client> = new Deferred<Client>();
        const conn: Client = new Client();
        let connectConfig: ConnectConfig = {
            host: this.rmMeta.ip,
            port: this.rmMeta.port,
            username: this.rmMeta.username };
        if (this.rmMeta.passwd) {
            connectConfig.password = this.rmMeta.passwd;                
        } else if(this.rmMeta.sshKeyPath) {
            if(!fs.existsSync(this.rmMeta.sshKeyPath)) {
                //SSh key path is not a valid file, reject
                deferred.reject(new Error(`${this.rmMeta.sshKeyPath} does not exist.`));
            }
            const privateKey: string = fs.readFileSync(this.rmMeta.sshKeyPath, 'utf8');

            connectConfig.privateKey = privateKey;
            connectConfig.passphrase = this.rmMeta.passphrase;
        } else {
            deferred.reject(new Error(`No valid passwd or sshKeyPath is configed.`));
        }
        conn.on('ready', () => {
            this.addNewSSHClient(conn);
            deferred.resolve(conn);
        }).on('error', (err: Error) => {
            // SSH connection error, reject with error message
            deferred.reject(new Error(err.message));
        }).connect(connectConfig);
      
        return deferred.promise;
    }
    
    /**
     * find a available ssh client in ssh array, if no ssh client available, return undefined
     */
    public async getAvailableSSHClient(): Promise<Client> {
        const deferred: Deferred<Client> = new Deferred<Client>();
        for (const index in this.sshClientArray) {
            let connectionNumber: number = this.sshClientArray[index].getUsedConnectionNumber;
            if(connectionNumber < this.maxTrialNumberPerConnection) {
                this.sshClientArray[index].addUsedConnectionNumber();
                deferred.resolve(this.sshClientArray[index].getSSHClientInstance);
                return deferred.promise;
            }
        };
        //init a new ssh client if could not get an available one
        return await this.initNewSSHClient();
    }
    
    /**
     * add a new ssh client to sshClientArray
     * @param sshClient
     */
    public addNewSSHClient(client: Client) {
        this.sshClientArray.push(new SSHClient(client, 1));
    }
    
    /**
     * first ssh clilent instance is used for gpu collector and host job
     */
    public getFirstSSHClient() {
        return this.sshClientArray[0].getSSHClientInstance;
    }
    
    /**
     * close all of ssh client
     */
    public closeAllSSHClient() {
        for (let sshClient of this.sshClientArray) {
            sshClient.getSSHClientInstance.end();
        }
    }
    
    /**
     * retrieve resource, minus a number for given ssh client
     * @param client
     */
    public releaseConnection(client: Client | undefined) {
        if(!client) {
            throw new Error(`could not release a undefined ssh client`);
        }
        for(let index in this.sshClientArray) {
            if(this.sshClientArray[index].getSSHClientInstance === client) {
                this.sshClientArray[index].minusUsedConnectionNumber();
                break;
            }
        }
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
python3 -m nni_trial_tool.trial_keeper --trial_command '{7}' --nnimanager_ip '{8}' --nnimanager_port '{9}' --nni_manager_version '{10}' --log_collection '{11}' 1>$NNI_OUTPUT_DIR/trialkeeper_stdout 2>$NNI_OUTPUT_DIR/trialkeeper_stderr
echo $? \`date +%s%3N\` >{12}`;

export const HOST_JOB_SHELL_FORMAT: string =
`#!/bin/bash
cd {0}
echo $$ >{1}
eval {2} >stdout 2>stderr
echo $? \`date +%s%3N\` >{3}`;
