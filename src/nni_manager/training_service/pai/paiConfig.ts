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

import {TrialConfig} from '../common/trialConfig'

export class PAITaskRole {
    // Name for the task role
    public readonly name: string;
    // Number of tasks for the task role, no less than 1
    public readonly taskNumber: number;
    // CPU number for one task in the task role, no less than 1
    public readonly cpuNumber: number;
    // Memory for one task in the task role, no less than 100
    public readonly memoryMB: number;
    // GPU number for one task in the task role, no less than 0
    public readonly gpuNumber: number;
    // Executable command for tasks in the task role, can not be empty
    public readonly command: string;
    //Shared memory for one task in the task role
    public readonly shmMB?: number;
    
    /**
     * Constructor
     * @param name Name for the task role
     * @param taskNumber Number of tasks for the task role, no less than 1
     * @param cpuNumber CPU number for one task in the task role, no less than 1
     * @param memoryMB Memory for one task in the task role, no less than 100
     * @param gpuNumber GPU number for one task in the task role, no less than 0
     * @param command Executable command for tasks in the task role, can not be empty
     */
    constructor(name : string, taskNumber : number, cpuNumber : number, memoryMB : number, gpuNumber : number, command : string, shmMB?: number) {
        this.name = name;
        this.taskNumber = taskNumber;
        this.cpuNumber = cpuNumber;
        this.memoryMB = memoryMB;
        this.gpuNumber = gpuNumber;
        this.command = command;    
        this.shmMB = shmMB;
    }
}

export class PAIJobConfig{
    // Name for the job, need to be unique
    public readonly jobName: string;
    // URL pointing to the Docker image for all tasks in the job
    public readonly image: string;
    // Data directory existing on HDFS
    public readonly dataDir: string;
    // Output directory on HDFS
    public readonly outputDir: string;
    // Code directory on HDFS
    public readonly codeDir: string;

    // List of taskRole, one task role at least
    public taskRoles: PAITaskRole[];

    //The virtual cluster job runs on.
    public readonly virtualCluster: string;

    /**
     * Constructor
     * @param jobName Name for the job, need to be unique
     * @param image URL pointing to the Docker image for all tasks in the job
     * @param dataDir Data directory existing on HDFS
     * @param outputDir Output directory on HDFS
     * @param taskRoles List of taskRole, one task role at least
     */
    constructor(jobName: string, image : string, dataDir : string, outputDir : string, codeDir : string, 
            taskRoles : PAITaskRole[], virtualCluster: string) {
        this.jobName = jobName;
        this.image = image;
        this.dataDir = dataDir;
        this.outputDir = outputDir;
        this.codeDir = codeDir;
        this.taskRoles = taskRoles;
        this.virtualCluster = virtualCluster;
    }
}

export class PAIClusterConfig {
    public readonly userName: string;
    public readonly passWord: string;
    public readonly host: string;

    /**
     * Constructor
     * @param userName User name of PAI Cluster
     * @param passWord password of PAI Cluster
     * @param host Host IP of PAI Cluster
     */
    constructor(userName: string, passWord : string, host : string){
        this.userName = userName;
        this.passWord = passWord;
        this.host = host;
    }
}

export class NNIPAITrialConfig extends TrialConfig{
    public readonly cpuNum: number;
    public readonly memoryMB: number;
    public readonly image: string;
    public readonly dataDir: string; 
    public outputDir: string;

    //The virtual cluster job runs on. If omitted, the job will run on default virtual cluster
    public virtualCluster?: string;
    //Shared memory for one task in the task role
    public shmMB?: number;

    constructor(command : string, codeDir : string, gpuNum : number, cpuNum: number, memoryMB: number, 
            image: string, dataDir: string, outputDir: string, virtualCluster?: string, shmMB?: number) {
        super(command, codeDir, gpuNum);
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.image = image;
        this.dataDir = dataDir;
        this.outputDir = outputDir;
        this.virtualCluster = virtualCluster;
        this.shmMB = shmMB;
    }
}

