// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import {TrialConfig} from '../../common/trialConfig';

/**
 * Task role for PAI
 */
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
    //portList to specify the port used in container
    public portList?: PortListMetaData[];

    /**
     * Constructor
     * @param name Name for the task role
     * @param taskNumber Number of tasks for the task role, no less than 1
     * @param cpuNumber CPU number for one task in the task role, no less than 1
     * @param memoryMB Memory for one task in the task role, no less than 100
     * @param gpuNumber GPU number for one task in the task role, no less than 0
     * @param command Executable command for tasks in the task role, can not be empty
     */
    constructor(name: string, taskNumber: number, cpuNumber: number, memoryMB: number, gpuNumber: number,
                command: string, shmMB?: number, portList?: PortListMetaData[]) {
        this.name = name;
        this.taskNumber = taskNumber;
        this.cpuNumber = cpuNumber;
        this.memoryMB = memoryMB;
        this.gpuNumber = gpuNumber;
        this.command = command;
        this.shmMB = shmMB;
        this.portList = portList;
    }
}

/**
 * Trial job configuration submitted to PAI
 */
export class PAIJobConfig {
    // Name for the job, need to be unique
    public readonly jobName: string;
    // URL pointing to the Docker image for all tasks in the job
    public readonly image: string;
    // Code directory on HDFS
    public readonly codeDir: string;
    //authentication file used for private Docker registry 
    public readonly authFile?: string;

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
    constructor(jobName: string, image: string, codeDir: string,
                taskRoles: PAITaskRole[], virtualCluster: string, authFile?: string) {
        this.jobName = jobName;
        this.image = image;
        this.codeDir = codeDir;
        this.taskRoles = taskRoles;
        this.virtualCluster = virtualCluster;
        this.authFile = authFile;
    }
}

/**
 * portList data structure used in PAI taskRole
 */
export class PortListMetaData {
    public readonly label: string = '';
    public readonly beginAt: number = 0;
    public readonly portNumber: number = 0;
}
  

/**
 * PAI trial configuration
 */
export class NNIPAITrialConfig extends TrialConfig {
    public readonly cpuNum: number;
    public readonly memoryMB: number;
    public readonly image: string;

    //The virtual cluster job runs on. If omitted, the job will run on default virtual cluster
    public virtualCluster?: string;
    //Shared memory for one task in the task role
    public shmMB?: number;
    //authentication file used for private Docker registry 
    public authFile?: string;
    //portList to specify the port used in container
    public portList?: PortListMetaData[];

    constructor(command: string, codeDir: string, gpuNum: number, cpuNum: number, memoryMB: number,
                image: string, virtualCluster?: string, shmMB?: number, authFile?: string, portList?: PortListMetaData[]) {
        super(command, codeDir, gpuNum);
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.image = image;
        this.virtualCluster = virtualCluster;
        this.shmMB = shmMB;
        this.authFile = authFile;
        this.portList = portList;
    }
}
