// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { TrialJobApplicationForm, TrialJobDetail, TrialJobStatus } from '../../common/trainingService';

export class PAIClusterConfig {
    public readonly userName: string;
    public readonly passWord?: string;
    public host: string;
    public readonly token?: string;
    public readonly reuse?: boolean;

    public cpuNum?: number;
    public memoryMB?: number;
    public gpuNum?: number;

    public useActiveGpu?: boolean;
    public maxTrialNumPerGpu?: number;

    /**
     * Constructor
     * @param userName User name of PAI Cluster
     * @param passWord password of PAI Cluster
     * @param host Host IP of PAI Cluster
     * @param token PAI token of PAI Cluster
     * @param reuse If job is reusable for multiple trials
     */
    constructor(userName: string, host: string, passWord?: string, token?: string, reuse?: boolean,
        cpuNum?: number, memoryMB?: number, gpuNum?: number) {
        this.userName = userName;
        this.passWord = passWord;
        this.host = host;
        this.token = token;
        this.reuse = reuse;
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.gpuNum = gpuNum;
    }
}

/**
 * PAI trial job detail
 */
export class PAITrialJobDetail implements TrialJobDetail {
    public id: string;
    public status: TrialJobStatus;
    public paiJobName: string;
    public submitTime: number;
    public startTime?: number;
    public endTime?: number;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: TrialJobApplicationForm;
    public logPath: string;
    public isEarlyStopped?: boolean;
    public paiJobDetailUrl?: string;

    constructor(id: string, status: TrialJobStatus, paiJobName: string,
                submitTime: number, workingDirectory: string, form: TrialJobApplicationForm, logPath: string, paiJobDetailUrl?: string) {
        this.id = id;
        this.status = status;
        this.paiJobName = paiJobName;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.tags = [];
        this.logPath = logPath;
        this.paiJobDetailUrl = paiJobDetailUrl;
    }
}
