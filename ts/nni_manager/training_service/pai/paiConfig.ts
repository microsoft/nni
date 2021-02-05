// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { TrialJobApplicationForm, TrialJobDetail, TrialJobStatus } from '../../common/trainingService';
import {TrialConfig} from '../common/trialConfig';

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

export const PAI_TRIAL_COMMAND_FORMAT: string =
`export NNI_PLATFORM=pai NNI_SYS_DIR={0} NNI_OUTPUT_DIR={1} NNI_TRIAL_JOB_ID={2} NNI_EXP_ID={3} NNI_TRIAL_SEQ_ID={4} MULTI_PHASE={5} \
&& NNI_CODE_DIR={6} && mkdir -p $NNI_SYS_DIR/code && cp -r $NNI_CODE_DIR/. $NNI_SYS_DIR/code && sh $NNI_SYS_DIR/install_nni.sh \
&& cd $NNI_SYS_DIR/code && python3 -m nni.tools.trial_tool.trial_keeper --trial_command '{7}' --nnimanager_ip '{8}' --nnimanager_port '{9}' \
--nni_manager_version '{10}' --log_collection '{11}' | tee $NNI_OUTPUT_DIR/trial.log`;

/**
 * PAI trial configuration
 */
export class NNIPAITrialConfig extends TrialConfig {
    public readonly cpuNum: number;
    public readonly memoryMB: number;
    public readonly image: string;
    public virtualCluster?: string;
    public readonly nniManagerNFSMountPath: string;
    public readonly containerNFSMountPath: string;
    public readonly paiStorageConfigName: string;
    public readonly paiConfigPath?: string;

    constructor(command: string, codeDir: string, gpuNum: number, cpuNum: number, memoryMB: number,
                image: string, nniManagerNFSMountPath: string, containerNFSMountPath: string,
                paiStorageConfigName: string, virtualCluster?: string, paiConfigPath?: string) {
        super(command, codeDir, gpuNum);
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.image = image;
        this.virtualCluster = virtualCluster;
        this.nniManagerNFSMountPath = nniManagerNFSMountPath;
        this.containerNFSMountPath = containerNFSMountPath;
        this.paiStorageConfigName = paiStorageConfigName;
        this.paiConfigPath = paiConfigPath;
    }
}
