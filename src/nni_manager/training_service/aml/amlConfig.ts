// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { TrialJobApplicationForm, TrialJobDetail, TrialJobStatus  } from '../../common/trainingService';
import {TrialConfig} from '../common/trialConfig';

export class AMLClusterConfig {
    public readonly subscriptionId: string;
    public readonly resourceGroup: string;
    public readonly workspaceName: string;

    constructor(subscriptionId: string, resourceGroup: string, workspaceName: string) {
        this.subscriptionId = subscriptionId;
        this.resourceGroup = resourceGroup;
        this.workspaceName = workspaceName;
    }
}

export class AMLTrialConfig extends TrialConfig {
    public readonly image: string;
    public readonly command: string;
    public readonly codeDir: string;
    public readonly nodeCount: number;
    public readonly computerTarget: string;

    constructor(codeDir: string, command: string, image: string, nodeCount: number, computerTarget: string) {
        super("", codeDir, 0);
        this.codeDir = codeDir;
        this.command = command;
        this.image = image;
        this.nodeCount = nodeCount;
        this.computerTarget = computerTarget;
    }
}

/**
 * AML trial job detail
 */
export class AMLTrialJobDetail implements TrialJobDetail {
    public id: string;
    public status: TrialJobStatus;
    public amlJobName: string;
    public submitTime: number;
    public startTime?: number;
    public endTime?: number;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: TrialJobApplicationForm;
    public logPath: string;
    public isEarlyStopped?: boolean;

    constructor(id: string, status: TrialJobStatus, amlJobName: string,
                submitTime: number, workingDirectory: string, form: TrialJobApplicationForm, logPath: string) {
        this.id = id;
        this.status = status;
        this.amlJobName = amlJobName;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.tags = [];
        this.logPath = logPath;
    }
}
