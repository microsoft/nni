// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { TrialJobApplicationForm, TrialJobDetail, TrialJobStatus  } from '../../../common/trainingService';
import { TrialConfig } from '../../common/trialConfig';
import { EnvironmentInformation } from '../environment';
import { AMLClient } from '../aml/amlClient';

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

export class AMLEnvironmentInformation extends EnvironmentInformation {
        public amlClient?: AMLClient;
}
