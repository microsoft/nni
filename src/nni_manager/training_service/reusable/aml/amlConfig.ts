// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

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
    public readonly computeTarget: string;

    constructor(codeDir: string, command: string, image: string, computeTarget: string) {
        super("", codeDir, 0);
        this.codeDir = codeDir;
        this.command = command;
        this.image = image;
        this.computeTarget = computeTarget;
    }
}

export class AMLEnvironmentInformation extends EnvironmentInformation {
    public amlClient?: AMLClient;
    public currentMessageIndex: number = -1;
}
