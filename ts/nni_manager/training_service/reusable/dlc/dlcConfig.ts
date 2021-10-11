// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { TrialConfig } from 'training_service/common/trialConfig';
import { EnvironmentInformation } from '../environment';
import { DlcClient } from '../dlc/dlcClient';

export class DlcClusterConfig {
    public readonly type: string;
    public readonly image: string;
    public readonly podCount: number;
    public readonly ecsSpec: string;

    constructor(type: string, image: string, podCount: number, ecsSpec: string) {
        this.type = type;
        this.image = image;
        this.podCount = podCount;
        this.ecsSpec = ecsSpec;
    }
}

export class DlcTrialConfig extends TrialConfig {
    public readonly image: string;
    public readonly command: string;
    public readonly codeDir: string;

    constructor(codeDir: string, command: string, image: string) {
        super("", codeDir, 0);
        this.codeDir = codeDir;
        this.command = command;
        this.image = image;
    }
}

export class DlcEnvironmentInformation extends EnvironmentInformation {
    public dlcClient?: DlcClient;
    public currentMessageIndex: number = -1;
}
