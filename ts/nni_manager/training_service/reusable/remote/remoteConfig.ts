// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { EnvironmentInformation } from '../environment';
import { RemoteMachineConfig } from '../../../common/experimentConfig';

/**
 * RemoteMachineEnvironmentInformation
 */
export class RemoteMachineEnvironmentInformation extends EnvironmentInformation {
    public rmMachineMeta?: RemoteMachineConfig;
}

export class RemoteConfig {
    public readonly reuse: boolean;
    
    /**
     * Constructor
     * @param reuse If job is reusable for multiple trials
     */
    constructor(reuse: boolean) {
        this.reuse = reuse;
    }
}
