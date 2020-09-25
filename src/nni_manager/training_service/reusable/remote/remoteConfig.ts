// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { EnvironmentInformation } from '../environment';
import { RemoteMachineTrialJobDetail } from '../../remote_machine/remoteMachineData';
import { TrialJobApplicationForm } from '../../../common/trainingService';


/**
 * work around here, need RemoteMachineTrialJobDetail data structure to schedule machines
 */
export class RemoteMachineMetaDetail extends RemoteMachineTrialJobDetail {
    constructor() {
        // work around, the form data is a placeholder
        const form: TrialJobApplicationForm = {
            sequenceId: 0,
            hyperParameters: {
                value: '',
                index: 0
            }
        };
        super('', 'WAITING', 1, '', form);
    }
}

/**
 * RemoteMachineEnvironmentInformation
 */
export class RemoteMachineEnvironmentInformation extends EnvironmentInformation {
    public rmMachineMetaDetail?: RemoteMachineMetaDetail;
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