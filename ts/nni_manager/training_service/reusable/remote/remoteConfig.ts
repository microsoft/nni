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
