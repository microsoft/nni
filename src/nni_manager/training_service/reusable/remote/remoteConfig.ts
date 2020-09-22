// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { EnvironmentInformation } from '../environment';
import { GPUInfo, GPUSummary, ScheduleResultType } from '../../common/gpuData';
import { RemoteMachineMeta } from '../../remote_machine/remoteMachineData';
import {
    ExecutorManager,
    RemoteMachineScheduleInfo, RemoteMachineScheduleResult, RemoteMachineTrialJobDetail
} from '../../remote_machine/remoteMachineData';


/**
 * work around here, need RemoteMachineTrialJobDetail data structure to schedule machine
 */
export class RemoteMachineMetaDetail extends RemoteMachineTrialJobDetail {
}

/**
 * RemoteMachineEnvironmentInformation
 */
export class RemoteMachineEnvironmentInformation extends EnvironmentInformation {
    public gpuIndices?: GPUInfo[];
    public rmMachineMetaDetail?: RemoteMachineMetaDetail;
}