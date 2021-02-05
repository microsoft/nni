// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { TrialJobApplicationForm, TrialJobDetail, TrialJobStatus } from "../../common/trainingService";
import { GPUInfo } from "../../training_service/common/gpuData";
import { EnvironmentInformation, NodeInformation } from "./environment";

export class TrialDetail implements TrialJobDetail {
    public id: string;
    public status: TrialJobStatus;
    public submitTime: number;
    public startTime?: number;
    public endTime?: number;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: TrialJobApplicationForm;
    public isEarlyStopped?: boolean;
    public environment?: EnvironmentInformation;

    // init settings of trial
    public settings = {};
    // it's used to aggregate node status for multiple node trial
    public nodes: Map<string, NodeInformation>;
    // assigned GPUs for multi-trial scheduled.
    public assignedGpus: GPUInfo[] | undefined;

    public readonly TRIAL_METADATA_DIR = ".nni";

    constructor(id: string, status: TrialJobStatus, submitTime: number,
        workingDirectory: string, form: TrialJobApplicationForm) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.tags = [];
        this.nodes = new Map<string, NodeInformation>();
    }
}
