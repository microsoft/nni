/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';

import { Logger, getLogger } from "../../common/log";
import { TrialJobApplicationForm, TrialJobDetail, TrialJobStatus } from "../../common/trainingService";
import { EnvironmentInformation } from "./environment";

export abstract class TrialService {
    protected readonly log: Logger;

    public abstract config(key: string, value: string): Promise<void>;
    public abstract refreshTrialsStatus(trials: TrialDetail[]): Promise<void>;
    public abstract updateTrial(trial: TrialDetail, form: TrialJobApplicationForm): Promise<void>;
    public abstract startTrial(trial: TrialDetail): Promise<void>;
    public abstract stopTrial(trial: TrialDetail): Promise<void>;

    constructor() {
        this.log = getLogger();
    }
}

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
    public nodeExitResults: TrialJobStatus[];

    public readonly TRIAL_METADATA_DIR = ".nni";

    constructor(id: string, status: TrialJobStatus, submitTime: number,
        workingDirectory: string, form: TrialJobApplicationForm) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.tags = [];
        this.nodeExitResults = [];
    }
}
