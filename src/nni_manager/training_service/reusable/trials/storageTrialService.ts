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

import * as component from "../../../common/component";
import { TrialJobApplicationForm } from "../../../common/trainingService";
import { generateParamFileName } from "../../../common/utils";
import { KILL_TRIAL_JOB, NEW_TRIAL_JOB } from '../../../core/commands';
import { CommandChannel } from "../commandChannel";
import { StorageService } from "../storageService";
import { TrialDetail, TrialService } from "../trial";

@component.Singleton
export class StorageTrialService extends TrialService {
    private commandChannel: CommandChannel | undefined;

    public async config(key: string, value: any): Promise<void> {
        switch (key) {
            case "channel":
                this.commandChannel = value;
                break;
        }
    }

    public async startTrial(trial: TrialDetail): Promise<void> {
        if (trial.environment === undefined) {
            throw new Error(`trialService: environment of trial ${trial.id} shouldn't be undefined!`);
        }
        if (this.commandChannel === undefined) {
            throw new Error(`trialService: commandChannel shouldn't be undefined!`);
        }
        await this.commandChannel.sendCommand(trial.environment, NEW_TRIAL_JOB, trial.settings);
    }

    public async stopTrial(trial: TrialDetail): Promise<void> {
        if (trial.environment === undefined) {
            throw new Error(`trialService: environment of trial ${trial.id} shouldn't be undefined!`);
        }
        if (this.commandChannel === undefined) {
            throw new Error(`trialService: commandChannel shouldn't be undefined!`);
        }
        await this.commandChannel.sendCommand(trial.environment, KILL_TRIAL_JOB, trial.id);
    }

    public async updateTrial(trial: TrialDetail, form: TrialJobApplicationForm): Promise<void> {
        const storageService = component.get<StorageService>(StorageService);
        const fileName = storageService.joinPath(trial.workingDirectory, generateParamFileName(form.hyperParameters))

        // Write file content ( parameter.cfg ) to working folders
        await storageService.save(form.hyperParameters.value, fileName);
    }
}
