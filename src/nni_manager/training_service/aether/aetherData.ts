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

import { ChildProcess } from 'child_process';
import { Deferred } from 'ts-deferred';
import {JobApplicationForm, TrialJobDetail, TrialJobStatus} from '../../common/trainingService';
import { TrialConfig } from '../../training_service/common/trialConfig';

/**
 * AetherTrialJobDetail
 */
class AetherTrialJobDetail implements TrialJobDetail {
    public readonly id: string;
    public status: TrialJobStatus;
    public readonly submitTime: number;
    public startTime?: number;
    public endTime?: number;
    public url: string;
    public guid: Deferred<string>; // GUID of Aether Experiment
    public readonly workingDirectory: string;
    public form: JobApplicationForm;
    public readonly sequenceId: number;
    public isEarlyStopped?: boolean;
    public clientProc: ChildProcess;
    public readonly aetherConfig: AetherConfig;

    constructor(
        id: string,
        status: TrialJobStatus,
        submitTime: number,
        workingDirectory: string,
        form: JobApplicationForm,
        sequenceId: number,
        clientProc: ChildProcess,
        aetherConfig: AetherConfig
    ) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.url = '';
        this.guid = new Deferred<string>();
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.sequenceId = sequenceId;
        this.isEarlyStopped = false;
        this.clientProc = clientProc;
        this.aetherConfig = aetherConfig;
    }
}

/**
 * AetherConfig, contains configuration for aether experiment
 */
class AetherConfig extends TrialConfig {
    public baseGraph: string;
    public outputNodeAlias: string;
    public outputName: string;

    constructor(command: string, codeDir: string, gpuNum: number, baseGraph: string, outputNodeAlias: string, outputName: string) {
        super(command, codeDir, gpuNum);
        this.baseGraph = baseGraph;
        this.outputNodeAlias = outputNodeAlias;
        this.outputName = outputName;
    }
}

export {AetherTrialJobDetail, AetherConfig};
