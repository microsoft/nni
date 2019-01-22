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

import * as assert from 'assert';
import * as os from 'os';
import * as path from 'path';
import * as component from '../common/component';

@component.Singleton
class ExperimentStartupInfo {
    private experimentId: string = '';
    private newExperiment: boolean = true;
    private basePort: number = -1;
    private initialized: boolean = false;
    private initTrialSequenceID: number = 0;
    private logDir: string = '';
    private logLevel: string = '';

    public setStartupInfo(newExperiment: boolean, experimentId: string, basePort: number, logDir?: string, logLevel?: string): void {
        assert(!this.initialized);
        assert(experimentId.trim().length > 0);

        this.newExperiment = newExperiment;
        this.experimentId = experimentId;
        this.basePort = basePort;
        this.initialized = true;

        if (logDir !== undefined && logDir.length > 0) {
            this.logDir = path.join(logDir, getExperimentId());
        } else {
            this.logDir = path.join(os.homedir(), 'nni', 'experiments', getExperimentId());
        }

        if (logLevel !== undefined && logLevel.length > 1) {
            this.logLevel = logLevel;
        }
    }

    public getExperimentId(): string {
        assert(this.initialized);

        return this.experimentId;
    }

    public getBasePort(): number {
        assert(this.initialized);

        return this.basePort;
    }

    public isNewExperiment(): boolean {
        assert(this.initialized);

        return this.newExperiment;
    }

    public getLogDir(): string {
        assert(this.initialized);

        return this.logDir;
    }

    public getLogLevel(): string {
        assert(this.initialized);

        return this.logLevel;
    }

    public setInitTrialSequenceId(initSequenceId: number): void {
        assert(this.initialized);
        this.initTrialSequenceID = initSequenceId;
    }

    public getInitTrialSequenceId(): number {
        assert(this.initialized);

        return this.initTrialSequenceID;
    }
}

function getExperimentId(): string {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).getExperimentId();
}

function getBasePort(): number {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).getBasePort();
}

function isNewExperiment(): boolean {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).isNewExperiment();
}

function setInitTrialSequenceId(initSequenceId: number): void {
    component.get<ExperimentStartupInfo>(ExperimentStartupInfo).setInitTrialSequenceId(initSequenceId);
}

function getInitTrialSequenceId(): number {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).getInitTrialSequenceId();
}

function getExperimentStartupInfo(): ExperimentStartupInfo {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo);
}

function setExperimentStartupInfo(
    newExperiment: boolean, experimentId: string, basePort: number, logDir?: string, logLevel?: string): void {
    component.get<ExperimentStartupInfo>(ExperimentStartupInfo)
    .setStartupInfo(newExperiment, experimentId, basePort, logDir, logLevel);
}

export { ExperimentStartupInfo, getBasePort, getExperimentId, isNewExperiment, getExperimentStartupInfo,
    setExperimentStartupInfo, setInitTrialSequenceId, getInitTrialSequenceId };
