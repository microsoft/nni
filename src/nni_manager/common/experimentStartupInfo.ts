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
import * as component from '../common/component';

@component.Singleton
class ExperimentStartupInfo {
    private experimentId: string = '';
    private newExperiment: boolean = true;
    private initialized: boolean = false;

    public setStartupInfo(newExperiment: boolean, experimentId: string): void {
        assert(!this.initialized);
        assert(experimentId.trim().length > 0);

        this.newExperiment = newExperiment;
        this.experimentId = experimentId;
        this.initialized = true;
    }

    public getExperimentId(): string {
        assert(this.initialized);

        return this.experimentId;
    }

    public isNewExperiment(): boolean {
        assert(this.initialized);

        return this.newExperiment;
    }
}

function getExperimentId(): string {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).getExperimentId();
}

function isNewExperiment(): boolean {
    return component.get<ExperimentStartupInfo>(ExperimentStartupInfo).isNewExperiment();
}

function setExperimentStartupInfo(newExperiment: boolean, experimentId: string): void {
    component.get<ExperimentStartupInfo>(ExperimentStartupInfo).setStartupInfo(newExperiment, experimentId);
}

export { ExperimentStartupInfo, getExperimentId, isNewExperiment, setExperimentStartupInfo };
