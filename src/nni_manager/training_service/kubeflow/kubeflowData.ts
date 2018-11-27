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

import { JobApplicationForm, TrialJobDetail, TrialJobStatus  } from '../../common/trainingService';

/**
 * KubeflowTrialJobDetail
 */
// tslint:disable-next-line:max-classes-per-file
export class KubeflowTrialJobDetail implements TrialJobDetail {
    public id: string;
    public status: TrialJobStatus;
    public submitTime: number;
    public startTime?: number;
    public endTime?: number;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: JobApplicationForm;
    public kubeflowJobName: string;
    public sequenceId: number;
    public queryJobFailedCount: number;
    public k8sPluralName: string
    
    constructor(id: string, status: TrialJobStatus, submitTime: number,
                workingDirectory: string, form: JobApplicationForm, 
                kubeflowJobName: string, sequenceId: number, url: string, k8sPluralName: string) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.kubeflowJobName = kubeflowJobName;
        this.sequenceId = sequenceId;
        this.tags = [];
        this.queryJobFailedCount = 0;
        this.url = url;
        this.k8sPluralName = k8sPluralName;
    }
}

export type KubeflowTFJobType = 'Created' | 'Running' | 'Failed' | 'Succeeded';