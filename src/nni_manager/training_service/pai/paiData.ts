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

import { JobApplicationForm, TrialJobDetail, TrialJobStatus  } from 'common/trainingService';

export class PAITrialJobDetail implements TrialJobDetail {
    public id: string;
    public status: TrialJobStatus;
    public paiJobName: string;
    public submitTime: number;
    public startTime?: number;
    public endTime?: number;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: JobApplicationForm;

    constructor(id: string, status: TrialJobStatus, paiJobName : string, 
            submitTime: number, workingDirectory: string, form: JobApplicationForm) {
        this.id = id;
        this.status = status;
        this.paiJobName = paiJobName;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.tags = [];
    }
}

export const PAI_TRIAL_COMMAND_FORMAT: string =
`pip3 install -v --user git+https://github.com/yds05/nni.git@master
&& export NNI_PLATFORM=pai NNI_SYS_DIR={0} NNI_OUTPUT_DIR={0} NNI_TRIAL_JOB_ID={1} NNI_EXP_ID={2} 
&& cd $NNI_SYS_DIR && mkdir .nni 
&& python3 -m trial.trial_keeper --trial_command '{3}' --nnimanager_ip '{4}' --pai_hdfs_output_dir '{5}' 
--pai_hdfs_host '{6}' --pai_user_name {7}`;

export const PAI_OUTPUT_DIR_FORMAT: string = 
`hdfs://{0}:9000/`;
