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
    public hdfsLogPath: string;

    constructor(id: string, status: TrialJobStatus, paiJobName : string, 
            submitTime: number, workingDirectory: string, form: JobApplicationForm, hdfsLogPath: string) {
        this.id = id;
        this.status = status;
        this.paiJobName = paiJobName;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.tags = [];
        this.hdfsLogPath = hdfsLogPath;
    }
}

export const RAI_INSTALL_NNI_SHELL_FORMAT: string = 
`#!/bin/bash
if python3 -c 'import nni' > /dev/null 2>&1; then
  # nni module is already installed, skip
  return
else
  # Install nni
  pip3 install -v --user git+https://github.com/Microsoft/nni.git@v0.2
fi`;

export const PAI_TRIAL_COMMAND_FORMAT: string =
`export NNI_PLATFORM=pai NNI_SYS_DIR={0} NNI_OUTPUT_DIR={1} NNI_TRIAL_JOB_ID={2} NNI_EXP_ID={3} 
&& cd $NNI_SYS_DIR && mkdir .nni 
&& sh install_nni.sh 
&& python3 -m trial_tool.trial_keeper --trial_command '{4}' --nnimanager_ip '{5}' --pai_hdfs_output_dir '{6}' 
--pai_hdfs_host '{7}' --pai_user_name {8}`;

export const PAI_OUTPUT_DIR_FORMAT: string = 
`hdfs://{0}:9000/`;

export const PAI_LOG_PATH_FORMAT: string = 
`http://{0}:50070/explorer.html#{1}`
