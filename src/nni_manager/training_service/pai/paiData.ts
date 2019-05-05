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
    public sequenceId: number;
    public hdfsLogPath: string;
    public isEarlyStopped?: boolean;

    constructor(id: string, status: TrialJobStatus, paiJobName : string, 
            submitTime: number, workingDirectory: string, form: JobApplicationForm, sequenceId: number, hdfsLogPath: string) {
        this.id = id;
        this.status = status;
        this.paiJobName = paiJobName;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.sequenceId = sequenceId;
        this.tags = [];
        this.hdfsLogPath = hdfsLogPath;
    }
}

export const PAI_INSTALL_NNI_SHELL_FORMAT: string = 
`#!/bin/bash
if python3 -c 'import nni' > /dev/null 2>&1; then
  # nni module is already installed, skip
  return
else
  # Install nni
  python3 -m pip install --user nni
fi`;

export const PAI_TRIAL_COMMAND_FORMAT: string =
`export NNI_PLATFORM=pai NNI_SYS_DIR={0} NNI_OUTPUT_DIR={1} NNI_TRIAL_JOB_ID={2} NNI_EXP_ID={3} NNI_TRIAL_SEQ_ID={4}
&& cd $NNI_SYS_DIR && sh install_nni.sh 
&& python3 -m nni_trial_tool.trial_keeper --trial_command '{5}' --nnimanager_ip '{6}' --nnimanager_port '{7}' 
--pai_hdfs_output_dir '{8}' --pai_hdfs_host '{9}' --pai_user_name {10} --nni_hdfs_exp_dir '{11}' --webhdfs_path '/webhdfs/api/v1' --nni_manager_version '{12}' --log_collection '{13}'`;

export const PAI_OUTPUT_DIR_FORMAT: string = 
`hdfs://{0}:9000/`;

export const PAI_LOG_PATH_FORMAT: string = 
`http://{0}/webhdfs/explorer.html#{1}`
