// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import { TrialJobApplicationForm, TrialJobDetail, TrialJobStatus  } from '../../common/trainingService';

/**
 * KubeflowTrialJobDetail
 */
export class KubernetesTrialJobDetail implements TrialJobDetail {
    public id: string;
    public status: TrialJobStatus;
    public submitTime: number;
    public startTime?: number;
    public endTime?: number;
    public tags?: string[];
    public url?: string;
    public workingDirectory: string;
    public form: TrialJobApplicationForm;
    public kubernetesJobName: string;
    public queryJobFailedCount: number;

    constructor(id: string, status: TrialJobStatus, submitTime: number,
                workingDirectory: string, form: TrialJobApplicationForm,
                kubernetesJobName: string, url: string) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.kubernetesJobName = kubernetesJobName;
        this.tags = [];
        this.queryJobFailedCount = 0;
        this.url = url;
    }
}

export const kubernetesScriptFormat: string =
`#!/bin/bash
export NNI_PLATFORM={0}
export NNI_SYS_DIR={1}
export NNI_OUTPUT_DIR={2}
export MULTI_PHASE=false
export NNI_TRIAL_JOB_ID={3}
export NNI_EXP_ID={4}
export NNI_CODE_DIR={5}
export NNI_TRIAL_SEQ_ID={6}
{7}
mkdir -p $NNI_SYS_DIR/code
mkdir -p $NNI_OUTPUT_DIR
cp -r $NNI_CODE_DIR/. $NNI_SYS_DIR/code
sh $NNI_SYS_DIR/install_nni.sh
cd $NNI_SYS_DIR/code
python3 -m nni.tools.trial_tool.trial_keeper --trial_command '{8}' --nnimanager_ip {9} --nnimanager_port {10} \
--nni_manager_version '{11}' --log_collection '{12}' 1>$NNI_OUTPUT_DIR/trialkeeper_stdout 2>$NNI_OUTPUT_DIR/trialkeeper_stderr`;
