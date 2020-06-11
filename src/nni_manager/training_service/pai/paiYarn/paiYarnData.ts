// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

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
`export NNI_PLATFORM=paiYarn NNI_SYS_DIR={0} NNI_OUTPUT_DIR={1} NNI_TRIAL_JOB_ID={2} NNI_EXP_ID={3} NNI_TRIAL_SEQ_ID={4} MULTI_PHASE={5} \
&& cd $NNI_SYS_DIR && sh install_nni.sh \
&& python3 -m nni_trial_tool.trial_keeper --trial_command '{6}' --nnimanager_ip '{7}' --nnimanager_port '{8}' \
--pai_hdfs_output_dir '{9}' --pai_hdfs_host '{10}' --pai_user_name {11} --nni_hdfs_exp_dir '{12}' --webhdfs_path '/webhdfs/api/v1' \
--nni_manager_version '{13}' --log_collection '{14}'`;

export const PAI_LOG_PATH_FORMAT: string =
`http://{0}/webhdfs/explorer.html#{1}`;
