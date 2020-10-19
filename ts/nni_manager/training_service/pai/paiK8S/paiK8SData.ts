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

export const PAI_K8S_TRIAL_COMMAND_FORMAT: string =
`export NNI_PLATFORM=pai NNI_SYS_DIR={0} NNI_OUTPUT_DIR={1} NNI_TRIAL_JOB_ID={2} NNI_EXP_ID={3} NNI_TRIAL_SEQ_ID={4} MULTI_PHASE={5} \
&& NNI_CODE_DIR={6} && mkdir -p $NNI_SYS_DIR/code && cp -r $NNI_CODE_DIR/. $NNI_SYS_DIR/code && sh $NNI_SYS_DIR/install_nni.sh \
&& cd $NNI_SYS_DIR/code && python3 -m nni.tools.trial_tool.trial_keeper --trial_command '{7}' --nnimanager_ip '{8}' --nnimanager_port '{9}' \
--nni_manager_version '{10}' --log_collection '{11}' | tee $NNI_OUTPUT_DIR/trial.log`;
