// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

export const DLTS_TRIAL_COMMAND_FORMAT: string =
`export NNI_PLATFORM=dlts NNI_SYS_DIR={0} NNI_OUTPUT_DIR={1} NNI_TRIAL_JOB_ID={2} NNI_EXP_ID={3} NNI_TRIAL_SEQ_ID={4} MULTI_PHASE={5} \
&& cd $NNI_SYS_DIR && sh install_nni.sh \
&& cd '{6}' && python3 -m nni.tools.trial_tool.trial_keeper --trial_command '{7}' \
--nnimanager_ip '{8}' --nnimanager_port '{9}' --nni_manager_version '{10}' --log_collection '{11}'`;
