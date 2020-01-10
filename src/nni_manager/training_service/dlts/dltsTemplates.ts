
export const DLTS_TRIAL_COMMAND_FORMAT: string =
`export NNI_PLATFORM=dlts NNI_SYS_DIR={0} NNI_OUTPUT_DIR={1} NNI_TRIAL_JOB_ID={2} NNI_EXP_ID={3} NNI_TRIAL_SEQ_ID={4} MULTI_PHASE={5} \
&& cd $NNI_SYS_DIR && python3 -m pip install --user nni \
&& python3 -m nni_trial_tool.trial_keeper --trial_command '{6}' --nnimanager_ip '{7}' --nnimanager_port '{8}' \
--nni_manager_version '{9}' --log_collection '{10}'`;