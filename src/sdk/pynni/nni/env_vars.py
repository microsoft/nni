# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import namedtuple


_trial_env_var_names = [
    'NNI_PLATFORM',
    'NNI_EXP_ID',
    'NNI_TRIAL_JOB_ID',
    'NNI_SYS_DIR',
    'NNI_OUTPUT_DIR',
    'NNI_TRIAL_SEQ_ID',
    'MULTI_PHASE'
]

_dispatcher_env_var_names = [
    'SDK_PROCESS',
    'NNI_MODE',
    'NNI_CHECKPOINT_DIRECTORY',
    'NNI_LOG_DIRECTORY',
    'NNI_LOG_LEVEL',
    'NNI_INCLUDE_INTERMEDIATE_RESULTS'
]

def _load_env_vars(env_var_names):
    env_var_dict = {k: os.environ.get(k) for k in env_var_names}
    return namedtuple('EnvVars', env_var_names)(**env_var_dict)

trial_env_vars = _load_env_vars(_trial_env_var_names)

dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_names)
