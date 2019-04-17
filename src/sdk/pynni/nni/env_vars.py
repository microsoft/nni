# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================

import os
from collections import namedtuple

_trial_env_var_dict = {
    'platform': os.environ.get('NNI_PLATFORM'),
    'trial_job_id': os.environ.get('NNI_TRIAL_JOB_ID'),
    'nni_sys_dir': os.environ.get('NNI_SYS_DIR'),
    'output_dir': os.environ.get('NNI_OUTPUT_DIR'),
    'trial_sequence_id': os.environ.get('NNI_TRIAL_SEQ_ID'),
    'multi_phase': os.environ.get('MULTI_PHASE')
}

_dispatcher_env_var_dict = {
    'nni_mode': os.environ.get('NNI_MODE'),
    'nni_checkpoint_dir': os.environ.get('NNI_CHECKPOINT_DIRECTORY'),
    'log_dir': os.environ.get('NNI_LOG_DIRECTORY'),
    'log_level': os.environ.get('NNI_LOG_LEVEL'),
    'include_intermediate_results': os.environ.get('NNI_INCLUDE_INTERMEDIATE_RESULTS')
}

def _load_env_vars(env_var_dict):
    return namedtuple('EnvVars', env_var_dict.keys())(**env_var_dict)

trial_env_vars = _load_env_vars(_trial_env_var_dict)

dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_dict)
