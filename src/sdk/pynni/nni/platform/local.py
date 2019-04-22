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
import sys
import json
import time
import subprocess
import json_tricks

from ..common import init_logger
from ..env_vars import trial_env_vars

_sysdir = trial_env_vars.NNI_SYS_DIR
if not os.path.exists(os.path.join(_sysdir, '.nni')):
    os.makedirs(os.path.join(_sysdir, '.nni'))
_metric_file = open(os.path.join(_sysdir, '.nni', 'metrics'), 'wb')

_outputdir = trial_env_vars.NNI_OUTPUT_DIR
if not os.path.exists(_outputdir):
    os.makedirs(_outputdir)

_nni_platform = trial_env_vars.NNI_PLATFORM
if _nni_platform == 'local':
   _log_file_path = os.path.join(_outputdir, 'trial.log')
   init_logger(_log_file_path)

_multiphase = trial_env_vars.MULTI_PHASE

_param_index = 0

def request_next_parameter():
    metric = json_tricks.dumps({
        'trial_job_id': trial_env_vars.NNI_TRIAL_JOB_ID,
        'type': 'REQUEST_PARAMETER',
        'sequence': 0,
        'parameter_index': _param_index
    })
    send_metric(metric)

def get_next_parameter():
    global _param_index
    params_file_name = ''
    if _multiphase and (_multiphase == 'true' or _multiphase == 'True'):
        params_file_name = ('parameter_{}.cfg'.format(_param_index), 'parameter.cfg')[_param_index == 0]
    else:
        if _param_index > 0:
            return None
        elif _param_index == 0:
            params_file_name = 'parameter.cfg'
        else:
            raise AssertionError('_param_index value ({}) should >=0'.format(_param_index))
    
    params_filepath = os.path.join(_sysdir, params_file_name)
    if not os.path.isfile(params_filepath):
        request_next_parameter()
    while not (os.path.isfile(params_filepath) and os.path.getsize(params_filepath) > 0):
        time.sleep(3)
    params_file = open(params_filepath, 'r')
    params = json.load(params_file)
    _param_index += 1
    return params

def send_metric(string):
    if _nni_platform != 'local':
        data = (string).encode('utf8')
        assert len(data) < 1000000, 'Metric too long'    
        print('NNISDK_ME%s' % (data), flush=True)
    else:
        data = (string + '\n').encode('utf8')
        assert len(data) < 1000000, 'Metric too long'    
        _metric_file.write(b'ME%06d%b' % (len(data), data))
        _metric_file.flush()
        if sys.platform == "win32":
            file = open(_metric_file.name)
            file.close()
        else:
            subprocess.run(['touch', _metric_file.name], check = True)

def get_sequence_id():
    return trial_env_vars.NNI_TRIAL_SEQ_ID
