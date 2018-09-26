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

import json_tricks
import json
import os
import time

from ..common import init_logger, env_args


_dir = os.environ['NNI_SYS_DIR']
_metric_file = open(os.path.join(_dir, '.nni', 'metrics'), 'wb')
_param_index = 0

_log_file_path = os.path.join(_dir, 'trial.log')
init_logger(_log_file_path)

def _send_request_parameter_metric():
    metric = json_tricks.dumps({
        'trial_job_id': env_args.trial_job_id,
        'type': 'REQUEST_PARAMETER',
        'sequence': 0,
        'parameter_index': _param_index
    })
    send_metric(metric)

def get_parameters():
    global _param_index
    params_filepath = os.path.join(_dir, 'parameter_{}.cfg'.format(_param_index))
    if not os.path.isfile(params_filepath):
        _send_request_parameter_metric()
    while not os.path.isfile(params_filepath):
        time.sleep(3)
    params_file = open(params_filepath, 'r')
    params = json.load(params_file)
    _param_index += 1
    return params

def send_metric(string):
    data = (string + '\n').encode('utf8')
    assert len(data) < 1000000, 'Metric too long'
    _metric_file.write(b'ME%06d%b' % (len(data), data))
    _metric_file.flush()
