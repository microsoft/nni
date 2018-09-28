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


import json
import os

from ..common import init_logger

_sysdir = os.environ['NNI_SYS_DIR']
if not os.path.exists(os.path.join(_sysdir, '.nni')):
    os.makedirs(os.path.join(_sysdir, '.nni'))
_metric_file = open(os.path.join(_sysdir, '.nni', 'metrics'), 'wb')

_outputdir = os.environ['NNI_OUTPUT_DIR']
_log_file_path = os.path.join(_outputdir, 'trial.log')
init_logger(_log_file_path)


def get_parameters():
    params_file = open(os.path.join(_sysdir, 'parameter.cfg'), 'r')
    return json.load(params_file)

def send_metric(string):
    data = (string + '\n').encode('utf8')
    assert len(data) < 1000000, 'Metric too long'
    _metric_file.write(b'ME%06d%b' % (len(data), data))
    _metric_file.flush()
