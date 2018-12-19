# ============================================================================================================================== #
# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================================================================== #

import argparse
import errno
import json
import os
import re
import requests

from datetime import datetime
from .constants import BASE_URL
from .log_utils import LogType, nni_log
from .rest_utils import rest_get, rest_post, rest_put, rest_delete
from .url_utils import gen_update_metrics_url

NNI_SYS_DIR = os.environ['NNI_SYS_DIR']
NNI_TRIAL_JOB_ID = os.environ['NNI_TRIAL_JOB_ID']
NNI_EXP_ID = os.environ['NNI_EXP_ID']
LEN_FIELD_SIZE = 6
MAGIC = 'ME'

class TrialMetricsReader():
    '''
    Read metrics data from a trial job
    '''
    def __init__(self):
        metrics_base_dir = os.path.join(NNI_SYS_DIR, '.nni')
        self.offset_filename = os.path.join(metrics_base_dir, 'metrics_offset')
        self.metrics_filename = os.path.join(metrics_base_dir, 'metrics')
        if not os.path.exists(metrics_base_dir):
            os.makedirs(metrics_base_dir)

    def _metrics_file_is_empty(self):
        if not os.path.isfile(self.metrics_filename):
            return True
        statinfo = os.stat(self.metrics_filename)
        return statinfo.st_size == 0

    def _get_offset(self):
        offset = 0
        if os.path.isfile(self.offset_filename):
            with open(self.offset_filename, 'r') as f:
                offset = int(f.readline())
        return offset

    def _write_offset(self, offset):
        statinfo = os.stat(self.metrics_filename)
        if offset < 0 or offset > statinfo.st_size:
            raise ValueError('offset value is invalid: {}'.format(offset))

        with open(self.offset_filename, 'w') as f:
            f.write(str(offset)+'\n')

    def _read_all_available_records(self, offset):
        new_offset = offset
        metrics = []
        with open(self.metrics_filename, 'r') as f:            
            f.seek(offset)
            while True:
                magic_string = f.read(len(MAGIC))
                # empty data means EOF
                if not magic_string:
                    break
                nni_log(LogType.Info, 'Metrics file offset is {}'.format(offset))
                strdatalen = f.read(LEN_FIELD_SIZE)
                # empty data means EOF
                if not strdatalen:
                    raise ValueError("metric file {} format error after offset: {}.".format(self.metrics_filename, new_offset))
                datalen = int(strdatalen)
                data = f.read(datalen)

                if datalen > 0 and len(data) == datalen:
                    nni_log(LogType.Info, 'data is \'{}\''.format(data))
                    new_offset = f.tell()
                    metrics.append(data)
                else:
                    raise ValueError("metric file {} format error after offset: {}.".format(self.metrics_filename, new_offset))
        self._write_offset(new_offset)
        return metrics

    def read_trial_metrics(self):
        '''
        Read available metrics data for a trial
        '''
        if self._metrics_file_is_empty():
            return []

        offset = self._get_offset()
        return self._read_all_available_records(offset)

def read_experiment_metrics(nnimanager_ip, nnimanager_port):
    '''
    Read metrics data for specified trial jobs
    '''
    result = {}
    try:
        reader = TrialMetricsReader()
        result['jobId'] = NNI_TRIAL_JOB_ID
        result['metrics'] = reader.read_trial_metrics()    
        if len(result['metrics']) > 0:
            nni_log(LogType.Info, 'Result metrics is {}'.format(json.dumps(result)))
            response = rest_post(gen_update_metrics_url(BASE_URL.format(nnimanager_ip), nnimanager_port, NNI_EXP_ID, NNI_TRIAL_JOB_ID), json.dumps(result), 10)
            nni_log(LogType.Info,'Report metrics to NNI manager completed, http response code is {}'.format(response.status_code))
    except Exception as e:
        #Error logging
        nni_log(LogType.Error, 'Error when reading metrics data: ' + str(e))

    return json.dumps(result)