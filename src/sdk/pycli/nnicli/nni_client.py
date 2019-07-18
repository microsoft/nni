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

import sys
import os
import subprocess
import requests

__all__ = [
    'start_nni',
    'stop_nni',
    'set_endpoint',
    'version',
    'get_experiment_status',
    'get_experiment_profile',
    'get_trial_job',
    'list_trial_jobs',
    'get_job_statistics',
    'get_job_metrics',
    'export_data'
]

EXPERIMENT_PATH = 'experiment'
VERSION_PATH = 'version'
STATUS_PATH = 'check-status'
JOB_STATISTICS_PATH = 'job-statistics'
TRIAL_JOBS_PATH = 'trial-jobs'
METRICS_PATH = 'metric-data'
EXPORT_DATA_PATH = 'export-data'

API_ROOT_PATH = 'api/v1/nni'

_api_endpoint = None

def set_endpoint(endpoint):
    """set endpoint of nni rest server for nnicli, for example:
    http://localhost:8080
    """
    global _api_endpoint
    _api_endpoint = endpoint

def _check_endpoint():
    if _api_endpoint is None:
        raise AssertionError("Please call set_endpoint to specify nni endpoint")

def _nni_rest_get(api_path, response_type='json'):
    _check_endpoint()
    uri = os.path.join(_api_endpoint, API_ROOT_PATH, api_path)
    res = requests.get(uri)
    if _http_succeed(res.status_code):
        if response_type == 'json':
            return res.json()
        elif response_type == 'text':
            return res.text
        else:
            raise AssertionError('Incorrect response_type')
    else:
        return None

def _http_succeed(status_code):
    return status_code // 100 == 2

def _create_process(cmd):
    if sys.platform == 'win32':
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    while process.poll() is None:
        output = process.stdout.readline()
        if output:
            print(output.decode('utf-8').strip())

def start_nni(config_file):
    """start nni experiment with specified configuration file"""
    cmd = 'nnictl create --config {}'.format(config_file).split(' ')
    _create_process(cmd)

def stop_nni():
    """stop nni experiment"""
    cmd = 'nnictl stop'.split(' ')
    _create_process(cmd)

def version():
    """return version of nni"""
    return _nni_rest_get(VERSION_PATH, 'text')

def get_experiment_status():
    """return experiment status as a dict"""
    return _nni_rest_get(STATUS_PATH)

def get_experiment_profile():
    """return experiment profile as a dict"""
    return _nni_rest_get(EXPERIMENT_PATH)

def get_trial_job(trial_job_id):
    """return trial job information as a dict"""
    assert trial_job_id is not None
    return _nni_rest_get(os.path.join(TRIAL_JOBS_PATH, trial_job_id))

def list_trial_jobs():
    """return information for all trial jobs as a list"""
    return _nni_rest_get(TRIAL_JOBS_PATH)

def get_job_statistics():
    """return trial job statistics information as a dict"""
    return _nni_rest_get(JOB_STATISTICS_PATH)

def get_job_metrics(trial_job_id=None):
    """return trial job metrics"""
    api_path = METRICS_PATH if trial_job_id is None else os.path.join(METRICS_PATH, trial_job_id)
    return _nni_rest_get(api_path)

def export_data():
    """return exported information for all trial jobs"""
    return _nni_rest_get(EXPORT_DATA_PATH)
