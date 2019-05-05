# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import contextlib
import collections
import json
import os
import sys
import subprocess
import requests
import ruamel.yaml as yaml

EXPERIMENT_DONE_SIGNAL = '"Experiment done"'

GREEN = '\33[32m'
RED = '\33[31m'
CLEAR = '\33[0m'

REST_ENDPOINT = 'http://localhost:8080/api/v1/nni'
EXPERIMENT_URL = REST_ENDPOINT + '/experiment'
STATUS_URL = REST_ENDPOINT + '/check-status'
TRIAL_JOBS_URL = REST_ENDPOINT + '/trial-jobs'
METRICS_URL = REST_ENDPOINT + '/metric-data'

def read_last_line(file_name):
    '''read last line of a file and return None if file not found'''
    try:
        *_, last_line = open(file_name)
        return last_line.strip()
    except (FileNotFoundError, ValueError):
        return None

def remove_files(file_list):
    '''remove a list of files'''
    for file_path in file_list:
        with contextlib.suppress(FileNotFoundError):
            os.remove(file_path)

def get_yml_content(file_path):
    '''Load yaml file content'''
    with open(file_path, 'r') as file:
        return yaml.load(file, Loader=yaml.Loader)

def dump_yml_content(file_path, content):
    '''Dump yaml file content'''
    with open(file_path, 'w') as file:
        file.write(yaml.dump(content, default_flow_style=False))

def setup_experiment(installed=True):
    '''setup the experiment if nni is not installed'''
    if not installed:
        os.environ['PATH'] = os.environ['PATH'] + ':' + os.getcwd()
        sdk_path = os.path.abspath('../src/sdk/pynni')
        cmd_path = os.path.abspath('../tools')
        pypath = os.environ.get('PYTHONPATH')
        if pypath:
            pypath = ':'.join([pypath, sdk_path, cmd_path])
        else:
            pypath = ':'.join([sdk_path, cmd_path])
        os.environ['PYTHONPATH'] = pypath

def fetch_nni_log_path(experiment_url):
    '''get nni's log path from nni's experiment url'''
    experiment_profile = requests.get(experiment_url)
    experiment_id = json.loads(experiment_profile.text)['id']
    experiment_path = os.path.join(os.path.expanduser('~'), 'nni', 'experiments', experiment_id)
    nnimanager_log_path = os.path.join(experiment_path, 'log', 'nnimanager.log')

    return nnimanager_log_path

def is_experiment_done(nnimanager_log_path):
    '''check if the experiment is done successfully'''
    assert os.path.exists(nnimanager_log_path), 'Experiment starts failed'
    if sys.platform == "win32":
        cmds = ['type', nnimanager_log_path, '|', 'find', EXPERIMENT_DONE_SIGNAL]
    else:
        cmds = ['cat', nnimanager_log_path, '|', 'grep', EXPERIMENT_DONE_SIGNAL]
    completed_process = subprocess.run(' '.join(cmds), shell=True)

    return completed_process.returncode == 0

def get_experiment_status(status_url):
    nni_status = requests.get(status_url).json()
    #print(nni_status)
    return nni_status['status']

def get_succeeded_trial_num(trial_jobs_url):
    trial_jobs = requests.get(trial_jobs_url).json()
    print(trial_jobs)
    num_succeed = 0
    for trial_job in trial_jobs:
        if trial_job['status'] in ['SUCCEEDED', 'EARLY_STOPPED']:
            num_succeed += 1
    print('num_succeed:', num_succeed)
    return num_succeed

def print_stderr(trial_jobs_url):
    trial_jobs = requests.get(trial_jobs_url).json()
    for trial_job in trial_jobs:
        if trial_job['status'] == 'FAILED':
            stderr_path = trial_job['stderrPath'].split(':')[-1]
            if sys.platform == "win32":
                subprocess.run(['type', stderr_path], shell=True)
            else:
                subprocess.run(['cat', stderr_path])

def parse_max_duration_time(max_exec_duration):
    unit = max_exec_duration[-1]
    time = max_exec_duration[:-1]
    units_dict = {'s':1, 'm':60, 'h':3600, 'd':86400}
    return int(time) * units_dict[unit]

def deep_update(source, overrides):
    """Update a nested dictionary or similar mapping.

    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source
