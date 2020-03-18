# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import contextlib
import collections
import os
import socket
import sys
import subprocess
import requests
import time
import ruamel.yaml as yaml

EXPERIMENT_DONE_SIGNAL = 'Experiment done'

GREEN = '\33[32m'
RED = '\33[31m'
CLEAR = '\33[0m'

REST_ENDPOINT = 'http://localhost:8080'
API_ROOT_URL = REST_ENDPOINT + '/api/v1/nni'
EXPERIMENT_URL = API_ROOT_URL + '/experiment'
STATUS_URL = API_ROOT_URL + '/check-status'
TRIAL_JOBS_URL = API_ROOT_URL + '/trial-jobs'
METRICS_URL = API_ROOT_URL + '/metric-data'

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

def get_experiment_id(experiment_url):
    experiment_id = requests.get(experiment_url).json()['id']
    return experiment_id

def get_experiment_dir(experiment_url):
    '''get experiment root directory'''
    experiment_id = get_experiment_id(experiment_url)
    return os.path.join(os.path.expanduser('~'), 'nni', 'experiments', experiment_id)

def get_nni_log_dir(experiment_url):
    '''get nni's log directory from nni's experiment url'''
    return os.path.join(get_experiment_dir(experiment_url), 'log')

def get_nni_log_path(experiment_url):
    '''get nni's log path from nni's experiment url'''
    return os.path.join(get_nni_log_dir(experiment_url), 'nnimanager.log')

def is_experiment_done(nnimanager_log_path):
    '''check if the experiment is done successfully'''
    assert os.path.exists(nnimanager_log_path), 'Experiment starts failed'
    
    with open(nnimanager_log_path, 'r') as f:
        log_content = f.read()

    return EXPERIMENT_DONE_SIGNAL in log_content

def get_experiment_status(status_url):
    nni_status = requests.get(status_url).json()
    return nni_status['status']

def get_trial_stats(trial_jobs_url):
    trial_jobs = requests.get(trial_jobs_url).json()
    trial_stats = collections.defaultdict(int)
    for trial_job in trial_jobs:
        trial_stats[trial_job['status']] += 1
    return trial_stats

def get_trial_jobs(trial_jobs_url, status=None):
    '''Return failed trial jobs'''
    trial_jobs = requests.get(trial_jobs_url).json()
    res = []
    for trial_job in trial_jobs:
        if status is None or trial_job['status'] == status:
            res.append(trial_job)
    return res

def get_failed_trial_jobs(trial_jobs_url):
    '''Return failed trial jobs'''
    return get_trial_jobs(trial_jobs_url, 'FAILED')

def print_trial_job_log(training_service, trial_jobs_url):
    '''Print job log of FAILED trial jobs'''
    trial_jobs = get_trial_jobs(trial_jobs_url)
    for trial_job in trial_jobs:
        log_files = []
        trial_log_dir = os.path.join(get_experiment_dir(EXPERIMENT_URL), 'trials', trial_job['id'])
        if training_service == 'local':
            log_files.append(os.path.join(trial_log_dir, 'stderr'))
            log_files.append(os.path.join(trial_log_dir, 'trial.log'))
        else:
            log_files.append(os.path.join(trial_log_dir, 'stdout_log_collection.log'))
        for log_file in log_files:
            with open(log_file, 'r') as f:
                log_content = f.read()
                print(log_file, flush=True)
                print(log_content, flush=True)

def print_failed_job_log(training_service, trial_jobs_url):
    '''Print job log of FAILED trial jobs'''
    trial_jobs = get_failed_trial_jobs(trial_jobs_url)
    for trial_job in trial_jobs:
        if training_service == 'local':
            if sys.platform == "win32":
                p = trial_job['stderrPath'].split(':')
                log_filename = ':'.join([p[-2], p[-1]])
            else:
                log_filename = trial_job['stderrPath'].split(':')[-1]
        else:
            log_filename = os.path.join(get_experiment_dir(EXPERIMENT_URL), 'trials', trial_job['id'], 'stdout_log_collection.log')
        with open(log_filename, 'r') as f:
            log_content = f.read()
            print(log_filename, flush=True)
            print(log_content, flush=True)

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

def detect_port(port):
    '''Detect if the port is used'''
    socket_test = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        socket_test.connect(('127.0.0.1', int(port)))
        socket_test.close()
        return True
    except:
        return False

def snooze():
    '''Sleep to make sure previous stopped exp has enough time to exit'''
    time.sleep(6)
