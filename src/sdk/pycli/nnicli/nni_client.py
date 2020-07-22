# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

""" A python wrapper for nni rest api

Example:

import nnicli as nc

nc.start_experiment('../../../../examples/trials/mnist-pytorch/config.yml')

nc.set_endpoint('http://localhost:8080')

print(nc.version())
print(nc.get_experiment_status())

print(nc.get_job_statistics())
print(nc.list_trial_jobs())

nc.stop_experiment()

"""

import sys
import os
import subprocess
import requests

__all__ = [
    'start_experiment',
    'set_endpoint',
    'stop_experiment',
    'resume_experiment',
    'view_experiment',
    'update_searchspace',
    'update_concurrency',
    'update_duration',
    'update_trailnum',
    'stop_experiment',
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
    """
    Set endpoint of nni rest server for nnicli, i.e., the url of Web UI.
    Everytime you want to change experiment, call this function first.

    Parameters
    ----------
    endpoint: str
        the endpoint of nni rest server for nnicli
    """
    global _api_endpoint
    _api_endpoint = endpoint

def _check_endpoint():
    if _api_endpoint is None:
        raise AssertionError("Please call set_endpoint to specify nni endpoint")

def _nni_rest_get(api_path, response_type='json'):
    _check_endpoint()
    uri = '{}/{}/{}'.format(_api_endpoint, API_ROOT_PATH, api_path)
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
    return process.returncode

def start_experiment(config_file, port=None, debug=False):
    """
    Start an experiment with specified configuration file.

    Parameters
    ----------
    config_file: str
        path to the config file
    port: int
        the port of restful server, bigger than 1024
    debug: boolean
        set debug mode
    """
    cmd = 'nnictl create --config {}'.format(config_file).split(' ')
    if port:
        cmd += '--port {}'.format(port).split(' ')
    if debug:
        cmd += ['--debug']
    if _create_process(cmd) != 0:
        raise RuntimeError('Failed to start experiment.')

def resume_experiment(exp_id, port=None, debug=False):
    """
    Resume a stopped experiment with specified experiment id

    Parameters
    ----------
    exp_id: str
        experiment id
    port: int
        the port of restful server, bigger than 1024
    debug: boolean
        set debug mode
    """
    cmd = 'nnictl resume {}'.format(exp_id).split(' ')
    if port:
        cmd += '--port {}'.format(port).split(' ')
    if debug:
        cmd += ['--debug']
    if _create_process(cmd) != 0:
        raise RuntimeError('Failed to resume experiment.')

def view_experiment(exp_id, port=None):
    """
    View a stopped experiment with specified experiment id

    Parameters
    ----------
    exp_id: str
        experiment id
    port: int
        the port of restful server, bigger than 1024
    """
    cmd = 'nnictl view {}'.format(exp_id).split(' ')
    if port:
        cmd += '--port {}'.format(port).split(' ')
    if _create_process(cmd) != 0:
        raise RuntimeError('Failed to view experiment.')

def update_searchspace(filename, exp_id=None):
    """
    Update an experiment's search space

    Parameters
    ----------
    filename: str
        path to the searchspace file
    exp_id: str
        experiment id
    """
    if not exp_id:
        cmd = 'nnictl update searchspace --filename {}'.format(filename).split(' ')
    else:
        cmd = 'nnictl update searchspace {} --filename {}'.format(exp_id, filename).split(' ')
    if _create_process(cmd) != 0:
        raise RuntimeError('Failed to update searchspace.')

def update_concurrency(value, exp_id=None):
    """
    Update an experiment's concurrency

    Parameters
    ----------
    value: int
        new concurrency value
    exp_id: str
        experiment id
    """
    if not exp_id:
        cmd = 'nnictl update concurrency --value {}'.format(value).split(' ')
    else:
        cmd = 'nnictl update concurrency {} --value {}'.format(exp_id, value).split(' ')
    if _create_process(cmd) != 0:
        raise RuntimeError('Failed to update concurrency.')

def update_duration(value, exp_id=None):
    """
    Update an experiment's duration

    Parameters
    ----------
    value: str
        SUFFIX may be 's' for seconds (the default), 'm' for minutes, 'h' for hours or 'd' for days. e.g., '1m', '2h'
    exp_id: str
        experiment id
    """
    if not exp_id:
        cmd = 'nnictl update duration --value {}'.format(value).split(' ')
    else:
        cmd = 'nnictl update duration {} --value {}'.format(exp_id, value).split(' ')
    if _create_process(cmd) != 0:
        raise RuntimeError('Failed to update duration.')

def update_trailnum(value, exp_id=None):
    """
    Update an experiment's maxtrialnum

    Parameters
    ----------
    value: int
        new trailnum value
    exp_id: str
        experiment id
    """
    if not exp_id:
        cmd = 'nnictl update trialnum --value {}'.format(value).split(' ')
    else:
        cmd = 'nnictl update trialnum {} --value {}'.format(exp_id, value).split(' ')
    if _create_process(cmd) != 0:
        raise RuntimeError('Failed to update trailnum.')

def stop_experiment(exp_id=None, port=None, stop_all=False):
    """Stop an experiment.

    Parameters
    ----------
    exp_id: str
        experiment id
    port: int
        the port of restful server
    stop_all: boolean
        if set to True, all the experiments will be stopped

    Note that if stop_all is set to true, exp_id and port will be ignored. Otherwise
    exp_id and port must correspond to the same experiment if they are both set.
    """
    if stop_all:
        cmd = 'nnictl stop --all'.split(' ')
    else:
        cmd = 'nnictl stop'.split(' ')
        if exp_id:
            cmd += [exp_id]
        if port:
            cmd += '--port {}'.format(port).split(' ')
    if _create_process(cmd) != 0:
        raise RuntimeError('Failed to stop experiment.')

def version():
    """
    Return version of nni.
    """
    return _nni_rest_get(VERSION_PATH, 'text')

def get_experiment_status():
    """
    Return experiment status as a dict.
    """
    return _nni_rest_get(STATUS_PATH)

def get_experiment_profile():
    """
    Return experiment profile as a dict.
    """
    return _nni_rest_get(EXPERIMENT_PATH)

def get_trial_job(trial_job_id):
    """
    Return trial job information as a dict.

    Parameters
    ----------
    trial_job_id: str
        trial id
    """
    assert trial_job_id is not None
    return _nni_rest_get(os.path.join(TRIAL_JOBS_PATH, trial_job_id))

def list_trial_jobs():
    """
    Return information for all trial jobs as a list.
    """
    return _nni_rest_get(TRIAL_JOBS_PATH)

def get_job_statistics():
    """
    Return trial job statistics information as a dict.
    """
    return _nni_rest_get(JOB_STATISTICS_PATH)

def get_job_metrics(trial_job_id=None):
    """
    Return trial job metrics.

    Parameters
    ----------
    trial_job_id: str
        trial id. if this parameter is None, all trail jobs' metrics will be returned.
    """
    api_path = METRICS_PATH if trial_job_id is None else os.path.join(METRICS_PATH, trial_job_id)
    return _nni_rest_get(api_path)

def export_data():
    """
    Return exported information for all trial jobs.
    """
    return _nni_rest_get(EXPORT_DATA_PATH)
