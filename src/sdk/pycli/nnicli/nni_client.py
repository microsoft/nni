# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

""" A python wrapper for nni rest api

Example:

from nnicli import NNIExperiment

exp.start_experiment('../../../../examples/trials/mnist-pytorch/config.yml')

exp.update_concurrency(3)

print(exp.get_experiment_status())
print(exp.get_job_statistics())
print(exp.list_trial_jobs())

exp.stop_experiment()

"""

import sys
import os
import subprocess
import re
import requests

__all__ = [
    'NNIExperiment'
]

EXPERIMENT_PATH = 'experiment'
STATUS_PATH = 'check-status'
JOB_STATISTICS_PATH = 'job-statistics'
TRIAL_JOBS_PATH = 'trial-jobs'
METRICS_PATH = 'metric-data'
EXPORT_DATA_PATH = 'export-data'
API_ROOT_PATH = 'api/v1/nni'

def _nni_rest_get(endpoint, api_path, response_type='json'):
    _check_endpoint(endpoint)
    uri = '{}/{}/{}'.format(endpoint.strip('/'), API_ROOT_PATH, api_path)
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

def _check_endpoint(endpoint):
    if endpoint is None:
        raise AssertionError("This instance hasn't been connect to an experiment.")

class NNIExperiment:
    def __init__(self):
        self.endpoint = None
        self.exp_id = None
        self.port = None

    def start_experiment(self, config_file, port=None, debug=False):
        """
        Start an experiment with specified configuration file and connect to it.

        Parameters
        ----------
        config_file: str
            path to the config file
        port: int
            the port of restful server, bigger than 1024
        debug: boolean
            set debug mode
        """
        if self.endpoint is not None:
            raise RuntimeError('This instance has been connected to an experiment.')
        cmd = 'nnictl create --config {}'.format(config_file).split(' ')
        if port:
            cmd += '--port {}'.format(port).split(' ')
        if debug:
            cmd += ['--debug']
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to start experiment, please check your config.')
        else:
            if port:
                self.port = port
            else:
                self.port = 8080
            self.endpoint = 'http://localhost:{}'.format(self.port)
            self.exp_id = self.get_experiment_profile()['id']

    def resume_experiment(self, exp_id, port=None, debug=False):
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
        if self.endpoint is not None:
            raise RuntimeError('This instance has been connected to an experiment.')
        cmd = 'nnictl resume {}'.format(exp_id).split(' ')
        if port:
            cmd += '--port {}'.format(port).split(' ')
        if debug:
            cmd += ['--debug']
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to resume experiment.')
        else:
            if port:
                self.port = port
            else:
                self.port = 8080
            self.endpoint = 'http://localhost:{}'.format(self.port)
            self.exp_id = self.get_experiment_profile()['id']

    def view_experiment(self, exp_id, port=None):
        """
        View a stopped experiment with specified experiment id.

        Parameters
        ----------
        exp_id: str
            experiment id
        port: int
            the port of restful server, bigger than 1024
        """
        if self.endpoint is not None:
            raise RuntimeError('This instance has been connected to an experiment.')
        cmd = 'nnictl view {}'.format(exp_id).split(' ')
        if port:
            cmd += '--port {}'.format(port).split(' ')
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to view experiment.')
        else:
            if port:
                self.port = port
            else:
                self.port = 8080
            self.endpoint = 'http://localhost:{}'.format(self.port)
            self.exp_id = self.get_experiment_profile()['id']

    def connect_experiment(self, endpoint):
        """
        Connect to an existing experiment.

        Parameters
        ----------
        endpoint: str
            the endpoint of nni rest server, i.e, the url of Web UI. Should be a format like `http://ip:port`
        """
        if self.endpoint is not None:
            raise RuntimeError('This instance has been connected to an experiment.')
        self.endpoint = endpoint
        try:
            self.exp_id = self.get_experiment_profile()['id']
        except TypeError:
            raise RuntimeError('Invalid experiment endpoint.')
        self.port = int(re.search(r':[0-9]+', self.endpoint).group().replace(':', ''))   

    def stop_experiment(self):
        """Stop the experiment.
        """
        _check_endpoint(self.endpoint)
        cmd = 'nnictl stop {}'.format(self.exp_id).split(' ')
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to stop experiment.')

    def update_searchspace(self, filename):
        """
        Update the experiment's search space.

        Parameters
        ----------
        filename: str
            path to the searchspace file
        """
        _check_endpoint(self.endpoint)
        cmd = 'nnictl update searchspace {} --filename {}'.format(self.exp_id, filename).split(' ')
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to update searchspace.')

    def update_concurrency(self, value):
        """
        Update an experiment's concurrency

        Parameters
        ----------
        value: int
            new concurrency value
        """
        _check_endpoint(self.endpoint)
        cmd = 'nnictl update concurrency {} --value {}'.format(self.exp_id, value).split(' ')
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to update concurrency.')

    def update_duration(self, value):
        """
        Update an experiment's duration

        Parameters
        ----------
        value: str
            Strings like '1m' for one minute or '2h' for two hours. SUFFIX may be 's' for seconds, 'm' for minutes, 'h' for hours or 'd' for days.
        """
        _check_endpoint(self.endpoint)
        cmd = 'nnictl update duration {} --value {}'.format(self.exp_id, value).split(' ')
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to update duration.')

    def update_trailnum(self, value):
        """
        Update an experiment's maxtrialnum

        Parameters
        ----------
        value: int
            new trailnum value
        """
        _check_endpoint(self.endpoint)
        cmd = 'nnictl update trialnum {} --value {}'.format(self.exp_id, value).split(' ')
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to update trailnum.')

    def get_experiment_status(self):
        """
        Return experiment status as a dict.
        """
        _check_endpoint(self.endpoint)
        return _nni_rest_get(self.endpoint, STATUS_PATH)

    def get_trial_job(self, trial_job_id):
        """
        Return trial job information as a dict.

        Parameters
        ----------
        trial_job_id: str
            trial id
        """
        _check_endpoint(self.endpoint)
        assert trial_job_id is not None
        return _nni_rest_get(self.endpoint, os.path.join(TRIAL_JOBS_PATH, trial_job_id))

    def list_trial_jobs(self):
        """
        Return information for all trial jobs as a list.
        """
        _check_endpoint(self.endpoint)
        return _nni_rest_get(self.endpoint, TRIAL_JOBS_PATH)

    def get_job_statistics(self):
        """
        Return trial job statistics information as a dict.
        """
        _check_endpoint(self.endpoint)
        return _nni_rest_get(self.endpoint, JOB_STATISTICS_PATH)

    def get_job_metrics(self, trial_job_id=None):
        """
        Return trial job metrics.

        Parameters
        ----------
        trial_job_id: str
            trial id. if this parameter is None, all trail jobs' metrics will be returned.
        """
        _check_endpoint(self.endpoint)
        api_path = METRICS_PATH if trial_job_id is None else os.path.join(METRICS_PATH, trial_job_id)
        return _nni_rest_get(self.endpoint, api_path)

    def export_data(self):
        """
        Return exported information for all trial jobs.
        """
        _check_endpoint(self.endpoint)
        return _nni_rest_get(self.endpoint, EXPORT_DATA_PATH)

    def get_experiment_profile(self):
        """
        Return experiment profile as a dict.
        """
        _check_endpoint(self.endpoint)
        return _nni_rest_get(self.endpoint, EXPERIMENT_PATH)
