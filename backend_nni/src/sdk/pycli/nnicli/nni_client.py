# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

""" A python wrapper for nni rest api

Example:

from nnicli import Experiment

exp = Experiment()
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
import json
import requests

__all__ = [
    'Experiment',
    'TrialResult',
    'TrialMetricData',
    'TrialHyperParameters',
    'TrialJob'
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
            raise RuntimeError('Incorrect response_type')
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
        raise RuntimeError("This instance hasn't been connect to an experiment.")

class TrialResult:
    """
    TrialResult stores the result information of a trial job.

    Parameters
    ----------
    json_obj: dict
        Json object that stores the result information.

    Attributes
    ----------
    parameter: dict
        Hyper parameters for this trial.
    value: serializable object, usually a number, or a dict with key "default" and other extra keys
        Final result.
    trialJobId: str
        Trial job id.
    """
    def __init__(self, json_obj):
        self.parameter = None
        self.value = None
        self.trialJobId = None
        for key in json_obj.keys():
            if key == 'id':
                setattr(self, 'trialJobId', json_obj[key])
            elif hasattr(self, key):
                setattr(self, key, json_obj[key])
        self.value = json.loads(self.value)

    def __repr__(self):
        return "TrialResult(parameter: {} value: {} trialJobId: {})".format(self.parameter, self.value, self.trialJobId)

class TrialMetricData:
    """
    TrialMetricData stores the metric data of a trial job.
    A trial job may have both intermediate metric and final metric.

    Parameters
    ----------
    json_obj: dict
        Json object that stores the metric data.

    Attributes
    ----------
    timestamp: int
        Time stamp.
    trialJobId: str
        Trial job id.
    parameterId: int
        Parameter id.
    type: str
        Metric type, `PERIODICAL` for intermediate result and `FINAL` for final result.
    sequence: int
        Sequence number in this trial.
    data: serializable object, usually a number, or a dict with key "default" and other extra keys
        Metric data.
    """
    def __init__(self, json_obj):
        self.timestamp = None
        self.trialJobId = None
        self.parameterId = None
        self.type = None
        self.sequence = None
        self.data = None
        for key in json_obj.keys():
            setattr(self, key, json_obj[key])
        self.data = json.loads(json.loads(self.data))

    def __repr__(self):
        return "TrialMetricData(timestamp: {} trialJobId: {} parameterId: {} type: {} sequence: {} data: {})" \
            .format(self.timestamp, self.trialJobId, self.parameterId, self.type, self.sequence, self.data)

class TrialHyperParameters:
    """
    TrialHyperParameters stores the hyper parameters of a trial job.

    Parameters
    ----------
    json_obj: dict
        Json object that stores the hyper parameters.

    Attributes
    ----------
    parameter_id: int
        Parameter id.
    parameter_source: str
        Parameter source.
    parameters: dict
        Hyper parameters.
    parameter_index: int
        Parameter index.
    """
    def __init__(self, json_obj):
        self.parameter_id = None
        self.parameter_source = None
        self.parameters = None
        self.parameter_index = None
        for key in json_obj.keys():
            if hasattr(self, key):
                setattr(self, key, json_obj[key])

    def __repr__(self):
        return "TrialHyperParameters(parameter_id: {} parameter_source: {} parameters: {} parameter_index: {})" \
            .format(self.parameter_id, self.parameter_source, self.parameters, self.parameter_index)

class TrialJob:
    """
    TrialJob stores the information of a trial job.

    Parameters
    ----------
    json_obj: dict
        json object that stores the hyper parameters

    Attributes
    ----------
    trialJobId: str
        Trial job id.
    status: str
        Job status.
    hyperParameters: list of `nnicli.TrialHyperParameters`
        See `nnicli.TrialHyperParameters`.
    logPath: str
        Log path.
    startTime: int
        Job start time (timestamp).
    endTime: int
        Job end time (timestamp).
    finalMetricData: list of `nnicli.TrialMetricData`
        See `nnicli.TrialMetricData`.
    parameter_index: int
        Parameter index.
    """
    def __init__(self, json_obj):
        self.trialJobId = None
        self.status = None
        self.hyperParameters = None
        self.logPath = None
        self.startTime = None
        self.endTime = None
        self.finalMetricData = None
        self.stderrPath = None
        for key in json_obj.keys():
            if key == 'id':
                setattr(self, 'trialJobId', json_obj[key])
            elif hasattr(self, key):
                setattr(self, key, json_obj[key])
        if self.hyperParameters:
            self.hyperParameters = [TrialHyperParameters(json.loads(e)) for e in self.hyperParameters]
        if self.finalMetricData:
            self.finalMetricData = [TrialMetricData(e) for e in self.finalMetricData]

    def __repr__(self):
        return ("TrialJob(trialJobId: {} status: {} hyperParameters: {} logPath: {} startTime: {} "
                "endTime: {} finalMetricData: {} stderrPath: {})") \
                    .format(self.trialJobId, self.status, self.hyperParameters, self.logPath,
                            self.startTime, self.endTime, self.finalMetricData, self.stderrPath)

class Experiment:
    def __init__(self):
        self._endpoint = None
        self._exp_id = None
        self._port = None

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def exp_id(self):
        return self._exp_id

    @property
    def port(self):
        return self._port

    def _exec_command(self, cmd, port=None):
        if self._endpoint is not None:
            raise RuntimeError('This instance has been connected to an experiment.')
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to establish experiment, please check your config.')
        else:
            if port:
                self._port = port
            else:
                self._port = 8080
            self._endpoint = 'http://localhost:{}'.format(self._port)
            self._exp_id = self.get_experiment_profile()['id']

    def start_experiment(self, config_file, port=None, debug=False):
        """
        Start an experiment with specified configuration file and connect to it.

        Parameters
        ----------
        config_file: str
            Path to the config file.
        port: int
            The port of restful server, bigger than 1024.
        debug: boolean
            Set debug mode.
        """
        cmd = 'nnictl create --config {}'.format(config_file).split(' ')
        if port:
            cmd += '--port {}'.format(port).split(' ')
        if debug:
            cmd += ['--debug']
        self._exec_command(cmd, port)

    def resume_experiment(self, exp_id, port=None, debug=False):
        """
        Resume a stopped experiment with specified experiment id

        Parameters
        ----------
        exp_id: str
            Experiment id.
        port: int
            The port of restful server, bigger than 1024.
        debug: boolean
            Set debug mode.
        """
        cmd = 'nnictl resume {}'.format(exp_id).split(' ')
        if port:
            cmd += '--port {}'.format(port).split(' ')
        if debug:
            cmd += ['--debug']
        self._exec_command(cmd, port)

    def view_experiment(self, exp_id, port=None):
        """
        View a stopped experiment with specified experiment id.

        Parameters
        ----------
        exp_id: str
            Experiment id.
        port: int
            The port of restful server, bigger than 1024.
        """
        cmd = 'nnictl view {}'.format(exp_id).split(' ')
        if port:
            cmd += '--port {}'.format(port).split(' ')
        self._exec_command(cmd, port)

    def connect_experiment(self, endpoint):
        """
        Connect to an existing experiment.

        Parameters
        ----------
        endpoint: str
            The endpoint of nni rest server, i.e, the url of Web UI. Should be a format like `http://ip:port`.
        """
        if self._endpoint is not None:
            raise RuntimeError('This instance has been connected to an experiment.')
        self._endpoint = endpoint
        try:
            self._exp_id = self.get_experiment_profile()['id']
        except TypeError:
            raise RuntimeError('Invalid experiment endpoint.')
        self._port = int(re.search(r':[0-9]+', self._endpoint).group().replace(':', ''))

    def stop_experiment(self):
        """Stop the experiment.
        """
        _check_endpoint(self._endpoint)
        cmd = 'nnictl stop {}'.format(self._exp_id).split(' ')
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to stop experiment.')
        self._endpoint = None
        self._exp_id = None
        self._port = None

    def update_searchspace(self, filename):
        """
        Update the experiment's search space.

        Parameters
        ----------
        filename: str
            Path to the searchspace file.
        """
        _check_endpoint(self._endpoint)
        cmd = 'nnictl update searchspace {} --filename {}'.format(self._exp_id, filename).split(' ')
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to update searchspace.')

    def update_concurrency(self, value):
        """
        Update an experiment's concurrency

        Parameters
        ----------
        value: int
            New concurrency value.
        """
        _check_endpoint(self._endpoint)
        cmd = 'nnictl update concurrency {} --value {}'.format(self._exp_id, value).split(' ')
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to update concurrency.')

    def update_duration(self, value):
        """
        Update an experiment's duration

        Parameters
        ----------
        value: str
            Strings like '1m' for one minute or '2h' for two hours.
            SUFFIX may be 's' for seconds, 'm' for minutes, 'h' for hours or 'd' for days.
        """
        _check_endpoint(self._endpoint)
        cmd = 'nnictl update duration {} --value {}'.format(self._exp_id, value).split(' ')
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to update duration.')

    def update_trailnum(self, value):
        """
        Update an experiment's maxtrialnum

        Parameters
        ----------
        value: int
            New trailnum value.
        """
        _check_endpoint(self._endpoint)
        cmd = 'nnictl update trialnum {} --value {}'.format(self._exp_id, value).split(' ')
        if _create_process(cmd) != 0:
            raise RuntimeError('Failed to update trailnum.')

    def get_experiment_status(self):
        """
        Return experiment status as a dict.

        Returns
        ----------
        dict
            Experiment status.
        """
        _check_endpoint(self._endpoint)
        return _nni_rest_get(self._endpoint, STATUS_PATH)

    def get_trial_job(self, trial_job_id):
        """
        Return a trial job.

        Parameters
        ----------
        trial_job_id: str
            Trial job id.

        Returns
        ----------
        nnicli.TrialJob
            A `nnicli.TrialJob` instance corresponding to `trial_job_id`.
        """
        _check_endpoint(self._endpoint)
        assert trial_job_id is not None
        trial_job = _nni_rest_get(self._endpoint, os.path.join(TRIAL_JOBS_PATH, trial_job_id))
        return TrialJob(trial_job)

    def list_trial_jobs(self):
        """
        Return information for all trial jobs as a list.

        Returns
        ----------
        list
            List of `nnicli.TrialJob`.
        """
        _check_endpoint(self._endpoint)
        trial_jobs = _nni_rest_get(self._endpoint, TRIAL_JOBS_PATH)
        return [TrialJob(e) for e in trial_jobs]

    def get_job_statistics(self):
        """
        Return trial job statistics information as a dict.

        Returns
        ----------
        list
            Job statistics information.
        """
        _check_endpoint(self._endpoint)
        return _nni_rest_get(self._endpoint, JOB_STATISTICS_PATH)

    def get_job_metrics(self, trial_job_id=None):
        """
        Return trial job metrics.

        Parameters
        ----------
        trial_job_id: str
            trial job id. if this parameter is None, all trail jobs' metrics will be returned.

        Returns
        ----------
        dict
            Each key is a trialJobId, the corresponding value is a list of `nnicli.TrialMetricData`.
        """
        _check_endpoint(self._endpoint)
        api_path = METRICS_PATH if trial_job_id is None else os.path.join(METRICS_PATH, trial_job_id)
        output = {}
        trail_metrics = _nni_rest_get(self._endpoint, api_path)
        for metric in trail_metrics:
            trial_id = metric["trialJobId"]
            if trial_id not in output:
                output[trial_id] = [TrialMetricData(metric)]
            else:
                output[trial_id].append(TrialMetricData(metric))
        return output

    def export_data(self):
        """
        Return exported information for all trial jobs.

        Returns
        ----------
        list
            List of `nnicli.TrialResult`.
        """
        _check_endpoint(self._endpoint)
        trial_results = _nni_rest_get(self._endpoint, EXPORT_DATA_PATH)
        return [TrialResult(e) for e in trial_results]

    def get_experiment_profile(self):
        """
        Return experiment profile as a dict.

        Returns
        ----------
        dict
            The profile of the experiment.
        """
        _check_endpoint(self._endpoint)
        return _nni_rest_get(self._endpoint, EXPERIMENT_PATH)
