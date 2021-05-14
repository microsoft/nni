# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket
import psutil
import re

BASE_URL = 'http://localhost'

API_ROOT_URL = '/api/v1/nni'

EXPERIMENT_API = '/experiment'

CLUSTER_METADATA_API = '/experiment/cluster-metadata'

IMPORT_DATA_API = '/experiment/import-data'

CHECK_STATUS_API = '/check-status'

TRIAL_JOBS_API = '/trial-jobs'

EXPORT_DATA_API = '/export-data'

TENSORBOARD_API = '/tensorboard'

METRIC_DATA_API = '/metric-data'

def path_validation(path):
    assert re.match("^[A-Za-z0-9_-]*$", path), "prefix url is invalid."

def formatURLPath(path):
    return '' if path is None else '/{0}'.format(path)

def metric_data_url(port,prefix):
    '''get metric_data url'''
    return '{0}:{1}{2}{3}{4}'.format(BASE_URL, port, API_ROOT_URL, formatURLPath(prefix), METRIC_DATA_API)

def check_status_url(port,prefix):
    '''get check_status url'''
    return '{0}:{1}{2}{3}{4}'.format(BASE_URL, port, API_ROOT_URL, formatURLPath(prefix), CHECK_STATUS_API)


def cluster_metadata_url(port,prefix):
    '''get cluster_metadata_url'''
    return '{0}:{1}{2}{3}{4}'.format(BASE_URL, port, API_ROOT_URL, formatURLPath(prefix), CLUSTER_METADATA_API)


def import_data_url(port,prefix):
    '''get import_data_url'''
    return '{0}:{1}{2}{3}{4}'.format(BASE_URL, port, API_ROOT_URL, formatURLPath(prefix), IMPORT_DATA_API)


def experiment_url(port,prefix):
    '''get experiment_url'''
    return '{0}:{1}{2}{3}{4}'.format(BASE_URL, port, API_ROOT_URL, formatURLPath(prefix), EXPERIMENT_API)


def trial_jobs_url(port,prefix):
    '''get trial_jobs url'''
    return '{0}:{1}{2}{3}{4}'.format(BASE_URL, port, API_ROOT_URL, formatURLPath(prefix), TRIAL_JOBS_API)


def trial_job_id_url(port, job_id,prefix):
    '''get trial_jobs with id url'''
    return '{0}:{1}{2}{3}{4}/{5}'.format(BASE_URL, port, API_ROOT_URL, formatURLPath(prefix), TRIAL_JOBS_API, job_id)


def export_data_url(port,prefix):
    '''get export_data url'''
    return '{0}:{1}{2}{3}{4}'.format(BASE_URL, port, API_ROOT_URL, formatURLPath(prefix), EXPORT_DATA_API)


def tensorboard_url(port,prefix):
    '''get tensorboard url'''
    return '{0}:{1}{2}{3}{4}'.format(BASE_URL, port, API_ROOT_URL, formatURLPath(prefix), TENSORBOARD_API)


def get_local_urls(port,prefix):
    '''get urls of local machine'''
    url_list = []
    for _, info in psutil.net_if_addrs().items():
        for addr in info:
            if socket.AddressFamily.AF_INET == addr.family:
                url_list.append('http://{0}:{1}{2}'.format(addr.address, port, formatURLPath(prefix)))
    return url_list
