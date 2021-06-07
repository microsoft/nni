# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket
import psutil

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

def format_url_path(path):
    return API_ROOT_URL if path is None else f'/{path}{API_ROOT_URL}'

def set_prefix_url(prefix_path):
    global API_ROOT_URL
    API_ROOT_URL = format_url_path(prefix_path)

def metric_data_url(port):
    '''get metric_data url'''
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, METRIC_DATA_API)

def check_status_url(port):
    '''get check_status url'''
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, CHECK_STATUS_API)


def cluster_metadata_url(port):
    '''get cluster_metadata_url'''
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, CLUSTER_METADATA_API)


def import_data_url(port):
    '''get import_data_url'''
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, IMPORT_DATA_API)


def experiment_url(port):
    '''get experiment_url'''
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, EXPERIMENT_API)


def trial_jobs_url(port):
    '''get trial_jobs url'''
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, TRIAL_JOBS_API)


def trial_job_id_url(port, job_id):
    '''get trial_jobs with id url'''
    return '{0}:{1}{2}{3}/{4}'.format(BASE_URL, port, API_ROOT_URL, TRIAL_JOBS_API, job_id)


def export_data_url(port):
    '''get export_data url'''
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, EXPORT_DATA_API)


def tensorboard_url(port):
    '''get tensorboard url'''
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, TENSORBOARD_API)


def get_local_urls(port,prefix):
    '''get urls of local machine'''
    url_list = []
    for _, info in psutil.net_if_addrs().items():
        for addr in info:
            if socket.AddressFamily.AF_INET == addr.family:
                url_list.append('http://{0}:{1}{2}'.format(addr.address, port, prefix))
    return url_list
