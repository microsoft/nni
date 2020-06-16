# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .constants import API_ROOT_URL, BASE_URL, STDOUT_API, NNI_TRIAL_JOB_ID, NNI_EXP_ID, VERSION_API, PARAMETER_META_API


def gen_send_stdout_url(ip, port, trial_id):
    '''Generate send stdout url'''
    if trial_id is None:
        trial_id = NNI_TRIAL_JOB_ID
    return '{0}:{1}{2}{3}/{4}/{5}'.format(BASE_URL.format(ip), port, API_ROOT_URL, STDOUT_API, NNI_EXP_ID, trial_id)


def gen_send_version_url(ip, port, trial_id):
    '''Generate send error url'''
    if trial_id is None:
        trial_id = NNI_TRIAL_JOB_ID
    return '{0}:{1}{2}{3}/{4}/{5}'.format(BASE_URL.format(ip), port, API_ROOT_URL, VERSION_API, NNI_EXP_ID, trial_id)


def gen_parameter_meta_url(ip, port):
    '''Generate send error url'''
    return '{0}:{1}{2}{3}'.format(BASE_URL.format(ip), port, API_ROOT_URL, PARAMETER_META_API)
