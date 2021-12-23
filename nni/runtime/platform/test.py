# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: skip-file

import copy
from nni.common import load


_params = None
_last_metric = None


def get_next_parameter():
    return _params


def get_experiment_id():
    return 'fakeidex'


def get_trial_id():
    return 'fakeidtr'


def get_sequence_id():
    return 0


def send_metric(string):
    global _last_metric
    _last_metric = string


def init_params(params):
    global _params
    _params = copy.deepcopy(params)


def get_last_metric():
    metrics = load(_last_metric)
    metrics['value'] = load(metrics['value'])

    return metrics
