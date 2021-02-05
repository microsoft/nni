# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: skip-file

import copy
import json_tricks


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
    metrics = json_tricks.loads(_last_metric)
    metrics['value'] = json_tricks.loads(metrics['value'])

    return metrics
