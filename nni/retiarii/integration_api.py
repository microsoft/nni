# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import NewType, Any

import nni

from .serializer import json_loads

# NOTE: this is only for passing flake8, we cannot import RetiariiAdvisor
# because it would induce cycled import
RetiariiAdvisor = NewType('RetiariiAdvisor', Any)

_advisor: 'RetiariiAdvisor' = None


def get_advisor() -> 'RetiariiAdvisor':
    global _advisor
    assert _advisor is not None
    return _advisor


def register_advisor(advisor: 'RetiariiAdvisor'):
    global _advisor
    assert _advisor is None
    _advisor = advisor


def send_trial(parameters: dict, placement_constraint=None) -> int:
    """
    Send a new trial. Executed on tuner end.
    Return a ID that is the unique identifier for this trial.
    """
    return get_advisor().send_trial(parameters, placement_constraint)

def report_search_space(search_space: dict) -> None:
    """
    Execution engine uses this API to report the search space extracted by
    strategy to nnimanager which stores the search space into db,
    so that webui can access the search space like HPO
    """
    return get_advisor().report_search_space(search_space)

def receive_trial_parameters() -> dict:
    """
    Received a new trial. Executed on trial end.
    Reload with our json loads because NNI didn't use Retiarii serializer to load the data.
    """
    params = nni.get_next_parameter()
    params = json_loads(json.dumps(params))
    return params


def get_experiment_id() -> str:
    return nni.get_experiment_id()
