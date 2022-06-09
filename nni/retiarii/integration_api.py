# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import NewType, Any

import nni
from nni.common.version import version_check

# NOTE: this is only for passing flake8, we cannot import RetiariiAdvisor
# because it would induce cycled import
RetiariiAdvisor = NewType('RetiariiAdvisor', Any)

_advisor = None  # type is RetiariiAdvisor


def get_advisor():
    # return type: RetiariiAdvisor
    global _advisor
    assert _advisor is not None
    return _advisor


def register_advisor(advisor):
    # type of advisor: RetiariiAdvisor
    global _advisor
    if _advisor is not None:
        warnings.warn('Advisor is already set.'
                      'You should avoid instantiating RetiariiExperiment twice in one proces.'
                      'If you are running in a Jupyter notebook, please restart the kernel.')
    _advisor = advisor


def send_trial(parameters: dict, placement_constraint=None) -> int:
    """
    Send a new trial. Executed on tuner end.
    Return a ID that is the unique identifier for this trial.
    """
    return get_advisor().send_trial(parameters, placement_constraint)

def receive_trial_parameters() -> dict:
    """
    Received a new trial. Executed on trial end.
    Reload with our json loads because NNI didn't use Retiarii serializer to load the data.
    """
    params = nni.get_next_parameter()

    # version check, optional
    raw_params = nni.trial._params
    if raw_params is not None and 'version_info' in raw_params:
        version_check(raw_params['version_info'])
    else:
        warnings.warn('Version check failed because `version_info` is not found.')

    return params


def get_experiment_id() -> str:
    return nni.get_experiment_id()
