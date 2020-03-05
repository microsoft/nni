# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import functools
from enum import Enum, unique
import json_tricks

from .common import init_logger
from .env_vars import dispatcher_env_vars

to_json = functools.partial(json_tricks.dumps, allow_nan=True)

@unique
class OptimizeMode(Enum):
    """Optimize Mode class

    if OptimizeMode is 'minimize', it means the tuner need to minimize the reward
    that received from Trial.

    if OptimizeMode is 'maximize', it means the tuner need to maximize the reward
    that received from Trial.
    """
    Minimize = 'minimize'
    Maximize = 'maximize'


class NodeType:
    """Node Type class
    """
    ROOT = 'root'
    TYPE = '_type'
    VALUE = '_value'
    INDEX = '_index'
    NAME = '_name'


class MetricType:
    """The types of metric data
    """
    FINAL = 'FINAL'
    PERIODICAL = 'PERIODICAL'
    REQUEST_PARAMETER = 'REQUEST_PARAMETER'


def split_index(params):
    """
    Delete index infromation from params
    """
    if isinstance(params, dict):
        if NodeType.INDEX in params.keys():
            return split_index(params[NodeType.VALUE])
        result = {}
        for key in params:
            result[key] = split_index(params[key])
        return result
    else:
        return params


def extract_scalar_reward(value, scalar_key='default'):
    """
    Extract scalar reward from trial result.

    Parameters
    ----------
    value : int, float, dict
        the reported final metric data
    scalar_key : str
        the key name that indicates the numeric number

    Raises
    ------
    RuntimeError
        Incorrect final result: the final result should be float/int,
        or a dict which has a key named "default" whose value is float/int.
    """
    if isinstance(value, (float, int)):
        reward = value
    elif isinstance(value, dict) and scalar_key in value and isinstance(value[scalar_key], (float, int)):
        reward = value[scalar_key]
    else:
        raise RuntimeError('Incorrect final result: the final result should be float/int, ' \
            'or a dict which has a key named "default" whose value is float/int.')
    return reward


def extract_scalar_history(trial_history, scalar_key='default'):
    """
    Extract scalar value from a list of intermediate results.

    Parameters
    ----------
    trial_history : list
        accumulated intermediate results of a trial
    scalar_key : str
        the key name that indicates the numeric number

    Raises
    ------
    RuntimeError
        Incorrect final result: the final result should be float/int,
        or a dict which has a key named "default" whose value is float/int.
    """
    return [extract_scalar_reward(ele, scalar_key) for ele in trial_history]


def convert_dict2tuple(value):
    """
    convert dict type to tuple to solve unhashable problem.
    """
    if isinstance(value, dict):
        for _keys in value:
            value[_keys] = convert_dict2tuple(value[_keys])
        return tuple(sorted(value.items()))
    return value


def init_dispatcher_logger():
    """
    Initialize dispatcher logging configuration
    """
    logger_file_path = 'dispatcher.log'
    if dispatcher_env_vars.NNI_LOG_DIRECTORY is not None:
        logger_file_path = os.path.join(dispatcher_env_vars.NNI_LOG_DIRECTORY, logger_file_path)
    init_logger(logger_file_path, dispatcher_env_vars.NNI_LOG_LEVEL)
