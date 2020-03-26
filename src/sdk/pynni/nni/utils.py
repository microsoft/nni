# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import functools
from enum import Enum, unique
import json_tricks

from .common import init_logger
from .env_vars import dispatcher_env_vars

import nni.parameter_expressions as parameter_expressions


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
    """ Initialize dispatcher logging configuration"""
    logger_file_path = 'dispatcher.log'
    if dispatcher_env_vars.NNI_LOG_DIRECTORY is not None:
        logger_file_path = os.path.join(dispatcher_env_vars.NNI_LOG_DIRECTORY, logger_file_path)
    init_logger(logger_file_path, dispatcher_env_vars.NNI_LOG_LEVEL)
    

def json2space(x, oldy=None, name=NodeType.ROOT):
    """
    Change search space from json format to hyperopt format

    """
    y = list()
    if isinstance(x, dict):
        if NodeType.TYPE in x.keys():
            _type = x[NodeType.TYPE]
            name = name + '-' + _type
            if _type == 'choice':
                if oldy is not None:
                    _index = oldy[NodeType.INDEX]
                    y += json2space(x[NodeType.VALUE][_index],
                                    oldy[NodeType.VALUE], name=name+'[%d]' % _index)
                else:
                    y += json2space(x[NodeType.VALUE], None, name=name)
            y.append(name)
        else:
            for key in x.keys():
                y += json2space(x[key], oldy[key] if oldy else None, name+"[%s]" % str(key))
    elif isinstance(x, list):
        for i, x_i in enumerate(x):
            if isinstance(x_i, dict):
                if NodeType.NAME not in x_i.keys():
                    raise RuntimeError('\'_name\' key is not found in this nested search space.')
            y += json2space(x_i, oldy[i] if oldy else None, name + "[%d]" % i)
    return y


def json2parameter(x, is_rand, random_state, oldy=None, Rand=False, name=NodeType.ROOT):
    """
    Json to pramaters.

    """
    if isinstance(x, dict):
        if NodeType.TYPE in x.keys():
            _type = x[NodeType.TYPE]
            _value = x[NodeType.VALUE]
            name = name + '-' + _type
            Rand |= is_rand[name]
            if Rand is True:
                if _type == 'choice':
                    _index = random_state.randint(len(_value))
                    y = {
                        NodeType.INDEX: _index,
                        NodeType.VALUE: json2parameter(
                            x[NodeType.VALUE][_index],
                            is_rand,
                            random_state,
                            None,
                            Rand,
                            name=name+"[%d]" % _index
                        )
                    }
                else:
                    y = getattr(parameter_expressions, _type)(*(_value + [random_state]))
            else:
                y = copy.deepcopy(oldy)
        else:
            y = dict()
            for key in x.keys():
                y[key] = json2parameter(
                    x[key],
                    is_rand,
                    random_state,
                    oldy[key] if oldy else None,
                    Rand,
                    name + "[%s]" % str(key)
                )
    elif isinstance(x, list):
        y = list()
        for i, x_i in enumerate(x):
            if isinstance(x_i, dict):
                if NodeType.NAME not in x_i.keys():
                    raise RuntimeError('\'_name\' key is not found in this nested search space.')
            y.append(json2parameter(
                x_i,
                is_rand,
                random_state,
                oldy[i] if oldy else None,
                Rand,
                name + "[%d]" % i
            ))
    else:
        y = copy.deepcopy(x)
    return y
