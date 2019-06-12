# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================
"""
utils.py
"""

import os
from enum import Enum, unique

from .common import init_logger
from .env_vars import dispatcher_env_vars

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
    if isinstance(value, float) or isinstance(value, int):
        reward = value
    elif isinstance(value, dict) and scalar_key in value and isinstance(value[scalar_key], (float, int)):
        reward = value[scalar_key]
    else:
        raise RuntimeError('Incorrect final result: the final result should be float/int, or a dict which has a key named "default" whose value is float/int.')
    return reward


def convert_dict2tuple(value):
    """
    convert dict type to tuple to solve unhashable problem.
    """
    if isinstance(value, dict):
        for _keys in value:
            value[_keys] = convert_dict2tuple(value[_keys])
        return tuple(sorted(value.items()))
    else:
        return value


def init_dispatcher_logger():
    """ Initialize dispatcher logging configuration"""
    logger_file_path = 'dispatcher.log'
    if dispatcher_env_vars.NNI_LOG_DIRECTORY is not None:
        logger_file_path = os.path.join(dispatcher_env_vars.NNI_LOG_DIRECTORY, logger_file_path)
    init_logger(logger_file_path, dispatcher_env_vars.NNI_LOG_LEVEL)


def randint_to_quniform(in_x):
    if isinstance(in_x, dict):
        if NodeType.TYPE in in_x.keys():
            if in_x[NodeType.TYPE] == 'randint':
                value = in_x[NodeType.VALUE]
                value.append(1)

                in_x[NodeType.TYPE] = 'quniform'
                in_x[NodeType.VALUE] = value
 
            elif in_x[NodeType.TYPE] == 'choice':
                randint_to_quniform(in_x[NodeType.VALUE])
        else:
            for key in in_x.keys():
                randint_to_quniform(in_x[key])
    elif isinstance(in_x, list):
        for temp in in_x:
            randint_to_quniform(temp)
