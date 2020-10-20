# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import functools
from enum import Enum, unique
import json_tricks
from schema import And

from . import parameter_expressions
from .runtime.common import init_logger
from .runtime.env_vars import dispatcher_env_vars


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

def merge_parameter(base_params, override_params):
    """
    Update the parameters in ``base_params`` with ``override_params``.
    Can be useful to override parsed command line arguments.

    Parameters
    ----------
    base_params : namespace or dict
        Base parameters. A key-value mapping.
    override_params : dict or None
        Parameters to override. Usually the parameters got from ``get_next_parameters()``.
        When it is none, nothing will happen.

    Returns
    -------
    namespace or dict
        The updated ``base_params``. Note that ``base_params`` will be updated inplace. The return value is
        only for convenience.
    """
    if override_params is None:
        return base_params
    is_dict = isinstance(base_params, dict)
    for k, v in override_params.items():
        if is_dict:
            if k not in base_params:
                raise ValueError('Key \'%s\' not found in base parameters.' % k)
            if type(base_params[k]) != type(v) and base_params[k] is not None:
                raise TypeError('Expected \'%s\' in override parameters to have type \'%s\', but found \'%s\'.' %
                                (k, type(base_params[k]), type(v)))
            base_params[k] = v
        else:
            if not hasattr(base_params, k):
                raise ValueError('Key \'%s\' not found in base parameters.' % k)
            if type(getattr(base_params, k)) != type(v) and getattr(base_params, k) is not None:
                raise TypeError('Expected \'%s\' in override parameters to have type \'%s\', but found \'%s\'.' %
                                (k, type(getattr(base_params, k)), type(v)))
            setattr(base_params, k, v)
    return base_params

class ClassArgsValidator(object):
    """
    NNI tuners/assessors/adivisors accept a `classArgs` parameter in experiment configuration file.
    This ClassArgsValidator interface is used to validate the classArgs section in exeperiment
    configuration file.
    """
    def validate_class_args(self, **kwargs):
        """
        Validate the classArgs configuration in experiment configuration file.

        Parameters
        ----------
        kwargs: dict
            kwargs passed to tuner/assessor/advisor constructor

        Raises:
            Raise an execption if the kwargs is invalid.
        """
        pass

    def choices(self, key, *args):
        """
        Utility method to create a scheme to check whether the `key` is one of the `args`.

        Parameters:
        ----------
        key: str
            key name of the data to be validated
        args: list of str
            list of the choices

        Returns: Schema
        --------
            A scheme to check whether the `key` is one of the `args`.
        """
        return And(lambda n: n in args, error='%s should be in [%s]!' % (key, str(args)))

    def range(self, key, keyType, start, end):
        """
        Utility method to create a schema to check whether the `key` is in the range of [start, end].

        Parameters:
        ----------
        key: str
            key name of the data to be validated
        keyType: type
            python data type, such as int, float
        start: type is specified by keyType
            start of the range
        end: type is specified by keyType
            end of the range

        Returns: Schema
        --------
            A scheme to check whether the `key` is in the range of [start, end].
        """
        return And(
            And(keyType, error='%s should be %s type!' % (key, keyType.__name__)),
            And(lambda n: start <= n <= end, error='%s should be in range of (%s, %s)!' % (key, start, end))
        )
