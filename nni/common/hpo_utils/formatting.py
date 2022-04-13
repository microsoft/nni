# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Helper class and functions for tuners to deal with search space.

This script provides a more program-friendly representation of HPO search space.
The format is considered internal helper and is not visible to end users.

You will find this useful when you want to support nested search space.

The random tuner is an intuitive example for this utility.
You should check its code before reading docstrings in this file.

.. attention::

    This module does not guarantee forward-compatibility.

    If you want to use it outside official NNI repo, it is recommended to copy the script.
"""

from __future__ import annotations

__all__ = [
    'ParameterSpec',
    'deformat_parameters',
    'format_parameters',
    'format_search_space',
]

import math
from types import SimpleNamespace
from typing import Any, Dict, NamedTuple, Tuple, cast

import numpy as np

from nni.typehint import Parameters, SearchSpace

ParameterKey = Tuple['str | int', ...]
FormattedParameters = Dict[ParameterKey, 'float | int']
FormattedSearchSpace = Dict[ParameterKey, 'ParameterSpec']

class ParameterSpec(NamedTuple):
    """
    Specification (aka space / range / domain) of one single parameter.

    NOTE: For `loguniform` (and `qloguniform`), the fields `low` and `high` are logarithm of original values.
    """

    name: str                       # The object key in JSON
    type: str                       # "_type" in JSON
    values: list[Any]               # "_value" in JSON

    key: ParameterKey               # The "path" of this parameter

    categorical: bool               # Whether this paramter is categorical (unordered) or numerical (ordered)
    size: int = cast(int, None)     # If it's categorical, how many candidates it has
    chosen_size: int | None = 1     # If it's categorical, it should choose how many candidates.
                                    # By default, 1. If none, arbitrary number of candidates can be chosen.

    # uniform distributed
    low: float = cast(float, None)  # Lower bound of uniform parameter
    high: float = cast(float, None) # Upper bound of uniform parameter

    normal_distributed: bool = cast(bool, None)
                                    # Whether this parameter is uniform or normal distrubuted
    mu: float = cast(float, None)   # µ of normal parameter
    sigma: float = cast(float, None)# σ of normal parameter

    q: float | None = None          # If not `None`, the parameter value should be an integer multiple of this
    clip: tuple[float, float] | None = None
                                    # For q(log)uniform, this equals to "values[:2]"; for others this is None

    log_distributed: bool = cast(bool, None)
                                    # Whether this parameter is log distributed
                                    # When true, low/high/mu/sigma describes log of parameter value (like np.lognormal)

    def is_activated_in(self, partial_parameters: FormattedParameters) -> bool:
        """
        For nested search space, check whether this parameter should be skipped for current set of paremters.
        This function must be used in a pattern similar to random tuner. Otherwise it will misbehave.
        """
        if self.is_nested():
            return partial_parameters[self.key[:-2]] == self.key[-2]
        else:
            return True

    def is_nested(self):
        """
        Check whether this parameter is inside a nested choice.
        """
        return len(self.key) >= 2 and isinstance(self.key[-2], int)

def format_search_space(search_space: SearchSpace) -> FormattedSearchSpace:
    """
    Convert user provided search space into a dict of ParameterSpec.
    The dict key is dict value's `ParameterSpec.key`.
    """
    formatted = _format_search_space(tuple(), search_space)
    return {spec.key: spec for spec in formatted}

def deformat_parameters(
        formatted_parameters: FormattedParameters,
        formatted_search_space: FormattedSearchSpace) -> Parameters:
    """
    Convert internal format parameters to users' expected format.

    "test/ut/sdk/test_hpo_formatting.py" provides examples of how this works.

    The function do following jobs:
     1. For "choice" and "randint", convert index (integer) to corresponding value.
     2. For "*log*", convert x to `exp(x)`.
     3. For "q*", convert x to `round(x / q) * q`, then clip into range.
     4. For nested choices, convert flatten key-value pairs into nested structure.
    """
    ret: Parameters = {}
    for key, x in formatted_parameters.items():
        spec = formatted_search_space[key]
        if spec.categorical:
            x = cast(int, x)
            if spec.type == 'randint':
                lower = min(math.ceil(float(x)) for x in spec.values)
                _assign(ret, key, int(lower + x))
            elif _is_nested_choices(spec.values):
                _assign(ret, tuple([*key, '_name']), spec.values[x]['_name'])
            else:
                _assign(ret, key, spec.values[x])
        else:
            if spec.log_distributed:
                x = math.exp(x)
            if spec.q is not None:
                x = round(x / spec.q) * spec.q
            if spec.clip:
                x = max(x, spec.clip[0])
                x = min(x, spec.clip[1])
            if isinstance(x, np.number):
                x = x.item()
            _assign(ret, key, x)
    return ret

def format_parameters(parameters: Parameters, formatted_search_space: FormattedSearchSpace) -> FormattedParameters:
    """
    Convert end users' parameter format back to internal format, mainly for resuming experiments.

    The result is not accurate for "q*" and for "choice" that have duplicate candidates.
    """
    # I don't like this function. It's better to use checkpoint for resuming.
    ret = {}
    for key, spec in formatted_search_space.items():
        if not spec.is_activated_in(ret):
            continue
        value: Any = parameters
        for name in key:
            if isinstance(name, str):
                value = value[name]
        if spec.categorical:
            if spec.type == 'randint':
                lower = min(math.ceil(float(x)) for x in spec.values)
                ret[key] = value - lower
            elif _is_nested_choices(spec.values):
                names = [nested['_name'] for nested in spec.values]
                ret[key] = names.index(value['_name'])
            else:
                ret[key] = spec.values.index(value)
        else:
            if spec.log_distributed:
                value = math.log(value)
            ret[key] = value
    return ret

def _format_search_space(parent_key: ParameterKey, space: SearchSpace) -> list[ParameterSpec]:
    formatted: list[ParameterSpec] = []
    for name, spec in space.items():
        if name == '_name':
            continue
        key = tuple([*parent_key, name])
        formatted.append(_format_parameter(key, spec['_type'], spec['_value']))
        if spec['_type'] == 'choice' and _is_nested_choices(spec['_value']):
            for index, sub_space in enumerate(spec['_value']):
                key = tuple([*parent_key, name, index])
                formatted += _format_search_space(key, sub_space)
    return formatted

def _format_parameter(key: ParameterKey, type_: str, values: list[Any]):
    spec = SimpleNamespace(
        name = key[-1],
        type = type_,
        values = values,
        key = key,
        categorical = type_ in ['choice', 'randint'],
    )

    if spec.categorical:
        if type_ == 'choice':
            spec.size = len(values)
        else:
            lower = math.ceil(float(values[0]))
            upper = math.ceil(float(values[1]))
            spec.size = upper - lower

    else:
        if type_.startswith('q'):
            spec.q = float(values[2])
        else:
            spec.q = None
        spec.log_distributed = ('log' in type_)

        if 'normal' in type_:
            spec.normal_distributed = True
            spec.mu = float(values[0])
            spec.sigma = float(values[1])

        else:
            spec.normal_distributed = False
            spec.low = float(values[0])
            spec.high = float(values[1])
            if spec.q is not None:
                spec.clip = (spec.low, spec.high)
            if spec.log_distributed:
                # make it align with mu
                spec.low = math.log(spec.low)
                spec.high = math.log(spec.high)

    return ParameterSpec(**spec.__dict__)

def _is_nested_choices(values: list[Any]) -> bool:
    assert values  # choices should not be empty
    for value in values:
        if not isinstance(value, dict):
            return False
        if '_name' not in value:
            return False
    return True

def _assign(params: Parameters, key: ParameterKey, x: Any) -> None:
    if len(key) == 1:
        params[cast(str, key[0])] = x
    elif isinstance(key[0], int):
        _assign(params, key[1:], x)
    else:
        if key[0] not in params:
            params[key[0]] = {}
        _assign(params[key[0]], key[1:], x)
