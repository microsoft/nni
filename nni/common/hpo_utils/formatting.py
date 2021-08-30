# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script provides a more program-friendly representation of HPO search space.
The format is considered internal helper and is not visible to end users.

You will find this useful when you want to support nested search space.
"""

__all__ = [
    'ParameterSpec',
    'deformat_parameters',
    'format_search_space',
]

import math
from typing import Any, List, NamedTuple, Optional, Tuple

class ParameterSpec(NamedTuple):
    """
    Specification (aka space / range) of one single parameter.
    """

    name: str                       # The object key in JSON
    type: str                       # "_type" in JSON
    values: List[Any]               # "_value" in JSON

    key: Tuple[str]                 # The "path" of this parameter
    parent_index: Optional[int]     # If the parameter is in a nested choice, this is its parent's index;
                                    # if the parameter is at top level, this is `None`.

    categorical: bool               # Whether this paramter is categorical (unordered) or numerical (ordered)
    size: int = None                # If it's categorical, how many canidiates it has

    # uniform distributed
    low: float = None               # Lower bound of uniform parameter
    high: float = None              # Upper bound of uniform parameter

    normal_distributed: bool = None # Whether this parameter is uniform or normal distrubuted
    mu: float = None                # Mean of normal parameter
    sigma: float = None             # Scale of normal parameter

    q: Optional[float] = None       # If not `None`, the value should be an integer multiple of this
    log_distributed: bool = None    # Whether this parameter is log distributed

    def is_activated(self, partial_parameters):
        """
        For nested search space, check whether this parameter should be skipped for current set of paremters.
        This function works because the return value of `format_search_space()` is sorted in a way that
        parents always appear before children.
        """
        return self.parent_index is None or partial_parameters.get(self.key[:-1]) == self.parent_index

def format_search_space(search_space, ordered_randint=False):
    formatted = _format_search_space(tuple(), None, search_space)
    if ordered_randint:
        for i, spec in enumerate(formatted):
            if spec.type == 'randint':
                formatted[i] = _format_ordered_randint(spec.key, spec.parent_index, spec.values)
    return {spec.key: spec for spec in formatted}

def deformat_parameters(parameters, formatted_search_space):
    """
    `paramters` is a dict whose key is `ParamterSpec.key`, and value is integer index if the parameter is categorical.
    Convert it to the format expected by end users.
    """
    ret = {}
    for key, x in parameters.items():
        spec = formatted_search_space[key]
        if not spec.categorical:
            _assign(ret, key, x)
        elif spec.type == 'randint':
            lower = min(math.ceil(float(x)) for x in spec.values)
            _assign(ret, key, lower + x)
        elif _is_nested_choices(spec.values):
            _assign(ret, tuple([*key, '_name']), spec.values[x]['_name'])
        else:
            _assign(ret, key, spec.values[x])
    return ret

def _format_search_space(parent_key, parent_index, space):
    formatted = []
    for name, spec in space.items():
        if name == '_name':
            continue
        key = tuple([*parent_key, name])
        formatted.append(_format_parameter(key, parent_index, spec['_type'], spec['_value']))
        if spec['_type'] == 'choice' and _is_nested_choices(spec['_value']):
            for index, sub_space in enumerate(spec['_value']):
                formatted += _format_search_space(key, index, sub_space)
    return formatted

def _format_parameter(key, parent_index, type_, values):
    spec = {}
    spec['name'] = key[-1]
    spec['type'] = type_
    spec['values'] = values

    spec['key'] = key
    spec['parent_index'] = parent_index

    if type_ in ['choice', 'randint']:
        spec['categorical'] = True
        if type_ == 'choice':
            spec['size'] = len(values)
        else:
            lower, upper = sorted(math.ceil(float(x)) for x in values)
            spec['size'] = upper - lower

    else:
        spec['categorical'] = False
        if type_.startswith('q'):
            spec['q'] = float(values[2])
        spec['log_distributed'] = ('log' in type_)

        if 'normal' in type_:
            spec['normal_distributed'] = True
            spec['mu'] = float(values[0])
            spec['sigma'] = float(values[1])

        else:
            spec['normal_distributed'] = False
            spec['low'], spec['high'] = sorted(float(x) for x in values[:2])
            if 'q' in spec:
                spec['low'] = math.ceil(spec['low'] / spec['q']) * spec['q']
                spec['high'] = math.floor(spec['high'] / spec['q']) * spec['q']

    return ParameterSpec(**spec)

def _format_ordered_randint(key, parent_index, values):
    lower, upper = sorted(math.ceil(float(x)) for x in values)
    return ParameterSpec(
        name = key[-1],
        type = 'randint',
        values = values,
        key = key,
        parent_index = parent_index,
        categorical = False,
        low = float(lower),
        high = float(upper - 1),
        normal_distributed = False,
        q = 1.0,
        log_distributed = False,
    )

def _is_nested_choices(values):
    if not values:
        return False
    for value in values:
        if not isinstance(value, dict):
            return False
        if '_name' not in value:
            return False
    return True

def _assign(params, key, x):
    if len(key) == 1:
        params[key[0]] = x
    else:
        if key[0] not in params:
            params[key[0]] = {}
        _assign(params[key[0]], key[1:], x)
