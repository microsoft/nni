# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = [
    'ParameterSpec',
    'deformat_parameters',
    'format_search_space',
]

import math
from typing import Any, List, NamedTuple, Optional, Tuple

class ParameterSpec(NamedTuple):
    name: str
    type: str
    values: List[Any]

    key: Tuple[str]
    parent_index: Optional[int]

    categorical: bool
    size: int = None
    nested_choice: bool = False

    # uniform distributed
    low: float = None
    high: float = None

    normal_distributed: bool = None
    mu: float = None
    sigma: float = None

    q: Optional[float] = None
    log_distributed: bool = None

    def is_activated(self, partial_parameters):
        return self.parent_index is None or partial_parameters.get(self.key[:-1]) == self.parent_index

def format_search_space(search_space, ordered_randint=False):
    formatted = _format_search_space(tuple(), None, search_space)
    if ordered_randint:
        for i, spec in enumerate(formatted):
            if spec.type == 'randint':
                formatted[i] = _format_ordered_randint(spec.key, spec.parent_index, spec.values)
    return {spec.key: spec for spec in formatted}

def deformat_parameters(parameters, formatted_search_space):
    ret = {}
    for key, x in parameters.items():
        spec = formatted_search_space[key]
        if spec.nested_choice:
            _assign(ret, tuple([*key, '_name']), spec.values[x])
        elif spec.categorical:
            if spec.type == 'choice':
                _assign(ret, key, spec.values[x])
            else:
                lower = min(math.ceil(float(x)) for x in spec.values)
                _assign(ret, key, lower + x)
        else:
            _assign(ret, key, x)
    return ret

def _format_search_space(parent_key, parent_index, space):
    formatted = []
    for name, spec in space.items():
        if name == '_name':
            continue
        key = tuple([*parent_key, name])
        if spec['_type'] == 'choice' and _is_nested_choice(spec['_value']):
            sub_space_names = [sub_space['_name'] for sub_space in spec['_value']]
            formatted.append(_format_parameter(key, parent_index, 'choice', sub_space_names, True))
            for index, sub_space in enumerate(spec['_value']):
                formatted += _format_search_space(key, index, sub_space)
        else:
            formatted.append(_format_parameter(key, parent_index, spec['_type'], spec['_value']))
    return formatted

def _format_parameter(key, parent_index, type_, values, nested_choice=False):
    spec = {}
    spec['name'] = key[-1]
    spec['type'] = type_
    spec['values'] = values

    spec['key'] = key
    spec['parent_index'] = parent_index
    spec['nested_choice'] = nested_choice

    if type_ in ['choice', 'randint']:
        spec['categorical'] = True
        if type_ == 'choice':
            spec['size'] = len(values)
        else:
            lower, upper = sorted(math.ceil(float(x)) for x in values)
            spec['size'] = upper - lower

    else:
        spec['categorical'] = False
        if 'q' in type_:
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

def _is_nested_choice(values):
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
