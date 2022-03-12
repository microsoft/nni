# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utility functions for experiment config classes.
"""

import dataclasses
import math
from pathlib import Path
from typing import Union

PathLike = Union[Path, str]

def is_missing(value):
    """
    Used to check whether a dataclass field has ever been assigned.

    If a field without default value has never been assigned, it will have a special value ``MISSING``.
    This function checks if the parameter is ``MISSING``.
    """
    # MISSING is not singleton and there is no official API to check it
    return isinstance(value, type(dataclasses.MISSING))

def canonical_gpu_indices(indices):
    """
    If ``indices`` is not None, cast it to list of int.
    """
    if isinstance(indices, str):
        return [int(idx) for idx in indices.split(',')]
    if isinstance(indices, int):
        return [indices]
    return indices

def validate_gpu_indices(indices):
    if indices is None:
        return
    if len(set(indices)) != len(indices):
        raise ValueError(f'Duplication detected in GPU indices {indices}')
    if any(idx < 0 for idx in indices):
        raise ValueError(f'Negative detected in GPU indices {indices}')

def parse_time(value):
    """
    If ``value`` is a string, convert it to integral number of seconds.
    """
    return _parse_unit(value, 's', _time_units)

def parse_memory_size(value):
    """
    If ``value`` is a string, convert it to integral number of mega bytes.
    """
    return _parse_unit(value, 'mb', _size_units)

_time_units = {'d': 24 * 3600, 'h': 3600, 'm': 60, 's': 1}
_size_units = {'tb': 1024 ** 4, 'gb': 1024 ** 3, 'mb': 1024 ** 2, 'kb': 1024, 'b': 1}

def _parse_unit(value, target_unit, all_units):
    if not isinstance(value, str):
        return value
    value = value.lower()
    for unit, factor in all_units.items():
        if value.endswith(unit):
            number = value[:-len(unit)]
            value = float(number) * factor
            return math.ceil(value / all_units[target_unit])
    supported_units = ', '.join(all_units.keys())
    raise ValueError(f'Bad unit in "{value}", supported units are {supported_units}')
