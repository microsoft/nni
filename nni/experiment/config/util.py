# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Miscellaneous utility functions.
"""

import math
import os.path
from pathlib import Path
from typing import Optional, Union

PathLike = Union[Path, str]

def case_insensitive(key: str) -> str:
    return key.lower().replace('_', '')

def camel_case(key: str) -> str:
    words = key.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

def canonical_path(path: Optional[PathLike]) -> Optional[str]:
    # Path.resolve() does not work on Windows when file not exist, so use os.path instead
    return os.path.abspath(os.path.expanduser(path)) if path is not None else None

def count(*values) -> int:
    return sum(value is not None and value is not False for value in values)

def training_service_config_factory(platform: str, **kwargs): # -> TrainingServiceConfig
    from .common import TrainingServiceConfig
    for cls in TrainingServiceConfig.__subclasses__():
        if cls.platform == platform:
            return cls(**kwargs)
    raise ValueError(f'Unrecognized platform {platform}')

def strip_optional(type_hint):
    return type_hint.__args__[0] if str(type_hint).startswith('typing.Optional[') else type_hint

def parse_time(time: str, target_unit: str = 's') -> int:
    return _parse_unit(time.lower(), target_unit, _time_units)

def parse_size(size: str, target_unit: str = 'mb') -> int:
    return _parse_unit(size.lower(), target_unit, _size_units)

_time_units = {'d': 24 * 3600, 'h': 3600, 'm': 60, 's': 1}
_size_units = {'gb': 1024 * 1024 * 1024, 'mb': 1024 * 1024, 'kb': 1024}

def _parse_unit(string, target_unit, all_units):
    for unit, factor in all_units.items():
        if string.endswith(unit):
            number = string[:-len(unit)]
            value = float(number) * factor
            return math.ceil(value / all_units[target_unit])
    raise ValueError(f'Unsupported unit in "{string}"')
