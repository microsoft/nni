# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Miscellaneous utility functions.
"""

import importlib
import json
import math
import os.path
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import nni.runtime.config

PathLike = Union[Path, str]

def case_insensitive(key_or_kwargs: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
    if isinstance(key_or_kwargs, str):
        return key_or_kwargs.lower().replace('_', '')
    else:
        return {key.lower().replace('_', ''): value for key, value in key_or_kwargs.items()}

def camel_case(key: str) -> str:
    words = key.strip('_').split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

def canonical_path(path: Optional[PathLike]) -> Optional[str]:
    # Path.resolve() does not work on Windows when file not exist, so use os.path instead
    return os.path.abspath(os.path.expanduser(path)) if path is not None else None

def count(*values) -> int:
    return sum(value is not None and value is not False for value in values)

def training_service_config_factory(
        platform: Union[str, List[str]] = None,
        config: Union[List, Dict] = None,
        base_path: Optional[Path] = None): # -> TrainingServiceConfig
    from .common import TrainingServiceConfig

    # import all custom config classes so they can be found in TrainingServiceConfig.__subclasses__()
    custom_ts_config_path = nni.runtime.config.get_config_file('training_services.json')
    custom_ts_config = json.load(custom_ts_config_path.open())
    for custom_ts_pkg in custom_ts_config.keys():
        pkg = importlib.import_module(custom_ts_pkg)
        _config_class = pkg.nni_training_service_info.config_class

    ts_configs = []
    if platform is not None:
        assert config is None
        platforms = platform if isinstance(platform, list) else [platform]
        for cls in TrainingServiceConfig.__subclasses__():
            if cls.platform in platforms:
                ts_configs.append(cls())
        if len(ts_configs) < len(platforms):
            bad = ', '.join(set(platforms) - set(ts_configs))
            raise RuntimeError(f'Bad training service platform: {bad}')
    else:
        assert config is not None
        supported_platforms = {cls.platform: cls for cls in TrainingServiceConfig.__subclasses__()}
        configs = config if isinstance(config, list) else [config]
        for conf in configs:
            if conf['platform'] not in supported_platforms:
                raise RuntimeError(f'Unrecognized platform {conf["platform"]}')
            ts_configs.append(supported_platforms[conf['platform']](_base_path=base_path, **conf))
    return ts_configs if len(ts_configs) > 1 else ts_configs[0]

def load_config(Type, value):
    if isinstance(value, list):
        return [load_config(Type, item) for item in value]
    if isinstance(value, dict):
        return Type(**value)
    return value

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

def canonical_gpu_indices(indices: Union[List[int], str, int, None]) -> Optional[List[int]]:
    if isinstance(indices, str):
        return [int(idx) for idx in indices.split(',')]
    if isinstance(indices, int):
        return [indices]
    return indices
