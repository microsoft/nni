# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utility functions for experiment config classes, internal part.

If you are implementing a config class for a training service, it's unlikely you will need these.
"""

from __future__ import annotations

__all__ = [
    'get_base_path', 'set_base_path', 'unset_base_path', 'resolve_path',
    'case_insensitive', 'camel_case',
    'fields', 'is_instance', 'validate_type', 'is_path_like',
    'guess_config_type', 'guess_list_config_type',
    'training_service_config_factory', 'load_training_service_config',
    'get_ipv4_address', 'diff'
]

import copy
import dataclasses
import importlib
import json
import os.path
from pathlib import Path
import socket
from typing import TYPE_CHECKING, get_type_hints

import typeguard

import nni.runtime.config

from .public import is_missing

if TYPE_CHECKING:
    from nni.nas.experiment.pytorch import RetiariiExperiment
    from nni.nas.experiment.config import NasExperimentConfig
    from ...experiment import Experiment
    from ..base import ConfigBase
    from ..experiment_config import ExperimentConfig
    from ..training_service import TrainingServiceConfig

## handle relative path ##

_current_base_path: Path | None = None

def get_base_path() -> Path:
    if _current_base_path is None:
        return Path()
    return _current_base_path

def set_base_path(path: Path) -> None:
    global _current_base_path
    assert _current_base_path is None
    _current_base_path = path

def unset_base_path() -> None:
    global _current_base_path
    _current_base_path = None

def resolve_path(path: Path | str, base_path: Path) -> str:
    assert path is not None
    # Path.resolve() does not work on Windows when file not exist, so use os.path instead
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(base_path, path)
    return str(os.path.realpath(path))  # it should be already str, but official doc does not specify it's type

## field name case convertion ##

def case_insensitive(key: str) -> str:
    return key.lower().replace('_', '')

def camel_case(key: str) -> str:
    words = key.strip('_').split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

## type hint utils ##

def fields(config: ConfigBase) -> list[dataclasses.Field]:
    # Similar to `dataclasses.fields()`, but use `typing.get_types_hints()` to get `field.type`.
    # This is useful when postponed evaluation is enabled.
    ret = [copy.copy(field) for field in dataclasses.fields(config)]
    types = get_type_hints(type(config))
    for field in ret:
        field.type = types[field.name]
    return ret

def is_instance(value, type_hint) -> bool:
    try:
        typeguard.check_type(value, type_hint)
    except typeguard.TypeCheckError:
        return False
    return True

def validate_type(config: ConfigBase) -> None:
    class_name = type(config).__name__
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        #check existense
        if is_missing(value):
            raise ValueError(f'{class_name}: {field.name} is not set')
        if not is_instance(value, field.type):
            raise ValueError(f'{class_name}: type of {field.name} ({repr(value)}) is not {field.type}')

def is_path_like(type_hint) -> bool:
    # only `PathLike` and `Any` accepts `Path`; check `int` to make sure it's not `Any`
    return is_instance(Path(), type_hint) and not is_instance(1, type_hint)

## type inference ##

def guess_config_type(obj, type_hint) -> ConfigBase | None:
    ret = guess_list_config_type([obj], type_hint, _hint_list_item=True)
    return ret[0] if ret else None

def guess_list_config_type(objs, type_hint, _hint_list_item=False) -> list[ConfigBase] | None:
    # avoid circular import
    from ..base import ConfigBase
    from ..training_service import TrainingServiceConfig

    # because __init__ of subclasses might be complex, we first create empty objects to determine type
    candidate_classes = []
    for cls in _all_subclasses(ConfigBase):
        if issubclass(cls, TrainingServiceConfig):  # training service configs are specially handled
            continue
        empty_list = [cls.__new__(cls)]
        if _hint_list_item:
            good_type = is_instance(empty_list[0], type_hint)
        else:
            good_type = is_instance(empty_list, type_hint)
        if good_type:
            candidate_classes.append(cls)

    if not candidate_classes:  # it does not accept config type
        return None
    if len(candidate_classes) == 1:  # the type is confirmed, raise error if cannot convert to this type
        return [candidate_classes[0](**obj) for obj in objs]

    # multiple candidates available, call __init__ to further verify
    candidate_configs = []
    for cls in candidate_classes:
        try:
            configs = [cls(**obj) for obj in objs]
        except Exception:
            # FIXME: The reason why the guess failed is eaten here. We should at least print one of them.
            continue
        candidate_configs.append(configs)

    if not candidate_configs:
        return None
    if len(candidate_configs) == 1:
        return candidate_configs[0]

    # still have multiple candidates, choose the common base class
    for base in candidate_configs:
        base_class = type(base[0])
        is_base = all(isinstance(configs[0], base_class) for configs in candidate_configs)
        if is_base:
            return base

    return None  # cannot detect the type, give up

def _all_subclasses(cls):
    subclasses = set(cls.__subclasses__())
    return subclasses.union(*[_all_subclasses(subclass) for subclass in subclasses])

def training_service_config_factory(platform: str) -> TrainingServiceConfig:
    cls = _get_ts_config_class(platform)
    if cls is None:
        raise ValueError(f'Bad training service platform: {platform}')
    return cls()

def load_training_service_config(config) -> TrainingServiceConfig:
    if isinstance(config, dict) and 'platform' in config:
        cls = _get_ts_config_class(config['platform'])
        if cls is not None:
            return cls(**config)
    # not valid json, don't touch
    return config  # type: ignore

def _get_ts_config_class(platform: str) -> type[TrainingServiceConfig] | None:
    from ..training_service import TrainingServiceConfig  # avoid circular import

    # import all custom config classes so they can be found in TrainingServiceConfig.__subclasses__()
    custom_ts_config_path = nni.runtime.config.get_config_file('training_services.json')
    with custom_ts_config_path.open() as config_file:
        custom_ts_config = json.load(config_file)
    for custom_ts_pkg in custom_ts_config.keys():
        pkg = importlib.import_module(custom_ts_pkg)
        _config_class = pkg.nni_training_service_info.config_class

    for cls in _all_subclasses(TrainingServiceConfig):
        if cls.platform == platform:
            return cls
    return None

## misc ##

def get_ipv4_address() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('192.0.2.0', 80))
    addr = s.getsockname()[0]
    s.close()
    return addr

def diff(config1: ConfigBase, config2: ConfigBase, from_: str = '', to_: str = '') -> str:
    # This method is not to get an accurate diff, but to give users a rough idea of what is changed.
    import difflib
    import pprint

    # The ideal diff should directly apply on the pprint of dataclass. However,
    #  1. pprint of dataclass is not stable. It actually changes from python 3.9 to 3.10.
    #  2. MISSING doesn't have a stable memory address. It might be different in different objects.

    str1, str2 = pprint.pformat(config1.json()), pprint.pformat(config2.json())

    return '\n'.join(difflib.unified_diff(str1.splitlines(), str2.splitlines(), from_, to_, lineterm=''))
