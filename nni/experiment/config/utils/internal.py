# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utility functions for experiment config classes, internal part.

If you are implementing a config class for a training service, it's unlikely you will need these.
"""

import dataclasses
import importlib
import json
import os.path
from pathlib import Path

import typeguard

import nni.runtime.config

from .public import is_missing

## handle relative path ##

_current_base_path = None

def get_base_path():
    if _current_base_path is None:
        return Path()
    return _current_base_path

def set_base_path(path):
    global _current_base_path
    assert _current_base_path is None
    _current_base_path = path

def unset_base_path():
    global _current_base_path
    _current_base_path = None

def resolve_path(path, base_path):
    if path is None:
        return None
    # Path.resolve() does not work on Windows when file not exist, so use os.path instead
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(base_path, path)
    return str(os.path.realpath(path))  # it should be already str, but official doc does not specify it's type

## field name case convertion ##

def case_insensitive(key):
    return key.lower().replace('_', '')

def camel_case(key):
    words = key.strip('_').split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

## type hint ##

def is_instance(value, type_hint):
    try:
        typeguard.check_type('_', value, type_hint)
    except TypeError:
        return False
    return True

def validate_type(config):
    class_name = type(config).__name__
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        #check existense
        if is_missing(value):
            raise ValueError(f'{class_name}: {field.name} is not set')
        if not is_instance(value, field.type):
            raise ValueError(f'{class_name}: type of {field.name} ({repr(value)}) is not {field.type}')

def is_path_like(type_hint):
    # only `PathLike` and `Any` accepts `Path`; check `int` to make sure it's not `Any`
    return is_instance(Path(), type_hint) and not is_instance(1, type_hint)

def guess_config_type(obj, type_hint):
    ret = guess_list_config_type([obj], type_hint, _hint_list_item=True)
    return ret[0] if ret else None

def guess_list_config_type(objs, type_hint, _hint_list_item=False):
    # because __init__ of subclasses might be complex, we first create empty objects to determine type
    from ..base import ConfigBase  # avoid circular import
    candidate_classes = []
    for cls in _all_subclasses(ConfigBase):
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
            candidate_configs.append(configs)
        except Exception:
            pass

    if not candidate_configs:
        return None
    if len(candidate_configs) == 1:
        return candidate_configs[0]

    # still have multiple candidates, inheritance relationship among these classes should make up a tree
    # if the tree only has one leaf (e.g. TrainingServiceConfig -> LocalConfig), choose the leaf
    # otherwise, choose the common base class (the root)
    roots = []
    leaves = []
    for cls_configs in candidate_configs:
        cls = type(cls_configs[0])
        subclass_cnt = sum(isinstance(configs[0], cls) for configs in candidate_configs)
        if subclass_cnt == len(candidate_configs):
            roots.append(cls_configs)
        if subclass_cnt == 1:
            leaves.append(cls_configs)
    if len(leaves) == 1:
        return leaves[0]
    if len(roots) == 1:
        return roots[0]

    return None  # cannot detect the type, give up

def _all_subclasses(cls):
    subclasses = set(cls.__subclasses__())
    return subclasses.union(*[_all_subclasses(subclass) for subclass in subclasses])

## training service factory ##

def training_service_config_factory(platform):
    from ..training_service import TrainingServiceConfig  # avoid circular import

    # import all custom config classes so they can be found in TrainingServiceConfig.__subclasses__()
    custom_ts_config_path = nni.runtime.config.get_config_file('training_services.json')
    custom_ts_config = json.load(custom_ts_config_path.open())
    for custom_ts_pkg in custom_ts_config.keys():
        pkg = importlib.import_module(custom_ts_pkg)
        _config_class = pkg.nni_training_service_info.config_class

    for cls in TrainingServiceConfig.__subclasses__():
        if cls.platform == platform:
            return cls()
    raise ValueError(f'Bad training service platform: {platform}')
