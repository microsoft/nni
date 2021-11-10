# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utility functions for experiment config classes, internal part.

If you are implementing a config class for a training service, it's unlikely you will need these.
"""

import dataclasses
from pathlib import Path

import typeguard

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
        return candidate_configs

    # still have multiple candidates, try to find the common base class
    for base in candidate_configs:
        base_class = type(base[0])
        is_base = all(isinstance(configs[0], base_class) for configs in candidate_configs)
        if is_base:
            return base

    return None  # cannot detect the type, give up

def _all_subclasses(cls):
    subclasses = set(cls.__subclasses__())
    return subclasses.union(*[_all_subclasses(subclass) for subclass in subclasses])
