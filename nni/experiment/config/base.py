# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

from . import util

__all__ = ['ConfigBase', 'PathLike']

T = TypeVar('T', bound='ConfigBase')

PathLike = util.PathLike

def _is_missing(obj: Any) -> bool:
    return isinstance(obj, type(dataclasses.MISSING))

class ConfigBase:
    """
    Base class of config classes.
    Subclass may override `_canonical_rules` and `_validation_rules`,
    and `validate()` if the logic is complex.
    """

    # Rules to convert field value to canonical format.
    # The key is field name.
    # The value is callable `value -> canonical_value`
    # It is not type-hinted so dataclass won't treat it as field
    _canonical_rules = {}  # type: ignore

    # Rules to validate field value.
    # The key is field name.
    # The value is callable `value -> valid` or `value -> (valid, error_message)`
    # The rule will be called with canonical format and is only called when `value` is not None.
    # `error_message` is used when `valid` is False.
    # It will be prepended with class name and field name in exception message.
    _validation_rules = {}  # type: ignore

    def __init__(self, *, _base_path: Optional[Path] = None, **kwargs):
        """
        Initialize a config object and set some fields.
        Name of keyword arguments can either be snake_case or camelCase.
        They will be converted to snake_case automatically.
        If a field is missing and don't have default value, it will be set to `dataclasses.MISSING`.
        """
        if 'basepath' in kwargs:
            _base_path = kwargs.pop('basepath')
        kwargs = {util.case_insensitive(key): value for key, value in kwargs.items()}
        if _base_path is None:
            _base_path = Path()
        for field in dataclasses.fields(self):
            value = kwargs.pop(util.case_insensitive(field.name), field.default)
            if value is not None and not _is_missing(value):
                # relative paths loaded from config file are not relative to pwd
                if 'Path' in str(field.type):
                    value = Path(value).expanduser()
                    if not value.is_absolute():
                        value = _base_path / value
            setattr(self, field.name, value)
        if kwargs:
            cls = type(self).__name__
            fields = ', '.join(kwargs.keys())
            raise ValueError(f'{cls}: Unrecognized fields {fields}')

    @classmethod
    def load(cls: Type[T], path: PathLike) -> T:
        """
        Load config from YAML (or JSON) file.
        Keys in YAML file can either be camelCase or snake_case.
        """
        data = yaml.safe_load(open(path))
        if not isinstance(data, dict):
            raise ValueError(f'Content of config file {path} is not a dict/object')
        return cls(**data, _base_path=Path(path).parent)

    def json(self) -> Dict[str, Any]:
        """
        Convert config to JSON object.
        The keys of returned object will be camelCase.
        """
        self.validate()
        return dataclasses.asdict(
            self.canonical(),
            dict_factory=lambda items: dict((util.camel_case(k), v) for k, v in items if v is not None)
        )

    def canonical(self: T) -> T:
        """
        Returns a deep copy, where the fields supporting multiple formats are converted to the canonical format.
        Noticeably, relative path may be converted to absolute path.
        """
        ret = copy.deepcopy(self)
        for field in dataclasses.fields(ret):
            key, value = field.name, getattr(ret, field.name)
            rule = ret._canonical_rules.get(key)
            if rule is not None:
                setattr(ret, key, rule(value))
            elif isinstance(value, ConfigBase):
                setattr(ret, key, value.canonical())
                # value will be copied twice, should not be a performance issue anyway
            elif isinstance(value, Path):
                setattr(ret, key, str(value))
        return ret

    def validate(self) -> None:
        """
        Validate the config object and raise Exception if it's ill-formed.
        """
        class_name = type(self).__name__
        config = self.canonical()

        for field in dataclasses.fields(config):
            key, value = field.name, getattr(config, field.name)

            # check existence
            if _is_missing(value):
                raise ValueError(f'{class_name}: {key} is not set')

            # check type (TODO)
            type_name = str(field.type).replace('typing.', '')
            optional = any([
                type_name.startswith('Optional['),
                type_name.startswith('Union[') and 'None' in type_name,
                type_name == 'Any'
            ])
            if value is None:
                if optional:
                    continue
                else:
                    raise ValueError(f'{class_name}: {key} cannot be None')

            # check value
            rule = config._validation_rules.get(key)
            if rule is not None:
                try:
                    result = rule(value)
                except Exception:
                    raise ValueError(f'{class_name}: {key} has bad value {repr(value)}')

                if isinstance(result, bool):
                    if not result:
                        raise ValueError(f'{class_name}: {key} ({repr(value)}) is out of range')
                else:
                    if not result[0]:
                        raise ValueError(f'{class_name}: {key} {result[1]}')

            # check nested config
            if isinstance(value, ConfigBase):
                value.validate()
