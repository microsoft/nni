# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``ConfigBase`` class. Nothing else.

Docstrings in this file are mainly for NNI contributors instead of end users.
"""

__all__ = ['ConfigBase']

import copy
import dataclasses
from pathlib import Path

import yaml

from . import utils

class ConfigBase:
    """
    The abstract base class of experiment config classes.

    A config class should be a type-hinted dataclass inheriting ``ConfigBase``.
    Or for a training service config class, it can inherit ``TrainingServiceConfig``.

    .. code-block:: python

        @dataclass(init=False)
        class ExperimentConfig(ConfigBase):
            name: Optional[str]
            ...

    Subclasses are suggested to override ``_canonicalize()`` and ``_validate_canonical()`` methods.

    Users can create a config object with constructor or ``ConfigBase.load()``,
    validate its legality with ``ConfigBase.validate()``,
    and finally convert it to the format accepted by NNI manager with ``ConfigBase.json()``.

    Example usage:

    .. code-block:: python

        # when using Python API
        config1 = ExperimentConfig(trialCommand='...', trialConcurrency=1, ...)
        config1.validate()
        print(config1.json())

        # when using config file
        config2 = ExperimentConfig.load('examples/config.yml')
        config2.validate()
        print(config2.json())

    Config objects will remember where they are loaded; therefore relative paths can be resolved smartly.
    If a config object is created with constructor, the base path will be current working directory.
    If it is loaded with ``ConfigBase.load(path)``, the base path will be ``path``'s parent.

    .. attention::

        All the classes that inherit ``ConfigBase`` are not allowed to use ``from __future__ import annotations``,
        because ``ConfigBase`` uses ``typeguard`` to perform runtime check and it does not support lazy annotations.
    """

    def __init__(self, **kwargs):
        """
        There are two common ways to use the constructor,
        directly writing kwargs and unpacking from JSON (YAML) object:

        .. code-block:: python

            config1 = AlgorithmConfig(name='TPE', class_args={'optimize_mode': 'maximize'})

            json = {'name': 'TPE', 'classArgs': {'optimize_mode': 'maximize'}}
            config2 = AlgorithmConfig(**json)

        If the config class has fields whose type is another config class, or list of another config class,
        they will recursively load dict values.

        Because JSON objects can use "camelCase" for field names,
        cases and underscores in ``kwargs`` keys are ignored in this constructor.
        For example if a config class has a field ``hello_world``,
        then using ``hello_world=1``, ``helloWorld=1``, and ``_HELLOWORLD_=1`` in constructor
        will all assign to the same field.

        If ``kwargs`` contain extra keys, `AttributeError` will be raised.

        If ``kwargs`` do not have enough key, missing fields are silently set to `MISSING()`.
        You can use ``utils.is_missing()`` to check them.
        """
        self._base_path = utils.get_base_path()
        args = {utils.case_insensitive(key): value for key, value in kwargs.items()}
        for field in utils.fields(self):
            value = args.pop(utils.case_insensitive(field.name), field.default)
            setattr(self, field.name, value)
        if args:  # maybe a key is misspelled
            class_name = type(self).__name__
            fields = ', '.join(args.keys())
            raise AttributeError(f'{class_name} does not have field(s) {fields}')

        # try to unpack nested config
        for field in utils.fields(self):
            value = getattr(self, field.name)
            if utils.is_instance(value, field.type):
                continue  # already accepted by subclass, don't touch it
            if isinstance(value, dict):
                config = utils.guess_config_type(value, field.type)
                if config is not None:
                    setattr(self, field.name, config)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                configs = utils.guess_list_config_type(value, field.type)
                if configs:
                    setattr(self, field.name, configs)

    @classmethod
    def load(cls, path):
        """
        Load a YAML config file from file system.

        Since YAML is a superset of JSON, it can also load JSON files.

        This method raises exception if:

        - The file is not available
        - The file content is not valid YAML
        - Top level value of the YAML is not object
        - The YAML contains not supported fields

        It does not raise exception when the YAML misses fields or contains bad fields.

        Parameters
        ----------
        path : PathLike
            Path of the config file.

        Returns
        -------
        cls
            An object of ConfigBase subclass.
        """
        with open(path, encoding='utf_8') as yaml_file:
            data = yaml.safe_load(yaml_file)
        if not isinstance(data, dict):
            raise TypeError(f'Conent of config file {path} is not a dict/object')
        utils.set_base_path(Path(path).parent)
        config = cls(**data)
        utils.unset_base_path()
        return config

    def canonical_copy(self):
        """
        Create a "canonical" copy of the config, and validate it.

        This function is mainly used internally by NNI.

        Term explanation:
        The config schema for end users is more flexible than the format NNI manager accepts,
        so config classes have to deal with the conversion.
        Here we call the converted format "canonical".

        Returns
        -------
        type(self)
            A deep copy.
        """
        canon = copy.deepcopy(self)
        canon._canonicalize([])
        canon._validate_canonical()
        return canon

    def validate(self):
        """
        Validate legality of the config object. Raise exception if any error occurred.

        This function does **not** return truth value. Do not write ``if config.validate()``.

        Returns
        -------
        None
        """
        self.canonical_copy()

    def json(self):
        """
        Convert the config to JSON object (not JSON string).

        In current implementation ``json()`` will invoke ``validate()``, but this might change in future version.
        It is recommended to call ``validate()`` before ``json()`` for now.

        Returns
        -------
        dict
            JSON object.
        """
        canon = self.canonical_copy()
        return dataclasses.asdict(canon, dict_factory=_dict_factory)  # this is recursive

    def _canonicalize(self, parents):
        """
        To be overrided by subclass.

        Convert the config object to canonical format.

        The default implementation will:

        1. Resolve all ``PathLike`` fields to absolute path
        2. Call ``_canonicalize([self] + parents)`` on all children config objects, including those inside list and dict

        If the subclass has nested config fields, be careful about where to call ``super()._canonicalize()``.

        Parameters
        ----------
        parents : list[ConfigBase]
            The upper level config objects.
            For example local training service's ``trialGpuNumber`` will be copied from top level when not set,
            in this case it will be invoked like ``localConfig._canonicalize([experimentConfig])``.
        """
        for field in utils.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (Path, str)) and utils.is_path_like(field.type):
                setattr(self, field.name, utils.resolve_path(value, self._base_path))
            else:
                _recursive_canonicalize_child(value, [self] + parents)

    def _validate_canonical(self):
        """
        To be overrided by subclass.

        Validate legality of a canonical config object. It's caller's responsibility to ensure the config is canonical.

        Raise exception if any problem found. This function does **not** return truth value.

        The default implementation will:

        1. Validate that all fields match their type hint
        2. Call ``_validate_canonical()`` on children config objects, including those inside list and dict
        """
        utils.validate_type(self)
        for field in utils.fields(self):
            value = getattr(self, field.name)
            _recursive_validate_child(value)

    def __setattr__(self, name, value):
        """
        To prevent typo, config classes forbid assigning to attribute that is not a config field,
        unless it starts with underscore.
        """
        if hasattr(self, name) or name.startswith('_'):
            super().__setattr__(name, value)
            return
        if name in [field.name for field in utils.fields(self)]:  # might happend during __init__
            super().__setattr__(name, value)
            return
        raise AttributeError(f'{type(self).__name__} does not have field {name}')

def _dict_factory(items):
    ret = {}
    for key, value in items:
        if value is not None:
            # NOTE (zhe):
            # It's hard for end users to set a field to missing, so I decided to treat None as "not set".
            # If a field needs explicit "None", use something like magic string.
            k = utils.camel_case(key)
            v = str(value) if isinstance(value, Path) else value
            ret[k] = v
    return ret

def _recursive_canonicalize_child(child, parents):
    if isinstance(child, ConfigBase):
        child._canonicalize(parents)
    elif isinstance(child, list):
        for item in child:
            _recursive_canonicalize_child(item, parents)
    elif isinstance(child, dict):
        for item in child.values():
            _recursive_canonicalize_child(item, parents)

def _recursive_validate_child(child):
    if isinstance(child, ConfigBase):
        child._validate_canonical()
    elif isinstance(child, list):
        for item in child:
            _recursive_validate_child(item)
    elif isinstance(child, dict):
        for item in child.values():
            _recursive_validate_child(item)
