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

    A config class should be a type-hinted dataclass inheriting ``ConfigBase``:

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
    """

    def __init__(self, **kwargs):
        """
        There are two common ways to use the constructor,
        directly writing Python code and unpacking from JSON(YAML) object:

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

        If ``kwargs`` contain extra keys, a `ValueError` will be raised.

        If ``kwargs`` do not have enough key, missing fields are silently set to `MISSING()`.
        You can use ``utils.is_missing()`` to check them.
        """
        self._base_path = utils.get_base_path()
        args = {utils.case_insensitive(key): value for key, value in kwargs.items()}
        for field in dataclasses.fields(self):
            value = args.pop(utils.case_insensitive(field.name), field.default)
            setattr(self, field.name, value)
        if args:  # maybe a key is misspelled
            class_name = type(self).__name__
            fields = ', '.join(args.keys())
            raise ValueError(f'{class_name} does not have field(s) {fields}')

        # try to unpack nested config
        for field in dataclasses.fields(self):
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
        data = yaml.safe_load(open(path))
        if not isinstance(data, dict):
            raise ValueError(f'Conent of config file {path} is not a dict/object')
        utils.set_base_path(Path(path).parent)
        config = cls(**data)
        utils.unset_base_path()
        return config

    def validate(self):
        """
        Validate legality of the config object. Raise exception if any error occurred.

        This function does **not** return truth value. Do not write ``if config.validate()``.

        Returns
        -------
        None
        """
        canon = copy.deepcopy(self)
        canon._canonicalize([])
        canon._validate_canonical()

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
        canon = copy.deepcopy(self)
        canon._canonicalize([])
        canon._validate_canonical()
        return dataclasses.asdict(canon, dict_factory=_dict_factory)  # this is recursive

    def _canonicalize(self, parents):
        """
        The config schema for end users is more flexible than the format NNI manager accepts.
        This method convert a config object to the constrained format accepted by NNI manager.

        The default implementation will:

        1. Resolve all ``PathLike`` fields to absolute path
        2. Call ``_canonicalize()`` on all children config objects, including those inside list and dict

        Subclasses are recommended to call ``super()._canonicalize(parents)`` at the end of their overrided version.

        Parameters
        ----------
        parents : list[ConfigBase]
            The upper level config objects.
            For example local training service's ``trialGpuNumber`` will be copied from top level when not set,
            in this case it will be invoked like ``localConfig._canonicalize([experimentConfig])``.
        """
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (Path, str)) and utils.is_path_like(field.type):
                setattr(self, field.name, utils.resolve_path(value, self._base_path))
            else:
                _recursive_canonicalize_child(value, [self] + parents)

    def _validate_canonical(self):
        """
        Validate legality of a canonical config object. It's caller's responsibility to ensure the config is canonical.

        Raise ``ValueError`` if any problem found. This function does **not** return truth value.

        The default implementation will:

        1. Validate that all fields match their type hint
        2. Call ``_validate_canonical()`` on children config objects, including those inside list and dict

        Subclasses are recommended to to call ``super()._validate_canonical()``.
        """
        utils.validate_type(self)
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            _recursive_validate_child(value)

def _dict_factory(items):
    ret = {}
    for key, value in items:
        if value is not None:
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
