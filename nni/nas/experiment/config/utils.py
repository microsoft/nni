# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['NamedSubclassConfigBase']

from typing import Type

from nni.experiment.config.base import ConfigBase


class NamedSubclassConfigBase(ConfigBase):
    """Base class for configs with ``name`` to specify the type."""

    # NOTE: This should've been ClassVar, but ClassVar won't appear in dataclass fields,
    #       which is used by `json()`.
    name: str
    """Different names correspond to different config types."""

    def __new__(cls, name: str | None = None, **kwargs):
        if name is not None:
            type_ = cls.config_class_from_name(name)
            return type_(**kwargs)
        return super().__new__(cls)

    def __init__(self, name: str | None = None, **kwargs):
        if name is not None:
            # `__init__` would be called with the name as an argument, regardless of whether it's passed to `__new__`.
            assert name == self.__class__.name
        super().__init__(**kwargs)

    def json(self) -> dict:
        return {
            'name': self.name,
            **super().json()
        }

    @classmethod
    def config_class_from_name(cls: Type[NamedSubclassConfigBase], name: str) -> Type[NamedSubclassConfigBase]:
        valid_names = []
        for subcls in cls.__subclasses__():
            valid_names.append(subcls.name)
            if subcls.name == name:
                return subcls
        raise ValueError(f'Invalid {cls.__name__} subclass: {name}. Valid subclass names are: {valid_names}')
