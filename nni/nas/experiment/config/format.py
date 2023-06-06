# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Optional, List

from .utils import NamedSubclassConfigBase


__all__ = [
    'ModelFormatConfig', 'GraphModelFormatConfig',
    'SimplifiedModelFormatConfig', 'RawModelFormatConfig',
]


@dataclass(init=False)
class ModelFormatConfig(NamedSubclassConfigBase):
    """Base class for model format config. Useful for instance check."""


@dataclass(init=False)
class GraphModelFormatConfig(ModelFormatConfig):
    """Model format config for graph-based model space."""
    name: str = 'graph'
    # input used in GraphConverterWithShape. Currently support shape tuple only.
    dummy_input: Optional[List[int]] = None


@dataclass(init=False)
class SimplifiedModelFormatConfig(ModelFormatConfig):
    """Model format that simplifies the model space to a dict of labeled mutables."""
    name: str = 'simplified'


@dataclass(init=False)
class RawModelFormatConfig(ModelFormatConfig):
    """Model format that keeps the original model space."""
    name: str = 'raw'
