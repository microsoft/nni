# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Config classes for tuner/assessor/advisor algorithms.

Use ``AlgorithmConfig`` to specify a built-in algorithm;
use ``CustomAlgorithmConfig`` to specify a custom algorithm.

Check the reference_ for explaination of each field.

You may also want to check `tuner's overview`_.

.. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html

.. _tuner's overview: https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html

"""

__all__ = ['AlgorithmConfig', 'CustomAlgorithmConfig']

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ConfigBase
from .utils import PathLike

@dataclass(init=False)
class _AlgorithmConfig(ConfigBase):
    """
    Common base class for ``AlgorithmConfig`` and ``CustomAlgorithmConfig``.

    It's a "union set" of 2 derived classes. So users can use it as either one.
    """

    name: Optional[str] = None
    class_name: Optional[str] = None
    code_directory: Optional[PathLike] = None
    class_args: Optional[Dict[str, Any]] = None

    def _validate_canonical(self):
        super()._validate_canonical()
        if self.class_name is None:  # assume it's built-in algorithm by default
            assert self.name
            assert self.code_directory is None
        else:  # custom algorithm
            assert self.name is None
            assert self.class_name
            if not Path(self.code_directory).is_dir():
                raise ValueError(f'CustomAlgorithmConfig: code_directory "{self.code_directory}" is not a directory')

@dataclass(init=False)
class AlgorithmConfig(_AlgorithmConfig):
    """
    Configuration for built-in algorithm.
    """
    name: str
    class_args: Optional[Dict[str, Any]] = None

@dataclass(init=False)
class CustomAlgorithmConfig(_AlgorithmConfig):
    """
    Configuration for custom algorithm.
    """
    class_name: str
    code_directory: Optional[PathLike] = '.'
    class_args: Optional[Dict[str, Any]] = None
