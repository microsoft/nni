# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``TrainingServiceConfig`` class.

Docstrings in this file are mainly for NNI contributors, or training service authors.
"""

__all__ = ['TrainingServiceConfig']

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import ConfigBase
from .utils import PathLike, is_missing

@dataclass(init=False)
class TrainingServiceConfig(ConfigBase):
    """
    The base class of training service config classes.

    See ``LocalConfig`` for example usage.
    """

    platform: str
    trial_command: str
    trial_code_directory: PathLike
    trial_gpu_number: Optional[int]
    nni_manager_ip: Optional[str]
    debug: bool

    def _canonicalize(self, parents):
        """
        Besides from ``ConfigBase._canonicalize()``, this overloaded version will also
        copy training service specific fields from ``ExperimentConfig``.
        """
        shortcuts = [  # fields that can set in root level config as shortcut
            'trial_command',
            'trial_code_directory',
            'trial_gpu_number',
            'nni_manager_ip',
            'debug',
        ]
        for field_name in shortcuts:
            if is_missing(getattr(self, field_name)):
                value = getattr(parents[0], field_name)
                setattr(self, field_name, value)
        super()._canonicalize(parents)

    def _validate_canonical(self):
        super()._validate_canonical()
        cls = type(self)
        assert self.platform == cls.platform
        if not Path(self.trial_code_directory).is_dir():
            raise ValueError(f'{cls.__name__}: trial_code_directory "{self.trial_code_directory}" is not a directory')
        assert self.trial_gpu_number is None or self.trial_gpu_number >= 0
