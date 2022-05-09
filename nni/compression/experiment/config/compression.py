# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Type

from torch.nn import Module

from nni.experiment.config import ExperimentConfig
from nni.experiment.config.base import ConfigBase
from .pruner import PrunerConfig
from .quantizer import QuantizerConfig
from .common import ComparableType

__all__ = ['CompressionConfig', 'CompressionExperimentConfig']


@dataclass
class CompressionConfig(ConfigBase):
    # constraints
    params: str | int | float | None = None
    flops: str | int | float | None = None
    # latency: float | None
    metric: ComparableType | None = None

    # compress scope description
    module_types: List[Type[Module] | str] | None = None
    module_names: List[str] | None = None
    exclude_module_names: List[str] | None = None

    # pruning algorithm description
    pruners: List[PrunerConfig] | None = None
    quantizers: List[QuantizerConfig] | None = None


@dataclass(init=False)
class CompressionExperimentConfig(ExperimentConfig):

    compression_setting: CompressionConfig

    def __init__(self, training_service_platform=None, compression_setting=None, **kwargs):
        super().__init__(training_service_platform, **kwargs)
        if compression_setting:
            self.compression_setting = compression_setting
        else:
            self.compression_setting = CompressionConfig()
