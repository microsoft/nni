# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Union

from torch.nn import Module

from nni.experiment.config.base import ConfigBase
from .pruner import PrunerConfig
from .quantizer import QuantizerConfig
from .common import ComparableType

__all__ = ['CompressionConfig']


@dataclass
class CompressionConfig(ConfigBase):
    # constraints
    params: Union[str, int, float, List[Union[str, int, float]], None] = None
    flops: Union[str, int, float, List[Union[str, int, float]], None] = None
    # latency: float, List[float], None
    metric: Union[ComparableType, List[ComparableType], None] = None

    # compress scope description
    module_types: Optional[List[Union[Type[Module], Dict[Type[Module], Dict], str, Dict[str, Dict]]]] = None
    module_names: Union[List[str], Dict[str, Dict], None] = None
    exclude_module_names: Union[List[str], Dict[str, Dict], None] = None

    # pruning algorithm description
    pruners: Optional[List[PrunerConfig]] = None
    quantizers: Optional[List[QuantizerConfig]] = None
