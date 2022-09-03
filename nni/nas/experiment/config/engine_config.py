# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Optional, List

from nni.experiment.config.base import ConfigBase

__all__ = ['ExecutionEngineConfig', 'BaseEngineConfig', 'OneshotEngineConfig',
           'PyEngineConfig', 'CgoEngineConfig', 'BenchmarkEngineConfig']

@dataclass(init=False)
class ExecutionEngineConfig(ConfigBase):
    name: str

@dataclass(init=False)
class PyEngineConfig(ExecutionEngineConfig):
    name: str = 'py'

@dataclass(init=False)
class OneshotEngineConfig(ExecutionEngineConfig):
    name: str = 'oneshot'

@dataclass(init=False)
class BaseEngineConfig(ExecutionEngineConfig):
    name: str = 'base'
    # input used in GraphConverterWithShape. Currently support shape tuple only.
    dummy_input: Optional[List[int]] = None

@dataclass(init=False)
class CgoEngineConfig(ExecutionEngineConfig):
    name: str = 'cgo'
    max_concurrency_cgo: Optional[int] = None
    batch_waiting_time: Optional[int] = None
    # input used in GraphConverterWithShape. Currently support shape tuple only.
    dummy_input: Optional[List[int]] = None

@dataclass(init=False)
class BenchmarkEngineConfig(ExecutionEngineConfig):
    name: str = 'benchmark'
    benchmark: Optional[str] = None