# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = [
    'ExecutionEngineConfig', 'TrainingServiceEngineConfig', 'CgoEngineConfig', 'SequentialEngineConfig',
]

import logging
from dataclasses import dataclass
from typing import Optional

from nni.experiment.config import ExperimentConfig
from nni.experiment.config.training_services import RemoteConfig
from nni.experiment.config.utils import parse_time

from .utils import NamedSubclassConfigBase

_logger = logging.getLogger(__name__)


@dataclass(init=False)
class ExecutionEngineConfig(NamedSubclassConfigBase):
    """Base class for execution engine config. Useful for instance check."""


@dataclass(init=False)
class TrainingServiceEngineConfig(ExecutionEngineConfig):
    """Engine used together with NNI training service.

    Training service specific configs should go here,
    but they are now in top-level experiment config for historical reasons.
    """
    name: str = 'ts'


@dataclass(init=False)
class SequentialEngineConfig(ExecutionEngineConfig):
    """Engine that executes the models sequentially."""
    name: str = 'sequential'
    continue_on_failure: bool = False
    max_model_count: Optional[int] = None
    max_duration: Optional[float] = None

    def _canonicalize(self, parents):
        assert len(parents) > 0
        parent_config = parents[0]
        assert isinstance(parent_config, ExperimentConfig), 'SequentialEngineConfig must be a child of ExperimentConfig'
        if self.max_model_count is None:
            self.max_model_count = parent_config.max_trial_number
        if self.max_duration is None and parent_config.max_trial_duration is not None:
            self.max_duration = parse_time(parent_config.max_trial_duration)
        if isinstance(parent_config.trial_concurrency, int) and parent_config.trial_concurrency > 1:
            _logger.warning('Sequential engine does not support trial concurrency > 1')
        return super()._canonicalize(parents)


@dataclass(init=False)
class CgoEngineConfig(ExecutionEngineConfig):
    """Engine for cross-graph optimization."""
    name: str = 'cgo'
    max_concurrency_cgo: int
    batch_waiting_time: int

    training_service: Optional[RemoteConfig] = None

    def _canonicalize(self, parents):
        """Copy the training service config from the parent experiment config."""
        assert len(parents) > 0
        parent_config = parents[0]
        assert isinstance(parent_config, ExperimentConfig), 'CgoEngineConfig must be a child of ExperimentConfig'
        if not isinstance(parent_config.training_service, RemoteConfig):
            raise TypeError("CGO execution engine currently only supports remote training service")
        self.training_service = parent_config.training_service
        return super()._canonicalize(parents)
