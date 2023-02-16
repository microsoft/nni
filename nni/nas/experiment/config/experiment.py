# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union, Optional, TYPE_CHECKING
from typing_extensions import Literal

from nni.experiment.config import utils, ExperimentConfig

from .engine import ExecutionEngineConfig, SequentialEngineConfig
from .format import ModelFormatConfig

if TYPE_CHECKING:
    from nni.nas.evaluator import Evaluator
    from nni.nas.nn.pytorch import ModelSpace
    from nni.nas.strategy import Strategy


__all__ = ['NasExperimentConfig']

RESERVED = '_reserved_'

_logger = logging.getLogger(__name__)


@dataclass(init=False)
class NasExperimentConfig(ExperimentConfig):
    # TODO: refactor this class to inherit from a new common base class with HPO config
    experiment_type: Literal['nas'] = 'nas'
    search_space: Any = RESERVED
    trial_code_directory: utils.PathLike = '.'
    trial_command: str = ''

    # New config field for NAS
    execution_engine: ExecutionEngineConfig
    model_format: ModelFormatConfig

    # Internal: to support customized fields in trial command
    # Useful when customized python / environment variables are needed
    _trial_command_params: Optional[Dict[str, Any]] = None

    def __init__(self,
                 execution_engine: str | ExecutionEngineConfig | None = None,
                 model_format: str | ModelFormatConfig | None = None,
                 training_service_platform: str | list[str] | None = None,
                 **kwargs):
        # `execution_engine` and `model_format` are two shortcuts for easy configuration.
        # We merge them into `kwargs` and let the parent class handle them.
        if isinstance(execution_engine, str):
            kwargs.update(execution_engine=ExecutionEngineConfig(name=execution_engine))
        elif isinstance(execution_engine, ExecutionEngineConfig):
            kwargs.update(execution_engine=execution_engine)
        elif execution_engine is not None:
            raise TypeError('execution_engine must be a string or an ExecutionEngineConfig object.')

        if isinstance(model_format, str):
            kwargs.update(model_format=ModelFormatConfig(name=model_format))
        elif isinstance(model_format, ModelFormatConfig):
            kwargs.update(model_format=model_format)
        elif model_format is not None:
            raise TypeError('model_format must be a string or a ModelFormatConfig object.')

        super().__init__(training_service_platform=training_service_platform, **kwargs)

    @classmethod
    def default(cls, model_space: ModelSpace, evaluator: Evaluator, strategy: Strategy) -> NasExperimentConfig:
        _logger.info('Config is not provided. Will try to infer.')

        trial_concurrency = 1       # no effect if not going parallel
        training_service = 'local'  # no effect if not using training service

        execution_engine = None
        model_format = None

        try:
            from nni.nas.oneshot.pytorch.strategy import OneShotStrategy, is_supernet
            if isinstance(strategy, OneShotStrategy):
                _logger.info('Strategy is found to be a one-shot strategy. '
                             'Setting execution engine to "sequential" and format to "keep".')
                execution_engine = 'sequential'
                model_format = 'raw'
            if is_supernet(model_space):
                _logger.info('Model space is found to be a one-shot supernet. '
                             'Setting execution engine to "sequential" and format to "keep" to preserve the weights.')
                execution_engine = 'sequential'
                model_format = 'raw'
        except ImportError:
            _logger.warning('Import of one-shot strategy failed. Assuming the strategy is not one-shot.')

        if execution_engine is None:
            _logger.info('Using execution engine based on training service. Trial concurrency is set to 1.')
            execution_engine = 'ts'
        if model_format is None:
            _logger.info('Using simplified model format.')
            model_format = 'simplified'

        if execution_engine == 'ts':
            _logger.info('Using local training service.')

            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                _logger.warning(
                    'GPU found but will not be used. Please set `experiment.config.trial_gpu_number` to '
                    'the number of GPUs you want to use for each trial.'
                )

        config = NasExperimentConfig(
            training_service_platform=training_service,
            execution_engine=execution_engine,
            model_format=model_format,
            trial_concurrency=trial_concurrency
        )

        return config

    def _canonicalize(self, parents):
        if self.search_space != RESERVED:
            raise ValueError('`search_space` field can not be customized in NAS experiment.')

        if not Path(self.trial_code_directory).samefile(Path.cwd()):
            raise ValueError('`trial_code_directory` field can not be customized in NAS experiment.')

        trial_command_suffix = '-m nni.nas.execution.training_service trial'
        trial_command_template = '{envs} {python} ' + trial_command_suffix
        if self.trial_command and not self.trial_command.endswith(trial_command_suffix):
            raise ValueError('`trial_command` field can not be customized in NAS experiment.')

        _trial_command_params = {
            # Default variables
            'envs': '',
            # TODO: maybe use sys.executable rendered in trial side (e.g., trial_runner)
            'python': sys.executable,

            # This should override the parameters above.
            **(self._trial_command_params or {})
        }

        self.trial_command = trial_command_template.format(**_trial_command_params).strip()

        if isinstance(self.execution_engine, SequentialEngineConfig):
            if not utils.is_missing(self.trial_concurrency) and self.trial_concurrency != 1:
                raise ValueError('`trial_concurrency` must be 1 for sequential execution engine.')
            self.trial_concurrency = 1

            if not utils.is_missing(self.training_service):
                _logger.warning('`training_service` will be overridden for sequential execution engine.')

            self.training_service = utils.training_service_config_factory('local')

        super()._canonicalize([self] + parents)

        self._canonical = True
