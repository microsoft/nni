# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Top level experiement configuration class, ``ExperimentConfig``.
"""

__all__ = ['ExperimentConfig']

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml

from .algorithm import _AlgorithmConfig
from .base import ConfigBase
from .shared_storage import SharedStorageConfig
from .training_service import TrainingServiceConfig
from . import utils

@dataclass(init=False)
class ExperimentConfig(ConfigBase):
    """
    Class of experiment configuration. Check the reference_ for explaination of each field.

    When used in Python experiment API, it can be constructed in two favors:

    1. Create an empty project then set each field

    .. code-block:: python

        config = ExperimentConfig('local')
        config.search_space = {...}
        config.tuner.name = 'random'
        config.training_service.use_active_gpu = True

     2. Use kwargs directly

     .. code-block:: python

        config = ExperimentConfig(
            search_space = {...},
            tuner = AlgorithmConfig(name='random'),
            training_service = LocalConfig(
                use_active_gpu = True
            )
        )

    Fields commented as "training service field" acts like shortcut for all training services.
    Users can either specify them here or inside training service config.
    In latter case hybrid training services can have different settings.

    .. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html
    """

    experiment_name: Optional[str] = None
    search_space_file: Optional[utils.PathLike] = None
    search_space: Any = None
    trial_command: Optional[str] = None  # training service field
    trial_code_directory: utils.PathLike = '.'  # training service field
    trial_concurrency: int
    trial_gpu_number: Optional[int] = None  # training service field
    max_experiment_duration: Union[str, int, None] = None
    max_trial_number: Optional[int] = None
    max_trial_duration: Union[str, int, None] = None
    nni_manager_ip: Optional[str] = None  # training service field
    use_annotation: bool = False
    debug: bool = False
    log_level: Optional[str] = None
    experiment_working_directory: utils.PathLike = '~/nni-experiments'
    tuner_gpu_indices: Union[List[int], int, str, None] = None
    tuner: Optional[_AlgorithmConfig] = None
    assessor: Optional[_AlgorithmConfig] = None
    advisor: Optional[_AlgorithmConfig] = None
    training_service: Union[TrainingServiceConfig, List[TrainingServiceConfig]]
    shared_storage: Optional[SharedStorageConfig] = None

    def __init__(self, training_service_platform=None, **kwargs):
        super().__init__(**kwargs)
        if training_service_platform is not None:
            assert utils.is_missing(self.training_service)
            if isinstance(training_service_platform, list):
                self.training_service = [utils.training_service_config_factory(ts) for ts in training_service_platform]
            else:
                self.training_service = utils.training_service_config_factory(training_service_platform)
            for algo_type in ['tuner', 'assessor', 'advisor']:
                # add placeholder items, so users can write `config.tuner.name = 'random'`
                if getattr(self, algo_type) is None:
                    setattr(self, algo_type, _AlgorithmConfig(name='_none_'))

    def _canonicalize(self, _parents):
        self.max_experiment_duration = utils.parse_time(self.max_experiment_duration)
        self.max_trial_duration = utils.parse_time(self.max_trial_duration)
        if self.log_level is None:
            self.log_level = 'debug' if self.debug else 'info'
        self.tuner_gpu_indices = utils.canonical_gpu_indices(self.tuner_gpu_indices)

        for algo_type in ['tuner', 'assessor', 'advisor']:
            algo = getattr(self, algo_type)
            if algo is not None and algo.name == '_none_':
                setattr(self, algo_type, None)

        super()._canonicalize([self])

    def _validate_canonical(self):
        super()._validate_canonical()

        space_cnt = (self.search_space is not None) + (self.search_space_file is not None)
        if self.use_annotation and space_cnt != 0:
            raise ValueError('ExperimentConfig: search space must not be set when annotation is enabled')
        if not self.use_annotation and space_cnt != 1:
            raise ValueError('ExperimentConfig: search_space and search_space_file must be set one')

        if self.search_space_file is not None:
            self.search_space = yaml.safe_load(open(self.search_space_file))

        assert self.trial_concurrency > 0
        assert self.max_experiment_duration is None or self.max_experiment_duration > 0
        assert self.max_trial_number is None or self.max_trial_number > 0
        assert self.max_trial_duration is None or self.max_trial_duration > 0
        assert self.log_level in ['fatal', 'error', 'warning', 'info', 'debug', 'trace']

        # following line is disabled because it has side effect
        # enable it if users encounter problems caused by failure in creating experiment directory
        # currently I have only seen one issue of this kind
        #Path(self.experiment_working_directory).mkdir(parents=True, exist_ok=True)

        utils.validate_gpu_indices(self.tuner_gpu_indices)

        tuner_cnt = (self.tuner is not None) + (self.advisor is not None)
        if tuner_cnt != 1:
            raise ValueError('ExperimentConfig: tuner and advisor must be set one')
