# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Top level experiement configuration class, ``ExperimentConfig``.
"""

__all__ = ['ExperimentConfig']

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union
from typing_extensions import Literal

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

    .. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html
    """
    # TODO:
    # The behavior described below is expected but does not work,
    # because some fields are consumed by TrialDispatcher outside environment service.
    # Add the lines to docstr when we fix this issue.

    # Fields commented as "training service field" acts like shortcut for all training services.
    # Users can either specify them here or inside training service config.
    # In latter case hybrid training services can have different settings.

    experiment_name: Optional[str] = None
    experiment_type: Literal['hpo'] = 'hpo'
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

    def __new__(cls, *args, **kwargs) -> 'ExperimentConfig':
        if cls is not ExperimentConfig:
            # The __new__ only applies to the base class.
            return super().__new__(cls)
        if kwargs.get('experimentType') == 'nas':
            # Loaded by JSON or YAML.
            # Send the kwargs to the NAS config constructor.
            from nni.nas.experiment import NasExperimentConfig
            return NasExperimentConfig.__new__(NasExperimentConfig)
        else:
            return super().__new__(cls)

    def __init__(self, training_service_platform=None, **kwargs):
        super().__init__(**kwargs)
        if training_service_platform is not None:
            # the user chose to init with `config = ExperimentConfig('local')` and set fields later
            # we need to create empty training service & algorithm configs to support `config.tuner.name = 'random'`
            assert utils.is_missing(self.training_service)
            if isinstance(training_service_platform, list):
                self.training_service = [utils.training_service_config_factory(ts) for ts in training_service_platform]
            else:
                self.training_service = utils.training_service_config_factory(training_service_platform)
            for algo_type in ['tuner', 'assessor', 'advisor']:
                # add placeholder items, so users can write `config.tuner.name = 'random'`
                if getattr(self, algo_type) is None:
                    setattr(self, algo_type, _AlgorithmConfig(name='_none_', class_args={}))
        elif not utils.is_missing(self.training_service):
            # training service is set via json or constructor
            if isinstance(self.training_service, list):
                self.training_service = [utils.load_training_service_config(ts) for ts in self.training_service]
            else:
                self.training_service = utils.load_training_service_config(self.training_service)

    def _canonicalize(self, _parents):
        if self.log_level is None:
            self.log_level = 'debug' if self.debug else 'info'
        self.tuner_gpu_indices = utils.canonical_gpu_indices(self.tuner_gpu_indices)

        for algo_type in ['tuner', 'assessor', 'advisor']:
            algo = getattr(self, algo_type)

            # TODO: need a more universal solution for similar problems
            if isinstance(algo, dict):
                # the base class should have converted it to `_AlgorithmConfig` if feasible
                # it is a `dict` here means an exception was raised during the convertion attempt
                # we do the convertion again to show user the error message
                _AlgorithmConfig(**algo)  # pylint: disable=not-a-mapping

            if algo is not None and algo.name == '_none_':  # type: ignore
                setattr(self, algo_type, None)

        if self.advisor is not None:
            assert self.tuner is None, '"advisor" is deprecated. You should only set "tuner".'
            self.tuner = self.advisor
            self.advisor = None

        super()._canonicalize([self])

        if self.search_space_file is not None:
            yaml_error = None
            try:
                self.search_space = _load_search_space_file(self.search_space_file)
            except Exception as e:
                yaml_error = repr(e)
            if yaml_error is not None:  # raise it outside except block to make stack trace clear
                msg = f'ExperimentConfig: Failed to load search space file "{self.search_space_file}": {yaml_error}'
                raise ValueError(msg)

        if self.nni_manager_ip is None:
            # show a warning if user does not set nni_manager_ip. we have many issues caused by this
            # the simple detection logic won't work for hybrid, but advanced users should not need it
            # ideally we should check accessibility of the ip, but it need much more work
            platform = getattr(self.training_service, 'platform')
            has_ip = isinstance(getattr(self.training_service, 'nni_manager_ip'), str)  # not None or MISSING
            if platform and platform != 'local' and not has_ip:
                ip = utils.get_ipv4_address()
                msg = f'nni_manager_ip is not set, please make sure {ip} is accessible from training machines'
                logging.getLogger('nni.experiment.config').warning(msg)

    def _validate_canonical(self):
        super()._validate_canonical()

        space_cnt = (self.search_space is not None) + (self.search_space_file is not None)
        if self.use_annotation and space_cnt != 0:
            raise ValueError('ExperimentConfig: search space must not be set when annotation is enabled')
        if not self.use_annotation and space_cnt < 1:
            raise ValueError('ExperimentConfig: search_space and search_space_file must be set one')

        # to make the error message clear, ideally it should be:
        # `if concurrency < 0: raise ValueError('trial_concurrency ({concurrency}) must greater than 0')`
        # but I believe there will be hardy few users make this kind of mistakes, so let's keep it simple
        assert self.trial_concurrency > 0
        assert self.max_experiment_duration is None or utils.parse_time(self.max_experiment_duration) > 0
        assert self.max_trial_number is None or self.max_trial_number > 0
        assert self.max_trial_duration is None or utils.parse_time(self.max_trial_duration) > 0
        assert self.log_level in ['fatal', 'error', 'warning', 'info', 'debug', 'trace']

        # following line is disabled because it has side effect
        # enable it if users encounter problems caused by failure in creating experiment directory
        # currently I have only seen one issue of this kind
        #Path(self.experiment_working_directory).mkdir(parents=True, exist_ok=True)

        if type(self).__name__ != 'NasExperimentConfig':
            utils.validate_gpu_indices(self.tuner_gpu_indices)

            if self.tuner is None:
                raise ValueError('ExperimentConfig: tuner must be set')

def _load_search_space_file(search_space_path):
    # FIXME
    # we need this because PyYAML 6.0 does not support YAML 1.2,
    # which means it is not fully compatible with JSON
    content = Path(search_space_path).read_text(encoding='utf8')
    try:
        return json.loads(content)
    except Exception:
        return yaml.safe_load(content)
