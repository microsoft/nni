# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import ConfigBase, PathLike
from . import util

__all__ = [
    'ExperimentConfig',
    'AlgorithmConfig',
    'CustomAlgorithmConfig',
    'TrainingServiceConfig',
]


@dataclass(init=False)
class _AlgorithmConfig(ConfigBase):
    name: Optional[str] = None
    class_name: Optional[str] = None
    code_directory: Optional[PathLike] = None
    class_args: Optional[Dict[str, Any]] = None

    def validate(self):
        super().validate()
        _validate_algo(self)


@dataclass(init=False)
class AlgorithmConfig(_AlgorithmConfig):
    name: str
    class_args: Optional[Dict[str, Any]] = None


@dataclass(init=False)
class CustomAlgorithmConfig(_AlgorithmConfig):
    class_name: str
    class_directory: Optional[PathLike] = None
    class_args: Optional[Dict[str, Any]] = None


class TrainingServiceConfig(ConfigBase):
    platform: str


@dataclass(init=False)
class ExperimentConfig(ConfigBase):
    experiment_name: Optional[str] = None
    search_space_file: Optional[PathLike] = None
    search_space: Any = None
    trial_command: str
    trial_code_directory: PathLike = '.'
    trial_concurrency: int
    trial_gpu_number: Optional[int] = None
    max_experiment_duration: Optional[str] = None
    max_trial_number: Optional[int] = None
    nni_manager_ip: Optional[str] = None
    use_annotation: bool = False
    debug: bool = False
    log_level: Optional[str] = None
    experiment_working_directory: Optional[PathLike] = None
    tuner_gpu_indices: Optional[Union[List[int], str]] = None
    tuner: Optional[_AlgorithmConfig] = None
    accessor: Optional[_AlgorithmConfig] = None
    advisor: Optional[_AlgorithmConfig] = None
    training_service: TrainingServiceConfig

    def __init__(self, training_service_platform: Optional[str] = None, **kwargs):
        kwargs = util.case_insensitive(kwargs)
        if training_service_platform is not None:
            assert 'trainingservice' not in kwargs
            kwargs['trainingservice'] = util.training_service_config_factory(training_service_platform)
        elif isinstance(kwargs.get('trainingservice'), dict):
            kwargs['trainingservice'] = util.training_service_config_factory(**kwargs['trainingservice'])
        super().__init__(**kwargs)

    def validate(self, initialized_tuner: bool = False) -> None:
        super().validate()
        if initialized_tuner:
            _validate_for_exp(self)
        else:
            _validate_for_nnictl(self)
        if self.trial_gpu_number and hasattr(self.training_service, 'use_active_gpu'):
            if self.training_service.use_active_gpu is None:
                raise ValueError('Please set "use_active_gpu"')

## End of public API ##

    @property
    def _canonical_rules(self):
        return _canonical_rules

    @property
    def _validation_rules(self):
        return _validation_rules


_canonical_rules = {
    'search_space_file': util.canonical_path,
    'trial_code_directory': util.canonical_path,
    'max_experiment_duration': lambda value: f'{util.parse_time(value)}s' if value is not None else None,
    'experiment_working_directory': util.canonical_path,
    'tuner_gpu_indices': lambda value: [int(idx) for idx in value.split(',')] if isinstance(value, str) else value
}

_validation_rules = {
    'search_space_file': lambda value: (Path(value).is_file(), f'"{value}" does not exist or is not regular file'),
    'trial_code_directory': lambda value: (Path(value).is_dir(), f'"{value}" does not exist or is not directory'),
    'trial_concurrency': lambda value: value > 0,
    'trial_gpu_number': lambda value: value >= 0,
    'max_experiment_duration': lambda value: util.parse_time(value) > 0,
    'max_trial_number': lambda value: value > 0,
    'log_level': lambda value: value in ["trace", "debug", "info", "warning", "error", "fatal"],
    'tuner_gpu_indices': lambda value: all(i >= 0 for i in value) and len(value) == len(set(value)),
    'training_service': lambda value: (type(value) is not TrainingServiceConfig, 'cannot be abstract base class')
}

def _validate_for_exp(config: ExperimentConfig) -> None:
    # validate experiment for nni.Experiment, where tuner is already initialized outside
    if config.use_annotation:
        raise ValueError('ExperimentConfig: annotation is not supported in this mode')
    if util.count(config.search_space, config.search_space_file) != 1:
        raise ValueError('ExperimentConfig: search_space and search_space_file must be set one')
    if util.count(config.tuner, config.accessor, config.advisor) != 0:
        raise ValueError('ExperimentConfig: tuner, accessor, and advisor must not be set in for this mode')
    if config.tuner_gpu_indices is not None:
        raise ValueError('ExperimentConfig: tuner_gpu_indices is not supported in this mode')

def _validate_for_nnictl(config: ExperimentConfig) -> None:
    # validate experiment for normal launching approach
    if config.use_annotation:
        if util.count(config.search_space, config.search_space_file) != 0:
            raise ValueError('ExperimentConfig: search_space and search_space_file must not be set with annotationn')
    else:
        if util.count(config.search_space, config.search_space_file) != 1:
            raise ValueError('ExperimentConfig: search_space and search_space_file must be set one')
    if util.count(config.tuner, config.advisor) != 1:
        raise ValueError('ExperimentConfig: tuner and advisor must be set one')

def _validate_algo(algo: AlgorithmConfig) -> None:
    if algo.name is None:
        if algo.class_name is None:
            raise ValueError('Missing algorithm name')
        if algo.code_directory is not None and not Path(algo.code_directory).is_dir():
            raise ValueError(f'code_directory "{algo.code_directory}" does not exist or is not directory')
    else:
        if algo.class_name is not None or algo.code_directory is not None:
            raise ValueError(f'When name is set for registered algorithm, class_name and code_directory cannot be used')
    # TODO: verify algorithm installation and class args
