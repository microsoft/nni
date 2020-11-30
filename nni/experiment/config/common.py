# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from .base import ConfigBase, PathLike
from . import util

__all__ = [
    'BasicExperimentConfig',
    'ExperimentConfig',
    'FullExperimentConfig',
    'TrainingServiceConfig',
]

ExperimentConfig = Union['BasicExperimentConfig', 'FullExperimentConfig']


class TrainingServiceConfig(ConfigBase):
    platform: str

    def __init__(self, **kwargs):
        if type(self) is TrainingServiceConfig:
            raise NotImplementedError('This class is abstract')
        super().__init__()


@dataclass(init=False)
class AlgorithmConfig(ConfigBase):
    registered_name: Optional[str]
    class_name: Optional[str]
    code_directory: Optional[PathLike]
    class_arguments: Dict[str, Any]

    def validate(self):
        super().validate()
        if self.registered_name is None:
            if self.class_name is None:
                raise ValueError('name or class_name must be set')
            if self.code_directory is not None and not Path(self.code_directory).is_dir():
                raise ValueError(f'code_directory "{self.code_directory}" does not exist or is not directory')
        else:
            if self.class_name is not None or self.code_directory is not None:
                raise ValueError(f'When registered_name is specified, class_name and code_directory cannot be used')


@dataclass(init=False)
class BasicExperimentConfig(ConfigBase):
    experiment_name: Optional[str]
    search_space: Any
    trial_command: Union[str, List[str]]
    trial_code_directory: PathLike = '.'
    trial_concurrency: int
    trial_gpu_number: Optional[int] = None
    max_experiment_duration: Optional[str] = None
    max_trial_number: Optional[int] = None
    training_service: TrainingServiceConfig

    @property
    def _canonical_rules(self):
        return _canonical_rules

    @property
    def _validation_rules(self):
        return _validation_rules


# not subclassing BasicExperimentConfig, for readability
@dataclass(init=False)
class FullExperimentConfig(ConfigBase):
    experiment_name: Optional[str]
    search_space_file: Optional[PathLike] = None
    search_space: Any = None
    trial_command: Union[str, List[str]]
    trial_code_directory: PathLike = '.'
    trial_concurrency: int
    trial_gpu_number: Optional[int] = None
    max_experiment_duration: Optional[str] = None
    max_trial_number: Optional[int] = None
    nni_manager_ip: Optional[str] = None
    use_annotation: bool = False
    debug: bool = False
    log_level: Optional[Literal["trace", "debug", "info", "warning", "error", "fatal"]] = None
    experiment_working_directory: Optional[PathLike] = None
    tuner_gpu_indices: Optional[Union[List[int], str]] = None
    tuner: Optional[AlgorithmConfig] = None
    accessor: Optional[AlgorithmConfig] = None
    advisor: Optional[AlgorithmConfig] = None
    training_service: TrainingServiceConfig

    @property
    def _canonical_rules(self):
        return _canonical_rules

    @property
    def _validation_rules(self):
        return _validation_rules

    def validate(self) -> None:
        super().validate()
        ss_cnt = sum([
            self.search_space_file is not None,
            self.search_space is not None,
            bool(self.use_annotation)
        ])
        if ss_cnt != 1:
            raise ValueError("One and only one of search_space_file, search_space, and use_annotation must be set")


_canonical_rules = {
    'search_space_file': util.canonical_path,
    'trial_command': lambda value: [value] if isinstance(value, str) else value,
    'trial_code_directory': util.canonical_path,
    'max_experiment_duration': lambda value: str(util.parse_time(value)) + 's',
    'experiment_working_directory': util.canonical_path,
    'tuner_gpu_indices': lambda value: [int(idx) for idx in value.split(',')] if isinstance(value, str) else value
}

_validation_rules = {
    'search_space_file': lambda value: (Path(value).is_file(), '"{value}" does not exist or is not regular file'),
    'trial_code_directory': lambda value: (Path(value).is_dir(), '"{value}" does not exist or is not directory'),
    'trial_concurrency': lambda value: value > 0,
    'trial_gpu_number': lambda value: value >= 0,
    'max_experiment_duration': lambda value: util.parse_time(value) > 0,
    'max_trial_number': lambda value: value > 0,
    'experiment_working_directory': lambda value: Path(value).mkdir(parents=True, exist_ok=True) or True,
    'tuner_gpu_indices': lambda value: all(i >= 0 for i in value) and len(value) == len(set(value))
}
