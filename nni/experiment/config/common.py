# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ConfigBase, PathLike
from . import util

__all__ = [
    'BasicExperimentConfig',
    'ExperimentConfig',
    'FullExperimentConfig',
    'TrainingServiceConfig',
]

ExperimentConfig = 'Union[BasicExperimentConfig, FullExperimentConfig]'

class TrainingServiceConfig(ConfigBase):
    def _cluster_metadata(self, experiment_config: ExperimentConfig) -> Any:
        raise NotImplementedError()


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
    training_service_config: TrainingServiceConfig

    @property
    def _field_validation_rules():
        return _field_validation

    def finalize(self) -> None:
        if isinstance(self.trial_command, str):
            self.trial_command = [self.trial_command]
        self.trial_code_directory = util.absolute_path(self.trial_code_directory, self)
        self.training_service_config.finalize()


@dataclass(init=False)
class FullExperimentConfig(ExperimentConfig):
    experiment_name: Optional[str]
    training_service: Optional[str]
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
    reuse_mode: bool = False
    debug: bool = False
    log_level: Optional[Literal["trace", "debug", "info", "warning", "error", "fatal"]] = None
    experiment_working_directory: Optional[PathLike] = None
    tuner_gpu_indices: Optional[Union[List[int], str]] = None
    tuner: Optional[AlgorithmConfig] = None
    accessor: Optional[AlgorithmConfig] = None
    advisor: Optional[AlgorithmConfig] = None
    training_service_config: TrainingServiceConfig

    @property
    def _field_validation_rules():
        return _field_validation

    @property
    def _class_validation_rules():
        return _class_validation_full

    def finalize(self) -> None:
        util.absolute_path(self, 'search_space_file')
        if isinstance(self.trial_command, str):
            self.trial_command = [self.trial_command]
        util.absolute_path(self, 'trial_code_directory')
        util.absolute_path(self, 'experiment_working_directory')
        if isinstance(str, self.tuner_gpu_indices):
            self.tuner_gpu_indices = [int(idx) for idx in self.tuner_gpu_indices.split(',')]
        if self.tuner is not None:
            self.tuner.finalize()
        if self.assessor is not None:
            self.assessor.finalize()
        if self.advisor is not None:
            self.advisor.finalize()
        self.training_service_config.finalize()

    @staticmethod
    def from_json(data: Dict[str, Any]) -> FullExperimentConfig:
        # rename fields like "local_config" to "training_service_config"
        data = {util.case_insensitive(key): value for key, value in data.items()}
        for key, value in data.items():
            if key.endswith('config') and key != 'trainingserviceconfig':
                ts = key[:-len('config')]
                if data.get('trainingservice', ts) != ts:
                    raise ValueError(f'Training service {data["trainingservice"]} does not match config {key}')
                if 'training_service_config' in data:
                    raise ValueError('Multiple training service config detected')
                ts_config = value
                data['trainingserviceconfig'] = TrainingService.from_json(value)
        return FullExperimentConfig(**data)


_field_validation = {
    'training_service': lambda val, config: (
        val == config.training_service_config._training_service,
        '{val} does not match training_service_config type {type(config.training_service_config).__name__}'
    ),
    'search_space_file': lambda val, _: (Path(val).is_file(), '"{val}" does not exist or is not regular file',
    'trial_code_directory': lambda val, _: (Path(val).is_dir(), '"{val}" does not exist or is not directory'),
    'trial_concurrency': lambda val, _: val > 0,
    'trial_gpu_number': lambda val, _: val >= 0,
    'max_experiment_duration': lambda val, _: unit.parse_time(val) > 0,
    'max_trial_number': lambda val, _: val > 0,
    'experiment_working_directory': lambda val, _: Path(val).mkdir(parents=True, exist_ok=True) or True,
    'tuner_gpu_indices': lambda val, _: isinstance(val, str) or (all(i >= 0 for i in val) and len(val) == len(set(val)))
}

_class_validation_full = [
    lambda config: (
        sum(config.search_space_file is not None, config.search_space is not None, config.use_annotation) == 1,
        "One and only one of search_space_file, search_space, and use_annotation must be set"
    )
]
