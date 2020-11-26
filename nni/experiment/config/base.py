# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclasses.dataclass(init=False)
class ExperimentConfig:
    experiment_name: str
    search_space: Any
    max_execution_seconds: Optional[int] = None
    max_trial_number: Optional[int] = None
    trial_concurrency: int
    trial_command: str
    trial_code_directory: Union[Path, str]
    trial_gpu_number: int = 0
    extra_config: Optional[Dict[str, str]] = None

    _training_service: str


    # these values will be used to create template object,
    # and the user should overwrite them later.
    _placeholder = {
        'experiment_name': '_unset_',
        'search_space': '_unset_',
        'trial_concurrency': -1,
        'trial_command': '_unset_',
        'trial_code_directory': '_unset_'
    }

    # simple validation functions
    # complex validation logic with special error message should go to `validate()` method instead
    _value_range = {
        'max_execution_seconds': lambda x: x is None or x > 0,
        'max_trial_number': lambda x: x is None or x > 0,
        'trial_concurrency': lambda x: x > 0,
        'trial_gpu_number': lambda x: x >= 0
    }


    def __init__(self, **kwargs):
        for field in dataclasses.fields(self):
            if field.name in kwargs:
                setattr(self, field.name, kwargs[field.name])
            elif field.default != dataclasses.MISSING:
                setattr(self, field.name, field.default)
            else:
                setattr(self, field.name, type(self)._placeholder[field.name])


    def validate(self) -> None:
        # check existence
        for key, placeholder_value in type(self)._placeholder.items():
            if getattr(self, key) == placeholder_value:
                raise ValueError(f'Field "{key}" is not set')

        # TODO: check type

        # check value
        for key, condition in type(self)._value_range.items():
            value = getattr(self, key)
            if not condition(value):
                raise ValueError(f'Field "{key}" ({repr(value)}) out of range')

        # check special fields
        if not Path(self.trial_code_directory).is_dir():
            raise ValueError(f'Trial code directory "{self.trial_code_directory}" does not exist or is not directory')


    def experiment_config_json(self) -> Dict[str, Any]:
        # this only contains the common part for most (if not all) training services
        # subclasses should override it to provide exclusive fields
        return {
            'authorName': '_',
            'experimentName': self.experiment_name,
            'trialConcurrency': self.trial_concurrency,
            'maxExecDuration': self.max_execution_seconds or (999 * 24 * 3600),
            'maxTrialNum': self.max_trial_number or 99999,
            'searchSpace': json.dumps(self.search_space),
            'trainingServicePlatform': self._training_service,
            'tuner': {'builtinTunerName': '_user_created_'},
            **(self.extra_config or {})
        }

    def cluster_metadata_json(self) -> Any:
        # the cluster metadata format is a total mess
        # leave it to each subclass before we refactoring nni manager
        raise NotImplementedError()


    @staticmethod
    def create_template(training_service: str) -> 'ExperimentConfig':
        for cls in ExperimentConfig.__subclasses__():
            for field in dataclasses.fields(cls):
                if field.name == '_training_service' and field.default == training_service:
                    return cls()
        raise ValueError(f'Unrecognized training service {training_service}')
