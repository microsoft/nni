# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Experiment configuration structures.
"""

from dataclasses import MISSING, dataclass, fields
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass(init=False)
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


    _placeholders = {
        'experiment_name': '_unset_',
        'search_space': '_unset_',
        'trial_concurrency': -1,
        'trial_command': '_unset_',
        'trial_code_directory': '_unset_'
    }

    _value_range = {
        'max_execution_seconds': lambda x: x is None or x > 0,
        'max_trial_number': lambda x: x is None or x > 0,
        'trial_concurrency': lambda x: x > 0,
        'trial_gpu_number': lambda x: x >= 0
    }

    _json_schema = {
        '_training_service': lambda value: ('trainingServicePlatform', value),
        'max_execution_seconds': lambda value: ('maxExecDuration', str(value) + 's'),
        'search_space': lambda value: ('searchSpace', json.dumps(value)),
        'trial_code_directory': lambda value: ('trialCodeDir', str(value)),
    }


    def __init__(self, **kwargs):
        for field in fields(self):
            if field.name in kwargs:
                setattr(self, field.name, kwargs[field.name])
            elif field.default != MISSING:
                setattr(self, field.name, field.default)
            else:
                setattr(self, field.name, type(self)._placeholders[field.name])


    def validate(self) -> None:
        self._general_validate()

        if not Path(self.trial_code_directory).is_dir():
            raise ValueError(f'Trial code directory "{self.trial_code_directory}" does not exist or is not directory')


    def to_json(self) -> Dict[str, Any]:
        ret = self._general_to_json()
        ret['authorName'] = ''
        ret['tuner'] = {'builtinTunerName': '_placeholder_'}
        return ret


    def _general_validate(self) -> None:
        # check for existence
        for key, placeholder_value in type(self)._placeholders.items():
            if getattr(self, key) == placeholder_value:
                raise ValueError(f'Field "{key}" is not set')

        # TODO: check for type

        # check for value
        for key, condition in type(self)._value_range.items():
            value = getattr(self, key)
            if not condition(getattr(self, key)):
                raise ValueError(f'Field "{key}" ({repr(value)}) out of range')


    def _general_to_json(self) -> Dict[str, Any]:
        ret = {}

        for field in fields(self):
            key = field.name
            value = getattr(self, key)
            special_schema = type(self)._json_schema.get(key)
            if special_schema is None:
                key = _to_camel_case(key)
            else:
                key, value = special_schema(value)
            ret[key] = value

        if self.extra_config:
            ret.update(self.extra_config)

        return ret


    @staticmethod
    def _create(training_service: str) -> 'ExperimentConfig':
        # create an empty configuration for given training service
        for cls in ExperimentConfig.__subclasses__():
            for field in fields(cls):
                if field.name == '_training_service' and field.default == training_service:
                    return cls(**cls._placeholders)
        raise ValueError(f'Unrecognized training service {training_service}')


def _to_camel_case(snake):
    words = snake.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])
