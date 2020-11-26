# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Experiment configuration structures.
"""

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


    _placeholder = {
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
        'search_space': lambda value: ('searchSpace', json.dumps(value)),
        'max_execution_seconds': lambda value: ('maxExecDuration', value or 99999),
        'max_trial_number': lambda value: ('maxTrialNum', value or 99999),
        'trial_command': lambda value: (None, None),
        'trial_code_directory': lambda value: (None, None),
        'trial_gpu_number': lambda value: (None, None),
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
        self._general_validate()

        if not Path(self.trial_code_directory).is_dir():
            raise ValueError(f'Trial code directory "{self.trial_code_directory}" does not exist or is not directory')


    def to_json(self) -> Dict[str, Any]:
        ret = self._general_to_json()
        ret['authorName'] = '_'
        ret['tuner'] = {'builtinTunerName': '_placeholder_'}
        return ret


    def _general_validate(self) -> None:
        # check existence
        for key, placeholder_value in type(self)._placeholder.items():
            if getattr(self, key) == placeholder_value:
                raise ValueError(f'Field "{key}" is not set')

        # TODO: check type

        # check value
        for key, condition in type(self)._value_range.items():
            value = getattr(self, key)
            if not condition(getattr(self, key)):
                raise ValueError(f'Field "{key}" ({repr(value)}) out of range')


    def _general_to_json(self) -> Dict[str, Any]:
        ret = {}

        for field in dataclasses.fields(self):
            key = field.name
            if key == 'extra_config':
                continue
            value = getattr(self, key)
            special_schema = type(self)._json_schema.get(key)
            if special_schema is None:
                ret[_to_camel_case(key)] = value
            else:
                key, value = special_schema(value)
                if key:
                    ret[key] = value

        if self.extra_config:
            ret.update(self.extra_config)

        return ret


    @staticmethod
    def create_template(training_service: str) -> 'ExperimentConfig':
        for cls in ExperimentConfig.__subclasses__():
            for field in dataclasses.fields(cls):
                if field.name == '_training_service' and field.default == training_service:
                    return cls()
        raise ValueError(f'Unrecognized training service {training_service}')


def _to_camel_case(snake):
    words = snake.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])
