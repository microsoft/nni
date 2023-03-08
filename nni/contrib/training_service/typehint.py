# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from typing import TypedDict, Any, Type, TypeVar, get_origin
from typing_extensions import Literal, TypeGuard

_logger = logging.getLogger(__name__)

T = TypeVar('T')


def typed_dict_validation(typ: Type[T], instance: Any) -> TypeGuard[T]:
    # https://stackoverflow.com/questions/66665336/validate-python-typeddict-at-runtime
    if not isinstance(instance, dict):
        _logger.error('Validation failed for %s. Instance is not a dict: %s', typ, type(instance))
        return False

    for property_name, property_type in typ.__annotations__.items():
        if property_name not in instance:
            # Check for missing keys
            _logger.error('Validation failed for %s. Missing key: %s', typ, property_name)
            return False

        value = instance[property_name]
        if property_type in (int, float, bool, str):
            # Check for type equality
            if not isinstance(value, property_type):
                _logger.error('Validation failed for %s. Wrong type: %s. Expected %s, got %s',
                              typ, property_name, property_type, type(value))
                return False

        elif get_origin(property_type) == Literal:
            # Check literal.
            if value not in property_type.__args__:
                _logger.error('Validation failed for %s. Expect literal to be one of %s, got %s',
                              typ, property_type.__args__, value)
                return False

        else:
            # Assuming a nested typed dict.
            result = typed_dict_validation(property_type, value)
            if result is False:
                return False

    return True


class Trial(TypedDict):
    id: str
    sequence: int
    experiment: str
    command: str
    parameter: str   # Serialized JSON string. If empty, the trial will receive no parameter.
    # time_limit: float

# Command types are as few as possible.
# The implementation also tries to avoid unnecessary dependencies,
# to increase the robustness.


UpstreamCommandType = Literal['create', 'kill', 'wakeup']       # manager -> worker
DownstreamCommandType = Literal['metric', 'status', 'awake']    # worker -> manager
Status = Literal['waiting', 'running', 'succeeded', 'failed', 'interrupted']


class CreateCommand(TypedDict):
    command_type: Literal['create']
    trial: Trial


class KillCommand(TypedDict):
    command_type: Literal['kill']
    id: str


class MetricCommand(TypedDict):
    command_type: Literal['metric']
    id: str
    metric: str  # Serialized JSON string.


class TrialStatusCommand(TypedDict):
    command_type: Literal['status']
    id: str
    status: Status


class WakeUpCommand(TypedDict):
    # Request the worker to report its status (more frequently).
    command_type: Literal['wakeup']


class ReportAwakeCommand(TypedDict):
    # The only way to report that the worker is alive (and idle or occupied).
    command_type: Literal['awake']
    time: float
    # NOTE: time here is only for verbose.
    # It should be avoided from usage because the cluster might have a different time from local.
    idle: bool
