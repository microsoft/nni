# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = [
    'Literal', 'TypeAlias',
    'Parameters', 'SearchSpace', 'TrialMetric', 'TrialRecord',
]

import sys
import typing

if typing.TYPE_CHECKING or sys.version_info >= (3, 10):
    from typing import Any, Literal, TypeAlias, TypedDict
    
    Parameters: TypeAlias = dict[str, Any]
    SearchSpace: TypeAlias = dict[str, Any]
    TrialMetric: TypeAlias = float
    
    class TrialRecord(TypedDict):
        parameter: Parameters
        value: TrialMetric

else:
    from typing import Any

    Literal = Any
    TypeAlias = Any

    Parameters = Any
    SearchSpace = Any
    TrialMetric = Any
    TrialRecord = Any
