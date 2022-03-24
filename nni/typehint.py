# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Types for static checking.
"""

__all__ = [
    'Literal',
    'Parameters', 'SearchSpace', 'TrialMetric', 'TrialRecord',
]

import sys
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING or sys.version_info >= (3, 8):
    from typing import Literal, TypedDict
else:
    from typing_extensions import Literal, TypedDict

Parameters = Dict[str, Any]
"""
Return type of :func:`nni.get_next_parameter`.

For built-in tuners, this is a ``dict`` whose content is defined by :doc:`search space </hpo/search_space>`.

Customized tuners do not need to follow the constraint and can use anything serializable.
"""

class _ParameterSearchSpace(TypedDict):
    _type: Literal[
        'choice', 'randint',
        'uniform', 'loguniform', 'quniform', 'qloguniform',
        'normal', 'lognormal', 'qnormal', 'qlognormal',
    ]
    _value: List[Any]

SearchSpace = Dict[str, _ParameterSearchSpace]
"""
Type of ``experiment.config.search_space``.

For built-in tuners, the format is detailed in :doc:`/hpo/search_space`.

Customized tuners do not need to follow the constraint and can use anything serializable, except ``None``.
"""

TrialMetric = float
"""
Type of the metrics sent to :func:`nni.report_final_result` and :func:`nni.report_intermediate_result`.

For built-in tuners it must be a number (``float``, ``int``, ``numpy.float32``, etc).

Customized tuners do not need to follow this constraint and can use anything serializable.
"""

class TrialRecord(TypedDict):
    parameter: Parameters
    value: TrialMetric
