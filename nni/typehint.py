# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Types for static checking.
"""

__all__ = ['Parameters', 'SearchSpace', 'TrialMetric', 'TrialRecord', 'ParameterRecord']

from typing import Any, Dict, List

from typing_extensions import Literal, TypedDict, NotRequired

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

class ParameterRecord(TypedDict):
    """The format which is used to record parameters at NNI manager side.

    :class:`~nni.runtime.msg_dispatcher.MsgDispatcher` packs the parameters generated by tuners
    into a :class:`ParameterRecord` and sends it to NNI manager.
    NNI manager saves the tuner into database and sends it to trial jobs when they ask for parameters.
    :class:`~nni.runtime.trial_command_channel.TrialCommandChannel` receives the :class:`ParameterRecord`
    and then hand it over to trial.

    Most users don't need to use this class directly.
    """
    parameter_id: int
    parameters: Parameters
    parameter_source: NotRequired[Literal['algorithm', 'customized', 'resumed']]

    # NOTE: in some cases the record might contain extra fields,
    # but they are undocumented and should not be used by users.
    parameter_index: NotRequired[int]
    trial_job_id: NotRequired[str]
