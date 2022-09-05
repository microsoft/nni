# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Deduplicate repeated parameters.

No guarantee for forward-compatibility.
"""

from __future__ import annotations

import logging
import typing

import nni
from .formatting import FormattedParameters, FormattedSearchSpace, ParameterSpec, deformat_parameters

# TODO:
# Move main logic of basic tuners (random and grid search) into SDK,
# so we can get rid of private methods and circular dependency.
if typing.TYPE_CHECKING:
    from nni.algorithms.hpo.gridsearch_tuner import GridSearchTuner

_logger = logging.getLogger(__name__)

class Deduplicator:
    """
    A helper for tuners to deduplicate generated parameters.

    When the tuner generates an already existing parameter,
    calling this will return a new parameter generated with grid search.
    Otherwise it returns the orignial parameter object.

    If all parameters have been generated, raise ``NoMoreTrialError``.

    All search space types, including nested choice, are supported.

    Resuming and updating search space are not supported for now.
    It will not raise error, but may return duplicate parameters.

    See random tuner's source code for example usage.
    """

    def __init__(self, formatted_search_space: FormattedSearchSpace):
        self._space: FormattedSearchSpace = formatted_search_space
        self._never_dup: bool = any(_spec_never_dup(spec) for spec in self._space.values())
        self._history: set[str] = set()
        self._grid_search: GridSearchTuner | None = None

    def __call__(self, formatted_parameters: FormattedParameters) -> FormattedParameters:
        if self._never_dup or self._not_dup(formatted_parameters):
            return formatted_parameters

        if self._grid_search is None:
            _logger.info(f'Tuning algorithm generated duplicate parameter: {formatted_parameters}')
            _logger.info(f'Use grid search for deduplication.')
            self._init_grid_search()

        while True:
            new = self._grid_search._suggest()  # type: ignore
            if new is None:
                raise nni.NoMoreTrialError()
            if self._not_dup(new):
                return new

    def _init_grid_search(self) -> None:
        from nni.algorithms.hpo.gridsearch_tuner import GridSearchTuner
        self._grid_search = GridSearchTuner()
        self._grid_search.history = self._history
        self._grid_search.space = self._space
        self._grid_search._init_grid()

    def _not_dup(self, formatted_parameters: FormattedParameters) -> bool:
        params = deformat_parameters(formatted_parameters, self._space)
        params_str = typing.cast(str, nni.dump(params, sort_keys=True))
        if params_str in self._history:
            return False
        else:
            self._history.add(params_str)
            return True

def _spec_never_dup(spec: ParameterSpec) -> bool:
    if spec.is_nested():
        return False  # "not chosen" duplicates with "not chosen"
    if spec.categorical or spec.q is not None:
        return False
    if spec.normal_distributed:
        return spec.sigma > 0
    else:
        return spec.low < spec.high
