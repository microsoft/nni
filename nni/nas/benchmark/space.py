# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['BenchmarkModelSpace', 'SlimBenchmarkSpace']

import logging
from typing import TYPE_CHECKING, overload

from nni.mutable import MutableDict
from nni.nas.space import RawFormatModelSpace, BaseModelSpace

if TYPE_CHECKING:
    from .evaluator import BenchmarkEvaluator

_logger = logging.getLogger(__name__)


class SlimBenchmarkSpace(BaseModelSpace, MutableDict):
    """Example model space without deep learning frameworks.

    When constructing this, the dict should've been already simplified and validated.

    It could look like::

        {
            'layer1': nni.choice('layer1', ['a', 'b', 'c']),
            'layer2': nni.choice('layer2', ['d', 'e', 'f']),
        }
    """


class BenchmarkModelSpace(RawFormatModelSpace):
    """
    Model space that is specialized for benchmarking.

    We recommend using this model space for benchmarking, for its validation and efficiency.

    Parameters
    ----------
    model_space
        If not provided, it will be set to the default model space of the evaluator.
    evaluator
        Evaluator that will be used to benchmark the space.

    Examples
    --------
    Can be either::

        BenchmarkModelSpace(evaluator)

    or::

        BenchmarkModelSpace(pytorch_model_space, evaluator)

    In the case where the model space is provided, it will be validated by the evaluator and must be a match.
    """

    @overload
    def __init__(self, model_space: BenchmarkEvaluator):
        ...

    @overload
    def __init__(self, model_space: BaseModelSpace):
        ...

    @overload
    def __init__(self, model_space: None, evaluator: BenchmarkEvaluator):
        ...

    def __init__(self, model_space: BaseModelSpace | BenchmarkEvaluator | None, evaluator: BenchmarkEvaluator | None = None):
        from .evaluator import BenchmarkEvaluator

        if isinstance(model_space, BenchmarkEvaluator):
            assert evaluator is None
            evaluator = model_space
            model_space = None

        if not isinstance(evaluator, BenchmarkEvaluator):
            raise ValueError(f'Expect evaluator to be BenchmarkEvaluator, got {evaluator}')
        if model_space is None:
            _logger.info('Model space is not set. Using default model space from evaluator: %s', evaluator)
            model_space = evaluator.default_space()
        else:
            evaluator.validate_space(model_space)

        super().__init__(model_space, evaluator)

    def executable_model(self):
        raise RuntimeError(f'{self.__class__.__name__} is not executable. Please use `sample` instead.')
