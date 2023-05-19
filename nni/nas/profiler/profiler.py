# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from nni.mutable import Sample
from nni.mutable.symbol import SymbolicExpression
from nni.nas.space import BaseModelSpace


class Profiler:
    """Profiler is a class that profiles the performance of a model within a space.

    Unlike the regular profilers, NAS profilers are initialized with a space,
    and are expected to do some pre-computation with the space,
    such that it can quickly computes the performance of a model given a sample within a space.

    A profiler can return many things, such as latency, throughput, model size, etc.
    Mostly things that can be computed instantly, or can be computed with a small overhead.
    For metrics that require training, please use :class:`~nni.nas.evaluator.Evaluator` instead.
    """

    def __init__(self, model_space: BaseModelSpace):
        pass

    def profile(self, sample: Sample) -> float:
        raise NotImplementedError()


class ExpressionProfiler(Profiler):
    """Profiler whose :meth:`profile` method is an evaluation of a precomputed expression.

    This type of profiler is useful for optimization and analysis.
    For example, to find the best model size is equivalent to find the minimum value of the expression.
    Users can also compute the mathematical expression for a distribution of model samples.
    """

    expression: SymbolicExpression | float

    def profile(self, sample: Sample) -> float:
        if isinstance(self.expression, (float, int)):
            return float(self.expression)
        else:
            return self.expression.evaluate(sample)
