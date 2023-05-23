# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Guide the one-shot strategy to sample architecture within a target latency.

This module converts the profiling results returned by profiler to something
that one-shot strategies can understand. For example, a loss or some penalty to the reward.

This file is experimentally placed in the oneshot package.
It might be moved to a more general place in the future.
"""

from __future__ import annotations

__all__ = [
    'ProfilerFilter', 'RangeProfilerFilter', 'ProfilerPenalty',
    'ExpectationProfilerPenalty', 'SampleProfilerPenalty'
]

import logging
from typing import cast
from typing_extensions import Literal

import numpy as np
import torch
from torch import nn

from nni.mutable import Sample
from nni.nas.profiler import Profiler, ExpressionProfiler

from .supermodule._expression_utils import expression_expectation

_logger = logging.getLogger(__name__)


class ProfilerFilter:
    """Filter the sample based on the result of the profiler.

    Subclass should implement the ``filter`` method that returns true or false
    to indicate whether the sample is valid.

    Directly call the instance of this class will call the ``filter`` method.
    """

    def __init__(self, profiler: Profiler):
        self.profiler = profiler

    def filter(self, sample: Sample) -> bool:
        raise NotImplementedError()

    def __call__(self, sample: Sample) -> bool:
        return self.filter(sample)


class RangeProfilerFilter(ProfilerFilter):
    """Give up the sample if the result of the profiler is out of range.

    ``min`` and ``max`` can't be both None.

    Parameters
    ----------
    profiler
        The profiler which is used to profile the sample.
    min
        The lower bound of the profiler result. None means no minimum.
    max
        The upper bound of the profiler result. None means no maximum.
    """

    def __init__(self, profiler: Profiler, min: float | None = None, max: float | None = None):  # pylint: disable=redefined-builtin
        super().__init__(profiler)
        self.min_value = min
        self.max_value = max
        if self.min_value is None and self.max_value is None:
            raise ValueError('min and max can\'t be both None')

    def filter(self, sample: Sample) -> bool:
        value = self.profiler.profile(sample)
        if self.min_value is not None and value < self.min_value:
            _logger.debug('Profiler returns %f (smaller than %f) for sample: %s', value, self.min_value, sample)
            return False
        if self.max_value is not None and value > self.max_value:
            _logger.debug('Profiler returns %f (larger than %f) for sample: %s', value, self.max_value, sample)
            return False
        return True


class ProfilerPenalty(nn.Module):
    r"""
    Give the loss a penalty with the result on the profiler.

    Latency losses in `TuNAS <https://arxiv.org/pdf/2008.06120.pdf>`__ and `ProxylessNAS <https://arxiv.org/pdf/1812.00332.pdf>`__
    are its special cases.

    The computation formula is divided into two steps,
    where we first compute a ``normalized_penalty``, whose zero point is when the penalty meets the baseline,
    and then we aggregate it with the original loss.

    .. math::

        \begin{aligned}
            \text{normalized_penalty} ={} & \text{nonlinear}(\frac{\text{penalty}}{\text{baseline}} - 1) \\
            \text{loss} ={} & \text{aggregate}(\text{original_loss}, \text{normalized_penalty})
        \end{aligned}

    where ``penalty`` here is the result returned by the profiler.

    For example, when ``nonlinear`` is ``positive`` and ``aggregate`` is ``add``, the computation formula is:

    .. math::

        \text{loss} = \text{original_loss} + \text{scale} * (max(\frac{\text{penalty}}{\text{baseline}}, 1) - 1, 0)

    Parameters
    ----------
    profiler
        The profiler which is used to profile the sample.
    scale
        The scale of the penalty.
    baseline
        The baseline of the penalty.
    nonlinear
        The nonlinear function to apply to :math:`\frac{\text{penalty}}{\text{baseline}}`.
        The result is called ``normalized_penalty``.
        If ``linear``, then keep the original value.
        If ``positive``, then apply the function :math:`max(0, \cdot)`.
        If ``negative``, then apply the function :math:`min(0, \cdot)`.
        If ``absolute``, then apply the function :math:`abs(\cdot)`.
    aggregate
        The aggregate function to merge the original loss with the penalty.
        If ``add``, then the final loss is :math:`\text{original_loss} + \text{scale} * \text{normalized_penalty}`.
        If ``mul``, then the final loss is :math:`\text{original_loss} * (1 + \text{normalized_penalty})^{\text{scale}}`.
    """

    def __init__(self,
                 profiler: Profiler,
                 baseline: float,
                 scale: float = 1.,
                 *,
                 nonlinear: Literal['linear', 'positive', 'negative', 'absolute'] = 'linear',
                 aggregate: Literal['add', 'mul'] = 'add'):
        super().__init__()
        self.profiler = profiler
        self.scale = scale
        self.baseline = baseline
        self.nonlinear = nonlinear
        self.aggregate = aggregate

    def forward(self, loss: torch.Tensor, sample: Sample) -> tuple[torch.Tensor, dict]:
        profiler_result = self.profile(sample)
        normalized_penalty = self.nonlinear_fn(profiler_result / self.baseline - 1)
        loss_new = self.aggregate_fn(loss, normalized_penalty)

        details = {
            'loss_original': loss,
            'penalty': profiler_result,
            'normalized_penalty': normalized_penalty,
            'loss_final': loss_new,
        }

        return loss_new, details

    def profile(self, sample: Sample) -> float:
        """Subclass overrides this to profile the sample."""
        raise NotImplementedError()

    def aggregate_fn(self, loss: torch.Tensor, normalized_penalty: float) -> torch.Tensor:
        if self.aggregate == 'add':
            return loss + self.scale * normalized_penalty
        if self.aggregate == 'mul':
            return loss * _pow(normalized_penalty + 1, self.scale)
        raise ValueError(f'Invalid aggregate: {self.aggregate}')

    def nonlinear_fn(self, normalized_penalty: float) -> float:
        if self.nonlinear == 'linear':
            return normalized_penalty
        if self.nonlinear == 'positive':
            return _relu(normalized_penalty)
        if self.nonlinear == 'negative':
            return -_relu(-normalized_penalty)
        if self.nonlinear == 'absolute':
            return _abs(normalized_penalty)
        raise ValueError(f'Invalid nonlinear: {self.nonlinear}')


class ExpectationProfilerPenalty(ProfilerPenalty):
    """:class:`ProfilerPenalty` for a sample with distributions.
    Value for each label is a a mapping from chosen value to probablity.
    """

    def profile(self, sample: Sample) -> float:
        """Profile based on a distribution of samples.

        Each value in the sample must be a dict representation a categorical distribution.
        """
        if not isinstance(self.profiler, ExpressionProfiler):
            raise TypeError('DifferentiableProfilerPenalty only supports ExpressionProfiler.')
        for key, value in sample.items():
            if not isinstance(value, dict):
                raise TypeError('Each value must be a dict representation a categorical distribution, '
                                f'but found {type(value)} for key {key}: {value}')
        return expression_expectation(self.profiler.expression, sample)


class SampleProfilerPenalty(ProfilerPenalty):
    """:class:`ProfilerPenalty` for a single sample.
    Value for each label is a specifically chosen value.
    """

    def profile(self, sample: Sample) -> float:
        """Profile based on a single sample."""
        return self.profiler.profile(sample)


# Operators that work for both simple numbers and tensors

def _pow(x: float, y: float) -> float:
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        return cast(float, torch.pow(cast(torch.Tensor, x), y))
    else:
        return np.power(x, y)


def _abs(x: float) -> float:
    if isinstance(x, torch.Tensor):
        return cast(float, torch.abs(x))
    else:
        return np.abs(x)


def _relu(x: float) -> float:
    if isinstance(x, torch.Tensor):
        return cast(float, nn.functional.relu(x))
    else:
        return np.maximum(x, 0)
