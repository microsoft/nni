# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""High-level API for mutables."""

from __future__ import annotations

__all__ = [
    'choice', 'uniform', 'quniform', 'loguniform', 'qloguniform',
    'normal', 'qnormal',
]

import logging
from typing import TYPE_CHECKING, TypeVar, overload, List, cast

from .mutable import Categorical, Numerical

if TYPE_CHECKING:
    from torch.nn import Module
    from nni.nas.nn.pytorch import LayerChoice

T = TypeVar('T')

_logger = logging.getLogger(__name__)


@overload
def choice(label: str, choices: list[T]) -> Categorical[T]:
    ...


@overload
def choice(label: str, choices: list[Module]) -> LayerChoice:
    ...


def choice(label: str, choices: list[T] | list[Module]) -> Categorical[T] | LayerChoice:
    """Choose from a list of options.

    By default, it will create a :class:`~nni.mutable.Categorical` object.
    ``choices`` should be a list of numbers or a list of strings.
    Using arbitrary objects as members of this list (like sublists, a mixture of numbers and strings, or null values)
    should work in most cases, but may trigger undefined behaviors.
    If PyTorch modules are presented in the choices, it will create a :class:`~nni.nas.nn.pytorch.LayerChoice`.

    For most search algorithms, choice are non-ordinal.
    Even if the choices are numbers, they will still be treated as individual options,
    and their numeric values will be neglected.

    Nested choices (i.e., choice inside one of the options) is not currently supported by this API.

    Examples
    --------
    >>> nni.choice('x', [1, 2, 3])
    Categorical([1, 2, 3], label='x')
    >>> nni.choice('conv', [nn.Conv2d(3, 3, 3), nn.Conv2d(3, 3, 5)])
    LayerChoice(
        label='conv'
        (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
        (1): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1))
    )
    """
    try:
        from torch.nn import Module
        if all(isinstance(c, Module) for c in choices):
            from nni.nas.nn.pytorch import LayerChoice
            return LayerChoice(cast(List[Module], choices), label=label)

        from torch import Tensor
        if any(isinstance(c, Tensor) for c in choices):
            raise TypeError(
                'Please do not use choice to choose from tensors. '
                'If you are using this in forward, please use `InputChoice` explicitly in `__init__` instead.')
    except ImportError:
        # In case PyTorch is not installed.
        pass

    return Categorical(cast(List[T], choices), label=label)


def uniform(label: str, low: float, high: float) -> Numerical:
    """Uniformly sampled between low and high.
    When optimizing, this variable is constrained to a two-sided interval.

    Examples
    --------
    >>> nni.uniform('x', 0, 1)
    Numerical(0, 1, label='x')
    """
    if low >= high:
        raise ValueError('low must be strictly smaller than high.')
    return Numerical(low, high, label=label)


def quniform(label: str, low: float, high: float, quantize: float) -> Numerical:
    """Sampling from ``uniform(low, high)`` but the final value is
    determined using ``clip(round(uniform(low, high) / q) * q, low, high)``,
    where the clip operation is used to constrain the generated value within the bounds.

    For example, for low, high, quantize being specified as 0, 10, 2.5 respectively,
    possible values are [0, 2.5, 5.0, 7.5, 10.0].
    For 2, 10, 5, possible values are [2., 5., 10.].

    Suitable for a discrete value with respect to which the objective is still somewhat “smooth”,
    but which should be bounded both above and below.
    Note that the return values will always be float.
    If you want to uniformly choose an **integer** from a range [low, high],
    you can use::

        nni.quniform(low - 0.5, high + 0.5, 1).int()

    Examples
    --------
    >>> nni.quniform('x', 2.5, 5.5, 2.)
    Numerical(2.5, 5.5, q=2.0, label='x')
    """

    if isinstance(quantize, int):
        _logger.warning('Though quantize is an integer (%d) in quniform, the returned value will always be float. '
                        'Use `.int()` to convert to integer.', quantize)
    if low >= high:
        raise ValueError('low must be strictly smaller than high.')

    return Numerical(low, high, quantize=quantize, label=label)


def loguniform(label: str, low: float, high: float) -> Numerical:
    """Draw from a range [low, high] according to a loguniform distribution::

        exp(uniform(log(low), log(high))),

    so that the logarithm of the return value is uniformly distributed.

    Since logarithm is taken here, low and high must be strictly greater than 0.

    This is often used in variables which are log-distributed in experience,
    such as learning rate (which we often choose from 1e-1, 1e-3, 1e-6...).

    Examples
    --------
    >>> nni.loguniform('x', 1e-5, 1e-3)
    Numerical(1e-05, 0.001, log_distributed=True, label='x')
    >>> list(nni.loguniform('x', 1e-5, 1e-3).grid(granularity=2))
    [3.1622776601683795e-05, 0.0001, 0.00031622776601683794]
    """
    if low >= high:
        raise ValueError('low must be strictly smaller than high.')
    if low <= 0 or high <= 0:
        raise ValueError('low and high must be strictly greater than 0.')
    return Numerical(low, high, log_distributed=True, label=label)


def qloguniform(label: str, low: float, high: float, quantize: float) -> Numerical:
    """A combination of :func:`quniform` and :func:`loguniform`.

    Note that the quantize is done **after** the sample is drawn from the log-uniform distribution.

    Examples
    --------
    >>> nni.qloguniform('x', 1e-5, 1e-3, 1e-4)
    Numerical(1e-05, 0.001, q=0.0001, log_distributed=True, label='x')
    """
    return Numerical(low, high, log_distributed=True, quantize=quantize, label=label)


def normal(label: str, mu: float, sigma: float) -> Numerical:
    """Declare a normal distribution with mean ``mu`` and standard deviation ``sigma``.

    The variable is unbounded, meaning that any real number from ``-inf`` to ``+inf`` can be possibly sampled.

    Examples
    --------
    >>> nni.normal('x', 0, 1)
    Numerical(-inf, inf, mu=0, sigma=1, label='x')
    >>> nni.normal('x', 0, 1).random()
    -0.30621273862239057
    """
    if sigma <= 0:
        raise ValueError('Standard deviation must be strictly greater than 0.')

    return Numerical(mu=mu, sigma=sigma, label=label)


def qnormal(label: str, mu: float, sigma: float, quantize: float) -> Numerical:
    """Similar to :func:`quniform`, except the uniform distribution is replaced with a normal distribution.

    Examples
    --------
    >>> nni.qnormal('x', 0., 1., 0.1)
    Numerical(-inf, inf, mu=0.0, sigma=1.0, q=0.1, label='x')
    >>> nni.qnormal('x', 0., 1., 0.1).random()
    -0.1
    """
    return Numerical(mu=mu, sigma=sigma, quantize=quantize, label=label)
