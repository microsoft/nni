# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Several planned features that are under discussion / not implemented yet."""

from __future__ import annotations

from .annotation import MutableAnnotation
from .mutable import LabeledMutable, MutableSymbol, Categorical, Numerical


def randint(label: str, lower: int, upper: int) -> Categorical[int]:
    """Choosing a random integer between lower (inclusive) and upper (exclusive).

    Currently it is translated to a :func:`choice`.
    This behavior might change in future releases.

    Examples
    --------
    >>> nni.randint('x', 1, 5)
    Categorical([1, 2, 3, 4], label='x')
    """
    return RandomInteger(lower, upper, label=label)


def lognormal(label: str, mu: float, sigma: float) -> Numerical:
    """Log-normal (in the context of NNI) is defined as the exponential transformation of a normal random variable,
    with mean ``mu`` and deviation ``sigma``. That is::

        exp(normal(mu, sigma))

    In another word, the logarithm of the return value is normally distributed.

    Examples
    --------
    >>> nni.lognormal('x', 4., 2.)
    Numerical(-inf, inf, mu=4.0, sigma=2.0, log_distributed=True, label='x')
    >>> nni.lognormal('x', 0., 1.).random()
    2.3308575497749584
    >>> np.log(x) for x in nni.lognormal('x', 4., 2.).grid(granularity=2)]
    [2.6510204996078364, 4.0, 5.348979500392163]
    """
    return Numerical(mu=mu, sigma=sigma, log_distributed=True, label=label)


def qlognormal(label: str, mu: float, sigma: float, quantize: float) -> Numerical:
    """A combination of :func:`qnormal` and :func:`lognormal`.

    Similar to :func:`qloguniform`, the quantize is done **after** the sample is drawn from the log-normal distribution.

    Examples
    --------
    >>> nni.qlognormal('x', 4., 2., 1.)
    Numerical(-inf, inf, mu=4.0, sigma=2.0, q=1.0, log_distributed=True, label='x')
    """
    return Numerical(mu=mu, sigma=sigma, log_distributed=True, quantize=quantize, label=label)


class Permutation(MutableSymbol):
    """Get a permutation of several values.
    Not implemented. Kept as a placeholder.
    """


class RandomInteger(Categorical[int]):
    """Sample from a list of consecutive integers.
    Kept as a placeholder.

    :class:`Categorical` is a more general version of this class,
    but this class gives better semantics,
    and is consistent with the old ``randint``.
    """
    def __init__(self, lower: int, upper: int, label: str | None = None) -> None:
        if not isinstance(lower, int) or not isinstance(upper, int):
            raise TypeError('lower and upper must be integers.')
        if lower >= upper:
            raise ValueError('lower must be strictly smaller than upper.')
        super().__init__(list(range(lower, upper)), label=label)


class NonNegativeRandomInteger(RandomInteger):
    """Sample from a list of consecutive natural integers, counting from 0.
    Kept as a placeholder.

    :class:`Categorical` and :class:`RandomInteger`
    can be simplified to this class for simpler processing.
    """
    pass


class UnitUniform(Numerical):
    """Sample from a uniform distribution in [0, 1).
    Not implemented yet.

    :class:`Numerical` can be simplified to this class for simpler processing.
    """

    def __init__(self, *, label: str | None = None) -> None:
        super().__init__(low=0.0, high=1.0, label=label)


class JointDistribution(MutableAnnotation):
    """Mutual-correlated distribution among multiple variables.
    Not implemented yet.
    """
    pass


class Graph(LabeledMutable):
    """Graph structure.
    Not implemented yet.
    """
    pass


class GraphAnnotation(MutableAnnotation):
    """When a graph is broken down into simple :class:`MutableSymbol` and :class:`MutableAnnotation`.
    Not implemented yet."""
    pass
