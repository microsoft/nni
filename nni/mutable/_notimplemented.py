# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: skip-file
# type: ignore

"""Several planned features that are under discussion / not implemented yet."""

from __future__ import annotations

from .annotation import MutableAnnotation
from .mutable import LabeledMutable, MutableSymbol, Discrete, Continuous


class Permutation(MutableSymbol):
    """Get a permutation of several values.
    Not implemented. Kept as a placeholder.
    """


class RandomInteger(Discrete[int]):
    """Sample from a list of consecutive integers.
    Kept as a placeholder.

    :class:`Discrete` is a more general version of this class,
    but this class gives better semantics,
    and is consistent with the old ``randint``.
    """
    pass


class NonNegativeRandomInteger(RandomInteger):
    """Sample from a list of consecutive natural integers, counting from 0.
    Kept as a placeholder.

    :class:`Discrete` and :class:`RandomInteger`
    can be simplified to this class for simpler processing.
    """
    pass


class UnitUniform(Continuous):
    """Sample from a uniform distribution in [0, 1).
    Not implemented yet.

    :class:`Continuous` can be simplified to this class for simpler processing.
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
