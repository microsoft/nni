# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = [
    'MutableAnnotation', 'Constraint', 'ExpressionConstraint',
]

import logging
from typing import Iterable, Callable

from numpy.random import RandomState

from .exception import ConstraintViolation, SampleValidationError
from .mutable import LabeledMutable, Mutable, Sample, MutableExpression
from .utils import auto_label

_logger = logging.getLogger(__name__)


class MutableAnnotation(LabeledMutable):
    """Provide extra annotation / hints for a search space.

    Sometimes, people who authored search strategies might want to recognize some hints from the search space.
    For example,

    - Adding some extra constraints between two parameters in the space.
    - Marking some choices as "nodes" in a cell.
    - Some parameter-combinations should be avoided or explored at the very first.

    :class:`MutableAnnotation` is defined to be *transparent*,
    i.e., it doesn't generate any new dimension by itself,
    and thus typically doesn't introduce a new key in the sample received by :meth:`nni.mutable.Mutable.freeze`,
    but it affects the sampling the freezing process implicitly.

    This class is useful for isinstance check.
    """


class Constraint(MutableAnnotation):
    """
    Constraints put extra requirements to make one sample valid.

    For example, a constraint can be used to express that a variable should be larger than another variable,
    or certain combinations of variables should be strictly avoided.

    :class:`Constraint` is a subclass of :class:`MutableAnnotation`, and thus can be used as a normal mutable.
    It has a special :meth:`contains` method, which is used to check whether a sample satisfies the constraint.
    A constraint is satisfied if and only if :meth:`contains` returns ``None``.

    In general, users should inherit from :class:`Constraint` to implement customized constraints.
    :class:`ExpressionConstraint` is a special constraint
    that can be used to express constraints in a more concise way.

    See Also
    --------
    ExpressionConstraint
    """

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        """Override this to implement customized constraint.
        It should return a list of leaf mutables that are used in the constraint.

        See Also
        --------
        nni.mutable.Mutable.leaf_mutables
        """
        return super().leaf_mutables(is_leaf)

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        """Override this to implement customized constraint.
        It should return ``None`` if the sample satisfies the constraint.
        Otherwise return a :exc:`~nni.mutable.exception.ConstraintViolation` exception.

        See Also
        --------
        nni.mutable.Mutable.check_contains
        """
        raise NotImplementedError()

    def freeze(self, sample: Sample) -> None:
        """Validate the sample (via ``validate()``) and returns None.

        See Also
        --------
        nni.mutable.Mutable.freeze
        nni.mutable.Mutable.validate
        """
        self.validate(sample)
        return None

    def default(self, memo: Sample | None = None) -> None:
        memo = {} if memo is None else memo
        for mutable in self.simplify().values():
            if mutable is not self:
                mutable.default(memo)
        # Exception could raise here.
        # Use `robust_default()` if the exception is expected and `random()` should be retried.
        return self.freeze(memo)

    def random(self, memo: Sample | None = None, random_state: RandomState | None = None) -> None:
        memo = {} if memo is None else memo
        for mutable in self.simplify().values():
            if mutable is not self:
                mutable.random(memo, random_state)
        # Exception could also raise here.
        return self.freeze(memo)

    def grid(self, memo: Sample | None = None, granularity: int | None = None) -> Iterable[None]:
        """Yield all samples that satisfy the constraint.

        If some samples the constraint relies on have not been frozen yet,
        it will be sampled here and put into the memo.
        After that, it checks whether the sample satisfies the constraint after sampling (via ``contains()``).
        If the sample doesn't satisfy the constraint, it will be discarded.

        Each yielded sample of the :meth:`Constraint.grid` itself is None,
        because :meth:`Constraint.freeze` also returns None.
        """
        memo = {} if memo is None else memo
        mutables_wo_self = [mutable for mutable in self.simplify().values() if mutable is not self]
        from .container import MutableList
        for _ in MutableList(mutables_wo_self).grid(memo, granularity):
            if self.contains(memo):
                yield self.freeze(memo)
            else:
                _logger.debug('Constraint violation detected. Skip this grid point: %s', memo)


class ExpressionConstraint(Constraint):
    """A constraint that is expressed as :class:`~nni.mutable.MutableExpression`.

    The expression must evaluate to be true to satisfy the constraint.

    Parameters
    ----------
    expression
        A :class:`~nni.mutable.MutableExpression` that evaluates to a boolean value.
    label
        The semantic name of the constraint.

    Examples
    --------
    >>> a = Categorical([1, 3])
    >>> b = Categorical([2, 4])
    >>> list(MutableList([a, b, ExpressionConstraint(a + b == 5)]).grid())
    [[1, 4, None], [3, 2, None]]
    """

    def __init__(self, expression: MutableExpression, *, label: str | None = None) -> None:
        self.label = auto_label(label)
        self.expression = expression

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        yield from self.expression.leaf_mutables(is_leaf)
        yield self

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        exception = self.expression.check_contains(sample)
        if exception is not None:
            return exception
        expr_val = self.expression.freeze(sample)
        if not expr_val:
            return ConstraintViolation(f'{self.expression} is not satisfied.')
        return None

    def extra_repr(self) -> str:
        return f'{self.expression!r}, label={self.label!r}'
