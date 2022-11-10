# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = [
    'Mutable', 'LabeledMutable', 'MutableSymbol', 'MutableExpression', 'Sample',
    # 'Discrete', 'DiscreteMultiple', 'Continuous',
]

import copy
import itertools
import logging
from typing import TypeVar, Type, Any, Generic, Dict, Iterable, Callable, List, TYPE_CHECKING, cast

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm, lognorm, uniform, loguniform

from .exception import SampleValidationError, SampleMissingError
from .symbol import SymbolicExpression, Symbol
from .utils import auto_label

if TYPE_CHECKING:
    from scipy.stats import _distn_infrastructure

Sample = Dict[str, Any]
Choice = TypeVar('Choice')

MISSING = '__missing__'

T = TypeVar('T')

_logger = logging.getLogger(__name__)


def _is_mutable_symbol(mutable: Mutable) -> bool:
    """Check if a mutable is a mutable symbol.

    The default implementation of ``is_leaf``.
    """
    return isinstance(mutable, LabeledMutable)


def dedup_labeled_mutables(mutables: Iterable[LabeledMutable]) -> dict[str, LabeledMutable]:
    """Deduplicate mutables based on labels, and reform a dict.

    Mutables are considered equal if they have the same label.
    We will also check whether mutables of the same label are equal, and raise an error if they are not.

    Parameters
    ----------
    mutables
        The mutables to deduplicate.

    Returns
    -------
    A dict. Keys are labels, values are mutables.
    """
    rv = {}
    for mutable in mutables:
        if not hasattr(mutable, 'label'):
            raise ValueError('Mutable %s does not have a label' % mutable)
        if mutable.label not in rv:
            rv[mutable.label] = mutable
        else:
            if not mutable.equals(rv[mutable.label]):
                raise ValueError(
                    f'Mutables have the same label {mutable.label} but are different: {mutable} and {rv[mutable.label]}'
                )
    return rv


class Mutable:
    """
    Mutable is the base class for every class representing a search space.

    To make a smooth experience of writing new search spaces,
    we provide multiple kinds of mutable subclasses.
    There are basically two types of needs:

    1. Express one variable (aka. one dimension / parameter) of the search space.
       We provide the expressiveness to describe the domain on this dimension.
    2. Composition of multiple dimensions. For example,
       A new variable that is the sum of two existing variables.
       Or a PyTorch module (in the NAS scenario) that relies on one or several variables.

    In most cases, spaces are type 2, because it's relatively straightforward to programmers,
    and easy to be put into a evaluation process. For example, when a model is to search,
    directly programming on the deep learning model would be the most straightforward way
    to define the space.

    On the other hand, most algorithms only care about the
    underlying variables that constitutes the space, rather than the complex compositions.
    That is, the basic dimensions of discrete / continuous values in the space.
    Note that, this is only algorithm-friendly, but not friendly to those who writes the space.

    We provide two methods to achieve the best both worlds.
    :meth:`simplify` is the method to get the basic dimensions (type 1).
    Algorithms then use the simplified space to run the search, generate the samples
    (which are also in the format of the simplified space), and then,
    :meth:`freeze` is the method to get the frozen version of the space with the sample.

    For example::

        >>> from nni.mutable import Discrete
        >>> mutable = Discrete([2, 3]) + Discrete([5, 7])
        >>> mutable
        Discrete([2, 3], label='global_1') + Discrete([5, 7], label='global_2')
        >>> mutable.simplify()
        {'global_1': Discrete([2, 3], label='global_1'), 'global_2': Discrete([5, 7], label='global_2')}
        >>> sample = {'global_1': 2, 'global_2': 7}
        >>> mutable.freeze(sample)
        9

    In the example above, we create a new mutable that is the sum of
    two existing variables (with :class:`MutableExpression`),
    and then simplify it to get the basic dimensions.
    The *sample* here is a dictionary of parameters.
    It should have the exactly same keys as the simplified space,
    and values are replaced with the sampled values.
    The sample can be used in both :meth:`contains` and :meth:`freeze`.

    * Use ``if mutable.contains(sample)`` to check whether a sample is valid.
    * Use ``mutable.freeze(sample)`` to create a fixed version of the mutable.

    One final note, :class:`Mutable` is designed to be framework agnostic.
    It doesn't have any dependency on deep learning frameworks like PyTorch.
    """

    def freeze(self, sample: Sample) -> Any:
        """Create a *frozen* (i.e., fixed) version of this mutable,
        based on sample in the format of :meth:`simplify`.

        For example, the frozen version of an integer variable is a constant.
        The frozen version of a mathematical expression is an evaluated value.
        The frozen version of a layer choice is a fixed layer.

        Parameters
        ----------
        sample
            The sample should be a dict, having the same keys as :meth:`simplify`.
            The values of the dict are the choice of the corresponding mutable,
            whose format varies depending on the specific mutable format.

        Returns
        -------
        The frozen version of this mutable.

        See Also
        --------
        LabeledMutable
        """
        raise NotImplementedError()

    def contains(self, sample: Sample) -> SampleValidationError | None:
        """Check whether sample is validly sampled from the mutable space.

        Parameters
        ----------
        sample
            See :meth:`freeze`.

        Returns
        -------
        Optionally a :exc:`~nni.mutable.exception.SampleValidationError` if the sample is invalid.
        """
        raise NotImplementedError()

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        """Return all the leaf mutables.

        The mutables could contain duplicates (duplicate instances / duplicate labels).
        All leaf mutables should be labeled for the purpose of deduplication in :meth:`simplify`.

        Subclass override this (and possibly call :meth:`leaf_mutables` of sub-mutables).

        Parameters
        ----------
        is_leaf
            A function that takes a mutable and returns whether it's a leaf mutable.

        Returns
        -------
        An iterable of leaf mutables.
        """
        raise NotImplementedError()

    def simplify(self, is_leaf: Callable[[Mutable], bool] | None = None) -> dict[str, LabeledMutable]:
        """Summarize all underlying uncertainties in a schema, useful for search algorithms.

        The default behavior of :meth:`simplify` is to call :meth:`leaf_mutables`
        to retrieve a list of mutables, and deduplicate them based on labels.
        Thus, subclasses only need to override :meth:`leaf_mutables`.

        Parameters
        ----------
        is_leaf
            A function to check whether a mutable is a leaf mutable.
            If not specified, :class:`MutableSymbol` instances will be treated as leaf mutables.
            ``is_leaf`` is useful for algorithms to decide whether to,
            (i) expand some mutables so that less mutable types need to be worried about,
            or (ii) collapse some mutables so that more information could be kept.

        Returns
        -------
        The keys are labels, and values are corresponding labeled mutables.

        Notes
        -----
        Ideally :meth:`simplify` should be idempotent. That being said,
        you can wrap the simplified results with a MutableDict and call simplify again,
        it will get you the same results.
        However, in practice, the order of dict keys might not be guaranteed.

        See Also
        --------
        LabeledMutable
        """
        if is_leaf is None:
            is_leaf = _is_mutable_symbol
        return dedup_labeled_mutables(self.leaf_mutables(is_leaf))

    def validate(self, sample: Sample) -> None:
        """Validate a sample.
        Calls :meth:`contains` and raises an exception if the sample is invalid.

        Parameters
        ----------
        sample
            See :meth:`freeze`.

        Raises
        ------
        nni.mutable.exception.SampleValidationError
            If the sample is invalid.

        Returns
        -------
        None
        """
        exception = self.contains(sample)
        if exception is not None:
            raise exception

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        """Return a string representation of the extra information."""
        return ''

    def equals(self, other: Any) -> bool:
        """Compare two mutables.

        Please use :meth:`equals` to compare two mutables,
        instead of ``==``, because ``==`` will generate mutable expressions.
        """
        return self.__class__ == other.__class__ and self.__dict__ == other.__dict__

    def default(self, memo: Sample | None = None) -> Any:
        """Return the default value of the mutable.
        Useful for debugging and sanity check.
        The returned value should be one of the possible results of :meth:`freeze`.

        The default implementation of :meth:`default` is to call :meth:`default`
        on each of the simplified values and then freeze the result.

        Parameters
        ----------
        memo
            A dict of mutable labels and their default values.
            Use this to share the sampled value among mutables with the same label.
        """
        sample: Sample = {} if memo is None else memo
        for mutable in self.simplify().values():
            mutable.default(sample)
        return self.freeze(sample)

    def robust_default(self, memo: Sample | None = None, retries: int = 1000) -> Any:
        """Return the default value of the mutable.
        Will retry with :meth:`random` in case of failure.

        It's equivalent to the following pseudo-code::

            for attempt in range(retries + 1):
                try:
                    if attempt == 0:
                        return self.default()
                    else:
                        return self.random()
                except SampleValidationError:
                    pass

        Parameters
        ----------
        memo
            A dict of mutable labels and their default values.
            Use this to share the sampled value among mutables with the same label.
        retries
            If the default sample is not valid, we will retry to invoke
            :meth:`random` for ``retries`` times, until a valid sample is found.
            Otherwise, an exception will be raised, complaining that no valid sample is found.
        """
        sample: Sample = {} if memo is None else memo
        for attempt in range(retries + 1):
            try:
                sample_copy = copy.copy(sample)
                if attempt == 0:
                    rv = self.default(sample_copy)
                else:
                    rv = self.random(sample_copy)
                sample.update(sample_copy)
                return rv
            except SampleValidationError:
                if attempt == retries:
                    raise ValueError(
                        f'Cannot find a valid default sample after {retries} retries, for {self}. '
                        'Please either set `default_value` manually, or loosen the constraints.')

        raise RuntimeError('This should not happen.')

    def random(self, memo: Sample | None = None, random_state: RandomState | None = None) -> Any:
        """Randomly sample a value of the mutable. Used in random strategy.
        The returned value should be one of the possible results of :meth:`freeze`.

        The default implementation of :meth:`random` is to call :meth:`random`
        on each of the simplified values and then freeze the result.

        It's possible that :meth:`random` raises :exc:`~nni.mutable.exception.SampleValidationError`,
        e.g., in cases when constraints are violated.

        Parameters
        ----------
        memo
            A dict of mutable labels and their random values.
            Use this to share the sampled value among mutables with the same label.
        """
        sample: Sample = {} if memo is None else memo
        if random_state is None:
            random_state = RandomState()
        for mutable in self.simplify().values():
            mutable.random(sample, random_state)
        return self.freeze(sample)

    def grid(self, memo: Sample | None = None, granularity: int | None = None) -> Iterable[Any]:
        """Return a grid of sample points
        that can be possibly sampled from the mutable. Used in grid search strategy.
        It should return all the possible results of :meth:`freeze`.

        The default implementation of :meth:`grid` is to call iterate over
        the product of all the simplified grid values.
        Specifically, the grid will be iterated over in a depth-first-search order.

        Parameters
        ----------
        memo
            A dict of mutable labels and their values in the current grid point.
            Use this to share the sampled value among mutables with the same label.
        granularity
            Optional integer to specify the level of granularity of the grid.
            This only affects the cases where the grid is not a finite set.
            See :class:`Continuous` for details.
        """
        def _iter(index: int) -> Iterable[Any]:
            if index == len(simplified):
                yield self.freeze(sample)
            else:
                for _ in simplified[index].grid(sample, granularity):
                    yield from _iter(index + 1)

        # No deduplication here as it will be done in the grid of simplified mutables.
        simplified: list[LabeledMutable] = list(self.simplify().values())

        # Same sample is used throughout the whole process.
        sample: Sample = {} if memo is None else memo

        yield from _iter(0)


class LabeledMutable(Mutable):
    """:class:`Mutable` with a label. This should be the super-class of most mutables.
    The labels are widely used in simplified result, as well as samples.
    Usually a mutable must be firstly converted into one or several :class:`LabeledMutable`,
    before strategy can recognize and process it.

    When two mutables have the same label, they semantically share the same choice.
    That means, the choices of the two mutables will be shared.

    The labeled mutables are by default created with one reproducible, auto-generated label.
    But it can also contain multiple labels if the mutable is more complex.
    The label can also be specified by users, or sub-classes.

    Being a :class:`LabeledMutable` doesn't necessarily mean that it is a leaf mutable.
    Some :class:`LabeledMutable` can be further simplified into multiple leaf mutables.
    """

    label: str

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        if is_leaf(self):
            yield self
        else:
            raise ValueError(f'is_leaf() should return True for this type of mutable: {type(self)}')

    def default(self, memo: Sample | None = None) -> Any:
        raise NotImplementedError(f'default() is not implemented for {self.__class__}')

    def random(self, memo: Sample | None = None, random_state: RandomState | None = None) -> Any:
        raise NotImplementedError(f'random() is not implemented in {self.__class__}.')

    def grid(self, memo: Sample | None = None, granularity: int | None = None) -> Iterable[Any]:
        raise NotImplementedError(f'grid() is not implemented in {self.__class__}.')


class MutableExpression(Mutable, SymbolicExpression):
    """
    Expression of mutables. Common use cases include:
    summation of several mutables, binary comparison between two mutables.

    The expression is defined by a operator and a list of operands,
    which must be one or several :class:`MutableSymbol` or :class:`MutableExpression`.

    The expression can be simplified into a dict of :class:`LabeledMutable`.
    It can also be evaluated to be a concrete value (via :meth:`~Mutable.freeze`),
    when the values it depends on have been given.

    See Also
    --------
    nni.mutable.symbol.SymbolicExpression
    """

    @property
    def expr_cls(self) -> Type[MutableExpression]:
        return MutableExpression

    def freeze(self, sample: Sample) -> Any:
        self.validate(sample)
        return self.evaluate(sample)

    def contains(self, sample: Sample) -> SampleValidationError | None:
        for symbol in self.leaf_symbols():
            if not isinstance(symbol, MutableSymbol):
                _logger.warning('The expression contains non-mutable symbols. This is not recommended: %s', self)
                break
            exception = symbol.contains(sample)
            if exception is not None:
                exception.paths.insert(0, 'expression')
                return exception
        return None

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        for symbol in self.leaf_symbols():
            if not isinstance(symbol, MutableSymbol):
                _logger.warning('The expression contains non-mutable symbols. This is not recommended: %s', self)
                break
            yield from symbol.leaf_mutables(is_leaf)

    def __repr__(self) -> str:
        return self.symbolic_repr()


class MutableSymbol(LabeledMutable, Symbol, MutableExpression):
    """:class:`MutableSymbol` corresponds to the concept of
    a variable / hyper-parameter / dimension.

    For example, a learning rate with a uniform distribution between 0.1 and 1,
    or a convolution filter that is either 32 or 64.

    :class:`MutableSymbol` is a subclass of :class:`Symbol`.
    Therefore they support arithmetic operations.
    The operation results will be a :class:`MutableExpression` object.

    See Also
    --------
    nni.mutable.symbol.Symbol
    """

    # MutableSymbol share the ``__init__`` with Symbol.

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'
