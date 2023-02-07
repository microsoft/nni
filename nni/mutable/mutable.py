# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = [
    'Mutable', 'LabeledMutable', 'MutableSymbol', 'MutableExpression', 'Sample',
    'Categorical', 'CategoricalMultiple', 'Numerical',
]

import copy
import itertools
import logging
from collections.abc import Sequence, Mapping, Set
from typing import TypeVar, Type, Any, Generic, Dict, Iterable, Callable, List, TYPE_CHECKING, cast

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm, lognorm, uniform, loguniform

from .exception import SampleValidationError, SampleMissingError
from .symbol import SymbolicExpression, Symbol
from .utils import auto_label, label_scope, label

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


def _mutable_equal(mutable1: Any, mutable2: Any) -> bool:
    """Check if two mutables are equal with :meth:`Mutable.equals`.

    Use this instead of ``==`` when comparing objects that could contain mutables.

    Parameters
    ----------
    mutable1
        The first mutable.
    mutable2
        The second mutable.

    Returns
    -------
    True if the two mutables are equal, False otherwise.
    """
    if isinstance(mutable1, Mutable):
        if isinstance(mutable2, Mutable):
            return mutable1.equals(mutable2)
        return False
    if isinstance(mutable2, Mutable):
        # mutable1 is not a Mutable, but mutable2 is.
        return False
    # Both are not Mutable.

    # Dealing with mapping, sequence and set manually,
    # because their ``==`` will invoke ``__eq__`` of their elements,
    # but we want to invoke ``Mutable.equals`` instead.
    if isinstance(mutable1, Mapping) and isinstance(mutable2, Mapping):
        if len(mutable1) != len(mutable2):
            return False
        for key in mutable1:
            if key not in mutable2:
                return False
            if not _mutable_equal(mutable1[key], mutable2[key]):
                return False
        return True
    if isinstance(mutable1, Sequence) and isinstance(mutable2, Sequence):
        if isinstance(mutable1, (str, label, label_scope)):
            return mutable1 == mutable2  # exclude strings to avoid infinite recursion
        if len(mutable1) != len(mutable2):
            return False
        for item1, item2 in zip(mutable1, mutable2):
            if not _mutable_equal(item1, item2):
                return False
        return True
    if isinstance(mutable1, Set):
        if not isinstance(mutable2, Set):
            return False
        if len(mutable1) != len(mutable2):
            return False
        for item1 in mutable1:
            for item2 in mutable2:
                if _mutable_equal(item1, item2):
                    break
            else:
                return False
        return True
    if isinstance(mutable1, np.ndarray):
        if not isinstance(mutable2, np.ndarray):
            return False
        return np.array_equal(mutable1, mutable2)

    return mutable1 == mutable2


def _dedup_labeled_mutables(mutables: Iterable[LabeledMutable]) -> dict[str, LabeledMutable]:
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
    That is, the basic dimensions of categorical / continuous values in the space.
    Note that, this is only algorithm-friendly, but not friendly to those who writes the space.

    We provide two methods to achieve the best both worlds.
    :meth:`simplify` is the method to get the basic dimensions (type 1).
    Algorithms then use the simplified space to run the search, generate the samples
    (which are also in the format of the simplified space), and then,
    :meth:`freeze` is the method to get the frozen version of the space with the sample.

    For example::

        >>> from nni.mutable import Categorical
        >>> mutable = Categorical([2, 3]) + Categorical([5, 7])
        >>> mutable
        Categorical([2, 3], label='global_1') + Categorical([5, 7], label='global_2')
        >>> mutable.simplify()
        {'global_1': Categorical([2, 3], label='global_1'), 'global_2': Categorical([5, 7], label='global_2')}
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

    Subclasses of mutables must implement :meth:`leaf_mutables` (which is the implementation of :meth:`simplify`),
    :meth:`check_contains`, and :meth:`freeze`.
    Subclasses of :class:`LabeledMutable` must also implement :meth:`default`,
    :meth:`random` and :meth:`grid`.

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

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        """Check whether sample is validly sampled from the mutable space.
        **Return** an exception if the sample is invalid, otherwise **return** ``None``.
        Subclass is recommended to override this rather than :meth:`contains`.

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
        When they are implemented, they could use ``is_leaf`` to check whether a mutable should be expanded,
        and use ``yield`` to return the leaf mutables.

        Parameters
        ----------
        is_leaf
            A function that takes a mutable and returns whether it's a leaf mutable.
            See :meth:`simplify`.

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

        There is also no guarantee that all mutables returned by :meth:`simplify` are leaf mutables
        that will pass the check of ``is_leaf``. There are certain mutables that are not leaf by default,
        but can't be expanded any more (e.g., :class:`~nni.mutable.annotation.MutableAnnotation`).
        As long as they are labeled, they are still valid return values.
        The caller can decide whether to raise an exception or simply ignore them.

        See Also
        --------
        LabeledMutable
        """
        if is_leaf is None:
            is_leaf = _is_mutable_symbol
        return _dedup_labeled_mutables(self.leaf_mutables(is_leaf))

    def contains(self, sample: Sample) -> bool:
        """Check whether sample is validly sampled from the mutable space.

        Parameters
        ----------
        sample
            See :meth:`freeze`.

        Returns
        -------
        Whether the sample is valid.
        """
        return self.check_contains(sample) is None

    def validate(self, sample: Sample) -> None:
        """Validate a sample.
        Calls :meth:`check_contains` and raises an exception if the sample is invalid.

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
        exception = self.check_contains(sample)
        if exception is not None:
            raise exception

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        """Return a string representation of the extra information."""
        return ''

    def as_legacy_dict(self) -> dict:
        """Convert the mutable into the legacy dict representation.

        For example, ``{"_type": "choice", "_value": [1, 2, 3]}`` is the legacy dict representation of
        ``nni.mutable.Categorical([1, 2, 3])``.
        """
        raise NotImplementedError(f'as_legacy_dict is not implemented for this type of mutable: {type(self)}.')

    def equals(self, other: Any) -> bool:
        """Compare two mutables.

        Please use :meth:`equals` to compare two mutables,
        instead of ``==``, because ``==`` will generate mutable expressions.
        """
        return self.__class__ == other.__class__ and _mutable_equal(self.__dict__, other.__dict__)

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
            # Will raise NotImplementedError here if the mutable is leaf but default is not implemented.
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
            # Will raise NotImplementedError here if the mutable is leaf but random is not implemented.
            mutable.random(sample, random_state)
        return self.freeze(sample)

    def grid(self, memo: Sample | None = None, granularity: int | None = None) -> Iterable[Any]:
        """Return a grid of sample points
        that can be possibly sampled from the mutable. Used in grid search strategy.
        It should return all the possible results of :meth:`freeze`.

        The default implementation of :meth:`grid` is to call iterate over
        the product of all the simplified grid values.
        Specifically, the grid will be iterated over in a depth-first-search order.

        The deduplication of :meth:`grid` (even with a certain granularity) is not guaranteed.
        But it will be done at a best-effort level.
        In most cases, results from :meth:`grid` with a lower granularity will be a subset of
        results from :meth:`grid` with a higher granularity.
        The caller should handle the deduplication.

        Parameters
        ----------
        memo
            A dict of mutable labels and their values in the current grid point.
            Use this to share the sampled value among mutables with the same label.
        granularity
            Optional integer to specify the level of granularity of the grid.
            This only affects the cases where the grid is not a finite set.
            See :class:`Numerical` for details.
        """
        def _iter(index: int) -> Iterable[Any]:
            if index == len(simplified):
                yield self.freeze(sample)
            else:
                # Will raise NotImplementedError here if the mutable is leaf but grid is not implemented.
                for _ in simplified[index].grid(sample, granularity):
                    yield from _iter(index + 1)

        # No deduplication here as it will be done in the grid of simplified mutables.
        simplified: list[LabeledMutable] = list(self.simplify().values())

        # Same sample is used throughout the whole process.
        sample: Sample = {} if memo is None else memo

        yield from _iter(0)

    def _unwrap_parameter(self):
        # Used in ``nni.trace``.
        # Calling ``ensure_frozen()`` by default.
        from .frozen import ensure_frozen
        return ensure_frozen(self, strict=False)


class LabeledMutable(Mutable):
    """:class:`Mutable` with a label. This should be the super-class of most mutables.
    The labels are widely used in simplified result, as well as samples.
    Usually a mutable must be firstly converted into one or several :class:`LabeledMutable`,
    before strategy can recognize and process it.

    When two mutables have the same label, they semantically share the same choice.
    That means, the choices of the two mutables will be shared.
    The labels can be either auto-generated, or provided by the user.

    Being a :class:`LabeledMutable` doesn't necessarily mean that it is a leaf mutable.
    Some :class:`LabeledMutable` can be further simplified into multiple leaf mutables.
    In the current implementation, there are basically two kinds of :class:`LabeledMutable`:

    1. :class:`MutableSymbol`. This is usually referred to as a "parameter". They produce a key-value in the sample.
    2. :class:`~nni.mutable.annotation.MutableAnnotation`. They function as some kind of hint,
       and do not generate a key-value in the sample. Sometimes they can also be simplified and
       the :class:`MutableSymbol` they depend on would appear in the simplified result.
    """

    label: str

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        if is_leaf(self):
            # By default, is_leaf is true for MutableSymbol, and false for MutableAnnotation.
            # So MutableAnnotation must implement `is_leaf`, even if it decides to yield itself.
            yield self
        else:
            raise ValueError(f'is_leaf() should return True for this type of mutable: {type(self)}')

    def default(self, memo: Sample | None = None) -> Any:
        raise NotImplementedError(f'default() is not implemented for {self.__class__}')

    def random(self, memo: Sample | None = None, random_state: RandomState | None = None) -> Any:
        raise NotImplementedError(f'random() is not implemented in {self.__class__}.')

    def grid(self, memo: Sample | None = None, granularity: int | None = None) -> Iterable[Any]:
        raise NotImplementedError(f'grid() is not implemented in {self.__class__}.')


class MutableExpression(Mutable, SymbolicExpression, Generic[T]):
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

    def freeze(self, sample: Sample) -> T:
        self.validate(sample)
        return self.evaluate(sample)

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        for symbol in self.leaf_symbols():
            if not isinstance(symbol, MutableSymbol):
                _logger.warning('The expression contains non-mutable symbols. This is not recommended: %s', self)
                break
            exception = symbol.check_contains(sample)
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

    def equals(self, other: MutableExpression) -> bool:
        if type(self) != type(other):
            return False
        return self.function == other.function and _mutable_equal(self.arguments, other.arguments)

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

    def equals(self, other: MutableSymbol) -> bool:
        return Mutable.equals(self, other)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def int(self) -> MutableExpression[int]:
        """Cast the mutable to an integer."""
        return MutableExpression.to_int(self)

    def float(self) -> MutableExpression[float]:
        """Cast the mutable to a float."""
        return MutableExpression.to_float(self)


class Categorical(MutableSymbol, Generic[Choice]):
    """Choosing one from a list of categorical values.

    Parameters
    ----------
    values
        The list of values to choose from.
        There are no restrictions on value types. They can be integers, strings, and even dicts and lists.
        There is no intrinsic ordering of the values, meaning that the order
        in which the values appear in the list doesn't matter.
        The values can also be an iterable, which will be expanded into a list.
    weights
        The probability distribution of the values. Should be an array with the same length as ``values``.
        The sum of the distribution should be 1.
        If not specified, the values will be chosen uniformly.
    default
        Default value of the mutable. If not specified, the first value will be used.
    label
        The label of the mutable. If not specified, a label will be auto-generated.

    Examples
    --------
    >>> x = Categorical([2, 3, 5], label='x1')
    >>> x.simplify()
    {'x1': Categorical([2, 3, 5], label='x1')}
    >>> x.freeze({'x1': 3})
    3
    """

    def __init__(
        self, values: Iterable[Choice], *,
        weights: list[float] | None = None,
        default: Choice | str = MISSING,
        label: str | None = None
    ) -> None:
        values = list(values)
        assert values, 'Categorical values must not be empty.'
        self.label: str = auto_label(label)
        self.values: list[Choice] = values
        self.weights = weights if weights is not None else [1 / len(values)] * len(values)

        if default is not MISSING:
            self.validate({self.label: default})
        self.default_value = default

        assert not(any(isinstance(value, Mutable) for value in values)), 'Discrete values must not contain mutables.'
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                assert values[i] != values[j], f'Discrete values must be unique, but {i} collides with {j}.'
        assert len(self.weights) == len(self.values), 'Distribution must have length n.'
        assert abs(sum(self.weights) - 1) < 1e-6, 'Distribution must sum to 1.'

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        if self.label not in sample:
            return SampleMissingError(self.label, list(sample.keys()))
        sample_val = sample[self.label]
        if sample_val not in self.values:
            return SampleValidationError(f'{sample_val} not found in {self.values}')
        return None

    def extra_repr(self) -> str:
        if len(self.values) <= 7:
            return f'{self.values!r}, label={self.label!r}'
        return '[' +  \
            ', '.join(map(repr, self.values[:3])) + \
            ', ..., ' + \
            ', '.join(map(repr, self.values[-3:])) + \
            f'], label={self.label!r}'

    def freeze(self, sample: Sample) -> Any:
        self.validate(sample)
        return sample[self.label]

    def __len__(self):
        return len(self.values)

    def as_legacy_dict(self) -> dict:
        return {
            '_type': 'choice',
            '_value': self.values,
        }

    def default(self, memo: Sample | None = None) -> Choice:
        """The default() of :class:`Categorical` is the first value unless default value is set.

        See Also
        --------
        Mutable.default
        """
        memo = {} if memo is None else memo
        err = self.check_contains(memo)
        if isinstance(err, SampleMissingError):
            if self.default_value is not MISSING:
                memo[self.label] = self.default_value
            else:
                memo[self.label] = self.values[0]
        rv = self.freeze(memo)
        if self.default_value is not MISSING and rv != self.default_value:
            raise ValueError(f'Default value is specified to be {self.default_value} but got {rv}. '
                             f'Please check the default value of {self.label}.')
        return rv

    def random(self, memo: Sample | None = None, random_state: RandomState | None = None) -> Choice:
        """Randomly sample a value from choices.
        Distribution is respected if provided.

        See Also
        --------
        Mutable.random
        """
        memo = {} if memo is None else memo
        if random_state is None:
            random_state = RandomState()
        err = self.check_contains(memo)
        if isinstance(err, SampleMissingError):
            index = random_state.choice(len(self.values), p=self.weights)
            memo[self.label] = self.values[index]
        return self.freeze(memo)

    def grid(self, memo: Sample | None = None, granularity: int | None = None) -> Iterable[Choice]:
        """Return also values as a grid. Sorted by distribution from most likely to least likely.

        See Also
        --------
        Mutable.grid
        """
        memo = {} if memo is None else memo
        err = self.check_contains(memo)

        if isinstance(err, SampleMissingError):
            if all(dis == self.weights[0] for dis in self.weights):
                # uniform distribution
                values_perm = self.values
            else:
                # More heavily-distributed items are put upfront.
                indices = sorted(range(len(self.values)), key=lambda i: self.weights[i], reverse=True)
                values_perm = [self.values[i] for i in indices]

            for value in values_perm:
                memo[self.label] = value
                yield self.freeze(memo)
            memo.pop(self.label)
        else:
            yield self.freeze(memo)


class CategoricalMultiple(MutableSymbol, Generic[Choice]):
    """Choosing multiple from a list of values without replacement.

    It's implemented with a different class because for most algorithms, it's very different from :class:`Categorical`.

    :class:`CategoricalMultiple` can be either treated as a atomic :class:`LabeledMutable` (i.e., *simple format*),
    or be further simplified into a series of more fine-grained mutables (i.e., *categorical format*).

    In *categorical format*, class:`CategoricalMultiple` can be broken down to a list of :class:`Categorical` of true and false,
    each indicating whether the choice on the corresponding position should be chosen.
    A constraint will be added if ``n_chosen`` is not None.
    This is useful for some algorithms that only support categorical mutables.
    Note that the prior distribution will be lost in this process.

    Parameters
    ----------
    values
        The list of values to choose from. See :class:`Categorical`.
    n_chosen
        The number of values to choose. If not specified, any number of values can be chosen.
    weights
        The probability distribution of the values. Should be an array with the same length as ``values``.
        When ``n_chosen`` is None, it's the probability that each candidate is chosen.
        When ``n_chosen`` is not None, the distribution should sum to 1.
    default
        Default value of the mutable. If not specified, the first ``n_chosen`` value will be used.
    label
        The label of the mutable. If not specified, a label will be auto-generated.

    Examples
    --------
    >>> x = CategoricalMultiple([2, 3, 5, 7], n_chosen=2, label='x2')
    >>> x.random()
    [2, 7]
    >>> x.simplify()
    {'x2': CategoricalMultiple([2, 3, 5, 7], n_chosen=2, label='x2')}
    >>> x.simplify(lambda t: not isinstance(t, CategoricalMultiple))
    {
        'x2/0': Categorical([True, False], label='x2/0'),
        'x2/1': Categorical([True, False], label='x2/1'),
        'x2/2': Categorical([True, False], label='x2/2'),
        'x2/3': Categorical([True, False], label='x2/3'),
        'x2/n': ExpressionConstraint(...)
    }
    >>> x.freeze({'x2': [3, 5]})
    [3, 5]
    >>> x.freeze({'x2/0': True, 'x2/1': False, 'x2/2': True, 'x2/3': False})
    [2, 5]
    """

    def __init__(
        self, values: Iterable[Choice], *,
        n_chosen: int | None = None,
        weights: list[float] | None = None,
        default: list[Choice] | str = MISSING,
        label: str | None = None,
    ) -> None:
        values = list(values)
        assert values, 'Discrete values must not be empty.'
        with label_scope(label) as self.label_scope:
            self.label = self.label_scope.name
        self.values = values
        self.n_chosen = n_chosen

        if default is not MISSING:
            self.validate({self.label: default})
        self.default_value = default

        assert len(set(values)) == len(values), 'Values must be unique.'
        assert not(any(isinstance(value, Mutable) for value in values)), 'Categorical values must not contain mutables.'
        assert self.n_chosen is None or 1 <= self.n_chosen <= len(self.values), 'n_chosen must be between 1 and n, or None.'
        if weights is not None:
            self.weights = weights
        elif self.n_chosen is None:
            self.weights = [0.5] * len(values)
        else:
            self.weights = [1 / len(values)] * len(values)
        assert len(self.weights) == len(self.values), 'Distribution must have length n.'

        if n_chosen is not None:
            assert abs(sum(self.weights) - 1) < 1e-6, f'Distribution must sum to 1 when n_chosen is {n_chosen}.'
        assert all(0 <= dis <= 1 for dis in self.weights), 'Distribution values must be between 0 and 1.'

    def extra_repr(self):
        if len(self.values) <= 7:
            return f'{self.values!r}, n_chosen={self.n_chosen!r}, label={self.label!r}'
        return '[' +  \
            ', '.join(map(repr, self.values[:3])) + \
            ', ..., ' + \
            ', '.join(map(repr, self.values[-3:])) + \
            f'], n_chosen={self.n_chosen!r}, label={self.label!r}'

    def _simplify_to_categorical_format(self) -> list[LabeledMutable]:
        with self.label_scope:
            mutables: list[LabeledMutable] = [Categorical([True, False], label=str(i)) for i in range(len(self.values))]
            if self.n_chosen is not None:
                from .annotation import ExpressionConstraint
                expr = sum(cast(List[Categorical], mutables)) == self.n_chosen
                assert isinstance(expr, MutableExpression)
                mutables.append(ExpressionConstraint(expr, label='n'))
        return mutables

    def _parse_simple_format(self, sample: Sample) -> SampleValidationError | list[Choice]:
        """Try to freeze the CategoricalMultiple in a simple format."""
        if self.label in sample:
            sample_val = sample[self.label]
            if len(set(sample_val)) != len(sample_val):
                return SampleValidationError(f'{sample_val} must not have duplicates.')
            if self.n_chosen is not None and len(sample_val) != self.n_chosen:
                return SampleValidationError(f'{sample_val} must have length {self.n_chosen}.')
            if not all(x in self.values for x in sample_val):
                return SampleValidationError(f'{sample_val} must be contained in {self.values}.')
            return sample_val
        else:
            return SampleMissingError(self.label, list(sample.keys()))

    def _parse_categorical_format(self, sample: Sample) -> SampleValidationError | list[Choice]:
        """Try to freeze the CategoricalMultiple in a categorical format."""
        mutables = self._simplify_to_categorical_format()
        rv = []
        for i, mutable in enumerate(mutables):
            exception = mutable.check_contains(sample)
            if exception is not None:
                exception.paths.insert(0, self.label)
                return exception
            value = mutable.freeze(sample)
            if i < len(self.values) and value:
                rv.append(self.values[i])
        return rv

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        possible_exc_types = []
        possible_reasons = []
        for parse_fn in [self._parse_simple_format, self._parse_categorical_format]:
            parse_result = parse_fn(sample)
            if not isinstance(parse_result, SampleValidationError):
                return None
            possible_exc_types.append(type(parse_result))
            possible_reasons.append(str(parse_result))
        msg = f'Possible reasons are:\n' + ''.join([f'  * {reason}\n' for reason in possible_reasons])
        if all(exc_type is SampleMissingError for exc_type in possible_exc_types):
            return SampleMissingError(msg)
        return SampleValidationError(msg)

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        """If invoking ``is_leaf`` returns true, return self.
        Otherwise, further break it down to several :class:`Categorical` and :class:`Constraint`.

        See Also
        --------
        Mutable.leaf_mutables
        """
        if is_leaf(self):
            yield self
        else:
            for mutable in self._simplify_to_categorical_format():
                yield from mutable.leaf_mutables(is_leaf)

    def freeze(self, sample: Sample) -> list[Choice]:
        self.validate(sample)
        for parse_fn in [self._parse_simple_format, self._parse_categorical_format]:
            choice = parse_fn(sample)
            if not isinstance(choice, SampleValidationError):
                return choice
        raise RuntimeError('Failed to parse. This should not happen.')

    def default(self, memo: Sample | None = None) -> list[Choice]:
        """The first ``n_chosen`` values. If ``n_chosen`` is None, return all values.

        See Also
        --------
        Mutable.default
        """
        memo = {} if memo is None else memo
        err = self.check_contains(memo)
        if isinstance(err, SampleMissingError):
            if self.default_value is not MISSING:
                memo[self.label] = self.default_value
            else:
                memo[self.label] = self.values[:self.n_chosen]
        rv = self.freeze(memo)
        if self.default_value is not MISSING and rv != self.default_value:
            raise ValueError(f'Default value is specified to be {self.default_value} but got {rv}. '
                             f'Please check the default value of {self.label}.')
        return rv

    def random(self, memo: Sample | None = None, random_state: RandomState | None = None) -> list[Choice]:
        """Randomly sample ``n_chosen`` values. If ``n_chosen`` is None, return an arbitrary subset.

        The random here takes distribution into account.

        See Also
        --------
        Mutable.random
        """
        memo = {} if memo is None else memo
        if random_state is None:
            random_state = RandomState()
        err = self.check_contains(memo)
        if isinstance(err, SampleMissingError):
            if self.n_chosen is None:
                chosen = [value for value in self.values if random_state.random() < self.weights[self.values.index(value)]]
            else:
                chosen = sorted(random_state.choice(len(self.values), self.n_chosen, replace=False, p=self.weights))
                chosen = [self.values[c] for c in chosen]
            memo[self.label] = chosen
        return self.freeze(memo)

    def grid(self, memo: Sample | None = None, granularity: int | None = None) -> Iterable[list[Choice]]:
        """Iterate over all possible values.

        If ``n_chosen`` is None, iterate over all possible subsets, in the order of increasing length.
        Otherwise, iterate over all possible combinations of ``n_chosen`` length,
        using the implementation of :func:`itertools.combinations`.

        See Also
        --------
        Mutable.grid
        """
        memo = {} if memo is None else memo
        err = self.check_contains(memo)

        if isinstance(err, SampleMissingError):
            if self.n_chosen is not None:
                gen = itertools.combinations(self.values, self.n_chosen)
            else:
                gen = itertools.chain.from_iterable(itertools.combinations(self.values, r) for r in range(len(self.values) + 1))

            assert self.label not in memo, 'Memo should not contain the label.'
            for value in gen:
                memo[self.label] = list(value)
                yield self.freeze(memo)
            memo.pop(self.label)
        else:
            yield self.freeze(memo)


class Numerical(MutableSymbol):
    """One variable from a univariate distribution.

    It supports most commonly used distributions including uniform, loguniform,
    normal, lognormal, as well as the quantized version.
    It also supports using arbitrary distribution from :mod:`scipy.stats`.

    Parameters
    ----------
    low
        The lower bound of the domain. Used for uniform and loguniform.
        It will also be used to clip the value if it is outside the domain.
    high
        The upper bound of the domain. Used for uniform and loguniform.
        It will also be used to clip the value if it is outside the domain.
    mu
        The mean of the domain. Used for normal and lognormal.
    sigma
        The standard deviation of the domain. Used for normal and lognormal.
    log_distributed
        Whether the domain is log distributed.
    quantize
        If specified, the final value will be postprocessed with
        ``clip(round(uniform(low, high) / q) * q, low, high)``,
        where the clip operation is used to constrain the generated value within the bounds.
        For example, when quantize is 2.5, all the values will be rounded to the nearest multiple of 2.5.
        Note that, if ``low`` or ``high`` is not a multiple of ``quantize``,
        it will be clipped to ``low`` or ``high`` **after** rounding.
    distribution
        The distribution to use. It should be a ``rv_frozen`` instance,
        which can be obtained by calling ``scipy.stats.distribution_name(...)``.
        If specified, ``low``, ``high``, ``mu``, ``sigma``, ``log_distributed`` will be ignored.
    default
        The default value. If not specified, the default value will be the median of distribution.
    label
        The label of the variable.

    Examples
    --------
    To create a variable uniformly sampled from 0 to 1::

        Numerical(low=0, high=1)

    To create a variable normally sampled with mean 2 and std 3::

        Numerical(mu=2, sigma=3)

    To create a normally sampled variable with mean 0 and std 1, but
    always in the range of [-1, 1] (note that it's not **truncated normal** though)::

        Numerical(mu=0, sigma=1, low=-1, high=1)

    To create a variable uniformly sampled from 0 to 100, but always multiple of 2::

        Numerical(low=0, high=100, quantize=2)

    To create a reciprocal continuous random variable in the range of [2, 6]::

        Numerical(low=2, high=6, log_distributed=True)

    To create a variable sampled from a custom distribution:

        from scipy.stats import beta
        Numerical(distribution=beta(2, 5))
    """

    def __init__(
        self,
        low: float = float('-inf'),
        high: float = float('inf'),
        *,
        mu: float | None = None,
        sigma: float | None = None,
        log_distributed: bool = False,
        quantize: float | None = None,
        distribution: _distn_infrastructure.rv_frozen | None = None,
        default: float | str = MISSING,
        label: str | None = None,
    ) -> None:
        self.quantize = quantize
        self.low = low
        self.high = high
        self.mu = mu
        self.sigma = sigma
        self.log_distributed = log_distributed

        self.label = auto_label(label)

        assert not(any(isinstance(value, Mutable) for value in [low, high, mu, sigma])), 'Numerical parameters must not be mutables.'

        if distribution is not None:
            if mu is not None or sigma is not None or log_distributed:
                raise ValueError('mu, sigma and log_distributed must not be specified if distribution is specified.')
            self.distribution = distribution

        elif mu is not None and sigma is not None:
            # as normal distribution
            if log_distributed:
                self.distribution = lognorm(s=sigma, scale=np.exp(mu))
            else:
                self.distribution = norm(loc=mu, scale=sigma)

        else:
            if log_distributed:
                self.distribution = loguniform(a=low, b=high)
            else:
                self.distribution = uniform(loc=low, scale=high - low)

        if default is not MISSING:
            self.validate({self.label: default})
        self.default_value = default

    def equals(self, other: Any) -> bool:
        """Checks whether two distributions are equal by examining the parameters.

        See Also
        --------
        Mutable.equals
        """
        return type(self) == type(other) and \
            self.distribution.args == other.distribution.args and \
            self.distribution.kwds == other.distribution.kwds and \
            type(self.distribution.dist) == type(other.distribution.dist) and \
            self.quantize == other.quantize and \
            self.default_value == other.default_value and \
            self.label == other.label

    def extra_repr(self) -> str:
        rv = f'{self.low}, {self.high}, '
        if self.mu is not None and self.sigma is not None:
            rv += f'mu={self.mu}, sigma={self.sigma}, '
        if self.quantize is not None:
            rv += f'q={self.quantize}, '
        if self.log_distributed:
            rv += 'log_distributed=True, '
        rv += f'label={self.label!r}'
        return rv

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        if self.label not in sample:
            return SampleMissingError(self.label, list(sample.keys()))
        sample_val = sample[self.label]
        if not isinstance(sample_val, (float, int)):
            raise SampleValidationError(f'Value of {self.label} must be a float or int, but got {type(sample_val)}')
        if self.low is not None and self.low > sample_val:
            return SampleValidationError(f'{sample_val} is lower than lower bound {self.low}')
        if self.high is not None and self.high < sample_val:
            return SampleValidationError(f'{sample_val} is higher than upper bound {self.high}')
        if self.distribution.pdf(sample_val) == 0:
            return SampleValidationError(f'{sample_val} is not in the distribution {self.distribution}')
        if self.quantize is not None and (
            abs(sample_val - self.low) > 1e-6 and
            abs(self.high - sample_val) > 1e-6 and
            abs(sample_val - round(sample_val / self.quantize) * self.quantize) > 1e-6
        ):
            return SampleValidationError(f'{sample_val} is not on the boundary and not a multiple of {self.quantize}')
        return None

    def qclip(self, x: float) -> float:
        """Quantize and clip the value, to satisfy the Q-constraint and low-high bounds."""
        if self.quantize is not None:
            x = round(x / self.quantize) * self.quantize
        if self.low is not None:
            x = max(x, self.low)
        if self.high is not None:
            x = min(x, self.high)
        return x

    def default(self, memo: Sample | None = None) -> float:
        """If default value is not specified, :meth:`Numerical.default` returns median.

        See Also
        --------
        Mutable.default
        """
        memo = {} if memo is None else memo
        err = self.check_contains(memo)
        if isinstance(err, SampleMissingError):
            if self.default_value is not MISSING:
                memo[self.label] = self.default_value
            else:
                memo[self.label] = self.qclip(self.distribution.median())
        rv = self.freeze(memo)
        if self.default_value is not MISSING and rv != self.default_value:
            raise ValueError(f'Default value is specified to be {self.default_value} but got {rv}. '
                             f'Please check the default value of {self.label}.')
        return rv

    def random(self, memo: Sample | None = None, random_state: RandomState | None = None) -> float:
        """Directly sample from the distribution.

        See Also
        --------
        Mutable.random
        """
        memo = {} if memo is None else memo
        if random_state is None:
            random_state = RandomState()
        err = self.check_contains(memo)
        if isinstance(err, SampleMissingError):
            memo[self.label] = self.qclip(self.distribution.rvs(random_state=random_state))
        return self.freeze(memo)

    def grid(self, memo: Sample | None = None, granularity: int | None = None) -> Iterable[float]:
        """Yield a list of samples within the distribution.

        Since the grid of continuous space is infinite, we use granularity to
        specify the number of samples to yield.
        If granularity = 1, grid only explores median point of the distribution.
        If granularity = 2, the quartile points of the distribution will also be generated.
        Granularity = 3 explores the 1/8th points of the distribution, and so on.
        If not specified, granularity defaults to 1.

        Grid will eliminate duplicates within the same granularity.
        Duplicates across different granularity will be ignored.

        Examples
        --------
        >>> list(Numerical(0, 1).grid(granularity=2))
        [0.25, 0.5, 0.75]
        >>> list(Numerical(0, 1).grid(granularity=3))
        [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
        >>> list(Numerical(mu=0, sigma=1).grid(granularity=2))
        [-0.6744897501960817, 0.0, 0.6744897501960817]
        >>> list(Numerical(mu=0, sigma=1, quantize=0.5).grid(granularity=3))
        [-1.0, -0.5, 0.0, 0.5, 1.0]

        See Also
        --------
        Mutable.grid
        """
        memo = {} if memo is None else memo

        if granularity is None:
            granularity = 1
        assert granularity > 0

        err = self.check_contains(memo)
        if isinstance(err, SampleMissingError):
            percentiles = [i / (2 ** granularity) for i in range(1, 2 ** granularity)]
            last_sample: float | None = None
            for p in percentiles:
                sample = self.qclip(self.distribution.ppf(p))
                if last_sample != sample:
                    memo[self.label] = sample
                    last_sample = sample
                    yield self.freeze(memo)
            memo.pop(self.label)
        else:
            yield self.freeze(memo)
