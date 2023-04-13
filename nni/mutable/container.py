# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = [
    'MutableList', 'MutableDict'
]

from typing import Any, Iterable, Iterator, Mapping, Callable, Sequence

from .exception import SampleValidationError
from .mutable import Mutable, Sample, LabeledMutable, _mutable_equal


class MutableList(Mutable):
    """The container for a list of mutables.

    :class:`MutableList` will be the cartesian product of all the mutables in the list.

    It can be indexed / sliced like a regular Python list,
    but it also looks like a :class:`nni.mutable.Mutable`,
    which supports :meth:`freeze`, :meth:`contains`, and :meth:`simplify`.

    Parameters
    ----------
    mutables
        A list of mutables.
        It's not encouraged to put non-mutable objects in the list,
        but it's allowed. In that case, the non-mutable objects will be simply ignored
        in all mutable-related operations (e.g., :meth:`simplify`).

    Notes
    -----
    To nest a :class:`MutableList` inside another :class:`MutableList`,
    the inner list must be wrapped in a :class:`MutableList`.
    Otherwise, the mutables inside in the inner list won't be recognized as mutables.
    For example::

        >>> a = [Categorical([1, 2]), Categorical([3, 4])]
        >>> b = Categorical([5, 6])
        >>> lst = MutableList([MutableList(a), b])
        >>> lst.random()
        [[1, 4], 6]

    However, this might NOT be what you expect::

        >>> lst = MutableList([a, b])
        >>> lst.random()
        [[Categorical([1, 2], label='global/1'), Categorical([3, 4], label='global/2')], 6]

    Examples
    --------
    >>> from nni.mutable import *
    >>> space = MutableList([Categorical(['a', 'b']), Categorical(['c', 'd'])])
    >>> space.random()
    ['b', 'd']
    """

    def __init__(self, mutables: Iterable[Mutable | Any] | None = None) -> None:
        if mutables:
            self.mutables: list[Mutable] = list(mutables)
        else:
            self.mutables: list[Mutable] = []

    def extra_repr(self) -> str:
        return repr(self.mutables)

    def freeze(self, sample: Sample) -> list:
        self.validate(sample)
        rv = []
        for mutable in self:
            if isinstance(mutable, Mutable):
                rv.append(mutable.freeze(sample))
            else:
                # In case it's not a mutable, we just return it.
                rv.append(mutable)
        return rv

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        for index, mutable in enumerate(self):
            if isinstance(mutable, Mutable):
                exception = mutable.check_contains(sample)
                if exception is not None:
                    exception.paths.insert(0, '[' + str(index) + ']')
                    return exception
        return None

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        for mutable in self:
            if isinstance(mutable, Mutable):
                yield from mutable.leaf_mutables(is_leaf)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return _mutable_equal(self.mutables, other.mutables)
        return False

    def __getitem__(self, idx: int | slice) -> Mutable | MutableList:
        if isinstance(idx, slice):
            return self.__class__(self.mutables[idx])
        return self.mutables[idx]

    def __setitem__(self, idx: int, mutable: Mutable) -> None:
        self[idx] = mutable

    def __delitem__(self, idx: int | slice) -> None:
        del self.mutables[idx]

    def __len__(self) -> int:
        return len(self.mutables)

    def __iter__(self) -> Iterator[Mutable]:
        return iter(self.mutables)

    def __contains__(self, item: Mutable) -> bool:
        return item in self.mutables

    def __iadd__(self, mutables: Iterable[Mutable]) -> MutableList:
        return self.extend(mutables)

    def insert(self, index: int, mutable: Mutable) -> None:
        """Insert a given mutable before a given index in the list.

        Parameters
        ----------
        index
            Index before which the mutable will be inserted.
        mutable
            Mutable to be inserted.
        """
        self.mutables.insert(index, mutable)

    def append(self, mutable: Mutable) -> MutableList:
        """Appends a given mutable to the end of the list.

        Parameters
        ----------
        mutable
            Mutable to be appended.
        """
        self.mutables.append(mutable)
        return self

    def extend(self, mutables: Iterable[Mutable]) -> MutableList:
        r"""Appends mutables from a Python iterable to the end of the list.

        Parameters
        ----------
        mutables
            Mutables to be appended.
        """
        self.mutables.extend(mutables)
        return self


class MutableDict(Mutable):
    """The container for a dict of mutables.
    It looks like a regular Python dict, but it also works like a :class:`nni.mutable.Mutable` instance.

    :class:`MutableDict` will be the cartesian product of all the mutables in the dict.
    It's guaranteed to be ordered by the insertion order
    (based on a `language feature <https://mail.python.org/pipermail/python-dev/2017-December/151283.html>`__ of Python 3.7+).

    :class:`MutableDict` is usually used to make a mutable space human-readable.
    It can further be nested and used together with :class:`MutableList`. For example::

        >>> search_space = MutableDict({
        ...     'trainer': MutableDict({
        ...         'optimizer': Categorical(['sgd', 'adam']),
        ...         'lr': Numerical(1e-4, 1e-2, log_distributed=True),
        ...         'decay_epochs': MutableList([
        ...             Categorical([10, 20]),
        ...             Categorical([30, 50])
        ...         ]),
        ...     }),
        ...     'model': MutableDict({
        ...         'type': Categorical(['resnet18', 'resnet50']),
        ...         'pretrained': Categorical([True, False])
        ...     }),
        ... })
        >>> search_space.random()
        {'trainer': {'optimizer': 'sgd', 'lr': 0.000176, 'decay_epochs': [10, 30]}, 'model': {'type': 'resnet18', 'pretrained': True}}

    There is a fundamental difference between the key appeared in the dict,
    and the label of the mutables. The key is used to access the mutable and make the frozen dict more human-readable.
    Yet the label is used to identify the mutable in the whole search space, and typically used by search algorithms.
    In the example above, although the each variable have the keys like ``'optimizer'``, ``'lr'``, ``'type'``,
    their label is still not specified and thus auto-generated::

        >>> search_space['trainer']['optimizer'].label
        'global/1'
        >>> search_space.simplify()
        {
            'global/1': Categorical(['sgd', 'adam'], label='global/1'),
            'global/2': Numerical(0.0001, 0.01, label='global/2'),
            'global/3': Categorical([10, 20], label='global/3'),
            'global/4': Categorical([30, 50], label='global/4'),
            'global/5': Categorical(['resnet18', 'resnet50'], label='global/5'),
            'global/6': Categorical([True, False], label='global/6')
        }
        >>> search_space.freeze({
        ...     'global/1': 'adam',
        ...     'global/2': 0.0001,
        ...     'global/3': 10,
        ...     "global/4': 50,
        ...     'global/5': 'resnet50',
        ...     'global/6': False
        ... })
        {'trainer': {'optimizer': 'adam', 'lr': 0.0001, 'decay_epochs': [10, 50]}, 'model': {'type': 'resnet50', 'pretrained': False}}

    Here's another example where label is manually specified to indicate the relationship between the mutables::

        >>> search_space = MutableList([
        ...     MutableDict({
        ...         'in_features': Categorical([10, 20], label='hidden_dim'),
        ...         'out_features': Categorical([10, 20], label='hidden_dim') * 2,
        ...     }),
        ...     MutableDict({
        ...         'in_features': Categorical([10, 20], label='hidden_dim') * 2,
        ...         'out_features': Categorical([10, 20], label='hidden_dim') * 4,
        ...     }),
        ... ])
        >>> search_space.random()
        [{'in_features': 20, 'out_features': 40}, {'in_features': 40, 'out_features': 80}]

    Parameters
    ----------
    mutables
        :class:`MutableDict` can be instantiated in one of two ways.
        Either you pass a dictionary to mutables, or you pass the mutables as keyword arguments
        (where keyword named ``mutables`` should be avoided).
        It's not encouraged to put non-mutable objects in the dict, but it's allowed,
        in which case they will be simply ignored.

    Examples
    --------
    The following two usages are equivalent::

        >>> MutableDict({'a': Categorical([1, 2]), 'b': Categorical([3, 4])})
        MutableDict({'a': Categorical([1, 2], label='global/1'), 'b': Categorical([3, 4], label='global/2')})
        >>> MutableDict(a=Categorical([1, 2]), b=Categorical([3, 4]))
        MutableDict({'a': Categorical([1, 2], label='global/3'), 'b': Categorical([3, 4], label='global/4')})
    """

    def __init__(self, mutables: Mapping[str, Mutable] | None = None, **mutable_kwargs: Mutable) -> None:
        if isinstance(mutables, Mapping):
            mutables = dict(mutables)
        elif isinstance(mutables, Sequence):
            mutables = dict(mutables)
        elif mutables is None:
            mutables = dict()
        else:
            assert isinstance(mutables, dict), f'mutables should be a dict, got {type(mutables)}'

        # Add additional kwargs to mutables
        for key, space in mutable_kwargs.items():
            if key not in mutables:
                mutables[key] = space
            else:
                raise ValueError(f'Keyword "{key}" already exists in the dictionary.')

        self.mutables = mutables

    def extra_repr(self) -> str:
        return repr(self.mutables)

    def freeze(self, sample: Sample) -> dict:
        self.validate(sample)
        rv = {}
        for key, mutable in self.items():
            if isinstance(mutable, Mutable):
                rv[key] = mutable.freeze(sample)
            else:
                # In case it's not a mutable, we just return it.
                rv[key] = mutable
        return rv

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        for key, value in self.items():
            if isinstance(value, Mutable):
                exception = value.check_contains(sample)
                if exception is not None:
                    exception.paths.insert(0, key)
                    return exception
        return None

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        for mutable in self.values():
            if isinstance(mutable, Mutable):
                yield from mutable.leaf_mutables(is_leaf)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return _mutable_equal(self.mutables, other.mutables)
        return False

    def __getitem__(self, key: str) -> Mutable:
        return self.mutables[key]

    def __setitem__(self, key: str, module: Mutable) -> None:
        self.mutables[key] = module

    def __delitem__(self, key: str) -> None:
        del self.mutables[key]

    def __len__(self) -> int:
        return len(self.mutables)

    def __iter__(self) -> Iterator[str]:
        return iter(self.mutables)

    def __contains__(self, key: str) -> bool:
        return key in self.mutables

    def clear(self) -> None:
        """Remove all items from the MutableDict."""
        self.mutables.clear()

    def pop(self, key: str) -> Mutable:
        """Remove key from the MutableDict and return its module."""
        return self.mutables.pop(key)

    def keys(self) -> Iterable[str]:
        """Return an iterable of the MutableDict keys."""
        return self.mutables.keys()

    def items(self) -> Iterable[tuple[str, Mutable]]:
        """Return an iterable of the MutableDict key/value pairs."""
        return self.mutables.items()

    def values(self) -> Iterable[Mutable]:
        """Return an iterable of the MutableDict values."""
        return self.mutables.values()

    def update(self, mutables: Mapping[str, Mutable]) -> None:
        """Update the mutable dict with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.
        """
        return self.mutables.update(mutables)
