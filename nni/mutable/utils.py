# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = [
    'uid', 'reset_uid',
    'NoContextError', 'ContextStack', 'get_current_context',
    'LabelNamespace', 'auto_label',
]

import logging
from collections import defaultdict
from typing import Any

_last_uid = defaultdict(int)

_logger = logging.getLogger(__name__)


def uid(namespace: str = 'default') -> int:
    """Global counter for unique id. Not thread-safe."""
    _last_uid[namespace] += 1
    return _last_uid[namespace]


def reset_uid(namespace: str = 'default') -> None:
    """Reset counter for a specific namespace."""
    _last_uid[namespace] = 0


class NoContextError(Exception):
    """Exception raised when context is missing."""
    pass


class ContextStack:
    """
    This is to maintain a globally-accessible context environment that is visible to everywhere.

    To initiate::

        with ContextStack(namespace, value):
            ...

    Inside the context, you can access the nearest value put into ``with``::

        get_current_context(namespace)

    Notes
    -----
    :class:`ContextStack` is not multi-processing safe. Also, the values will get cleared for a new process.
    """

    _stack: dict[str, list] = defaultdict(list)

    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value

    def __enter__(self):
        self.push(self.key, self.value)
        return self

    def __exit__(self, *args, **kwargs):
        self.pop(self.key)

    @classmethod
    def push(cls, key: str, value: Any):
        cls._stack[key].append(value)

    @classmethod
    def pop(cls, key: str) -> Any:
        if not cls._stack[key]:
            raise NoContextError(f'Context with key {key} is empty.')
        return cls._stack[key].pop()

    @classmethod
    def top(cls, key: str) -> Any:
        if not cls._stack[key]:
            raise NoContextError(f'Context with key {key} is empty.')
        return cls._stack[key][-1]

    @classmethod
    def stack(cls, key: str) -> list:
        return list(cls._stack[key])


def get_current_context(key: str) -> Any:
    return ContextStack.top(key)


# The default namespace
_DEFAULT_LABEL_NAMESPACE = 'param'


class LabelNamespace:
    """
    To support automatic labeling of mutables.

    A namespace is bounded to a key. Namespace bounded to different keys are completed isolated.
    The default key is ``param``.
    When entering a :class:`LabelNamespace` with-context, the number of mutables will recount from 1,
    and thus the generation of labels are reproducible.

    The label namespace is NOT thread-safe. The behavior is undefined if multiple threads are
    trying to enter the namespace at the same time.

    Namespace can have sub-namespaces (with the same key). The numbering will be chained (e.g., ``model_1_4_2``).

    :class:`LabelNamespace` is implemented based on :class:`ContextStack`.

    Examples
    --------
    >>> with LabelNamespace('param'):
    ...     label1 = auto_label()       # param_1
    ...     label2 = auto_label()       # param_2
    ...     with LabelNamespace('param'):
    ...         label3 = auto_label()   # param_3_1
    ...         label4 = auto_label()   # param_3_2
    ...     with LabelNamespace('another'):
    ...         label5 = auto_label('another')   # another_1
    ...     with LabelNamespace('param'):
    ...         label6 = auto_label()   # param_4_1
    >>> with LabelNamespace('param'):
    ...     label7 = auto_label()       # param_1, because the counter is reset
    """

    def __init__(self, key: str = _DEFAULT_LABEL_NAMESPACE):
        assert '_' not in key, f'Key cannot contain underscore: {key}'

        # for example, key: "param"
        self.key = key

        # the "path" of current name
        # By default, it's ``[]``
        # If a LabelNamespace is nested inside another one, it will become something like ``[1, 3, 2]``.
        # See ``__enter__``.
        self.name_path: list[int] = []

    def __enter__(self):
        # Enter the label namespace resets the counter associated with the namespace.
        #
        # It also pushes itself into the stack, so as to support nested namespace.
        # For example, currently the top of stack is [1, 2, 2], and [1, 2, 2, 3] is used,
        # the next thing up is [1, 2, 2, 4].
        # `reset_uid` to count from zero for "param_1_2_2_4"
        try:
            parent_context: LabelNamespace = LabelNamespace.current(self.key)
            next_uid = uid(parent_context._simple_name())
            self.name_path = parent_context.name_path + [next_uid]
            ContextStack.push(self.key, self)
            reset_uid(self._simple_name())
        except NoContextError:
            # not found, no existing namespace
            self.name_path = []
            ContextStack.push(self.key, self)
            reset_uid(self._simple_name())
        return self

    def __exit__(self, *args, **kwargs):
        ContextStack.pop(self.key)

    def _simple_name(self) -> str:
        return self.key + ''.join(['_' + str(k) for k in self.name_path])

    def __repr__(self):
        return f'LabelNamespace(name={self._simple_name()})'

    @staticmethod
    def current(key: str = _DEFAULT_LABEL_NAMESPACE) -> LabelNamespace:
        """Get the current namespace associated with the key.

        Examples
        --------
        >>> with LabelNamespace("param") as namespace:
        ...     # somewhere in the middle of the code.
        ...     LabelNamespace.current("param")     # Will return the same namespace

        Raises
        ------
        NoContextError
            If no context is found.
        """
        return ContextStack.top(key)


def auto_label(key: str = _DEFAULT_LABEL_NAMESPACE) -> str:
    """Automatically generate a label in case the label is not set.

    It will look for the nearest :class:`LabelNamespace` with the specified key.
    If not found, it prints a warning and use the ``global`` namespace.

    Parameters
    ----------
    key
        The key of the namespace to use. If not specified, the default namespace (param) will be used.

    Examples
    --------
    >>> label1 = auto_label()               # global_1
    >>> with LabelNamespace('param'):
    ...     label2 = auto_label('param')    # param_1, because in the namespace "param"
    >>> with LabelNamespace():
    ...     label3 = auto_label()           # param_1, default key is used
    >>> with LabelNamespace('another'):
    ...     label4 = auto_label('another')  # another_1
    ...     label5 = auto_label()           # global_2
    """

    # NOTE: It's not recommended to use the default namespace because the stable label numbering cannot be guaranteed.
    # However, we allow such usage currently because it's mostly used in evaluator,
    # whose initialization relies on trace, and doesn't need to be reproducible in trial code.
    try:
        current_context = ContextStack.top(key)
    except NoContextError:
        # fallback to use "default" namespace
        # it won't be registered
        _logger.warning(
            'LabelNamespace is missing. Global numbering is used. '
            'Note that we always recommend specifying `label=...` manually.',
        )
        current_context = LabelNamespace('global')

    next_uid = uid(current_context._simple_name())
    return current_context._simple_name() + '_' + str(next_uid)
