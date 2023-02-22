# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

"""Utilities to freeze mutables upon its creation (either before or after),
such that when a proper context is provided, the mutables should look exactly the same as its frozen version.
"""

__all__ = [
    'ensure_frozen', 'frozen_context', 'frozen_factory',
]

import logging
from contextlib import contextmanager
from typing import Any, Callable

from .mutable import Mutable, Sample
from .utils import NoContextError, ContextStack

_ENSURE_FROZEN_STRICT = True
_FROZEN_CONTEXT_KEY = '_frozen'

_logger = logging.getLogger(__name__)


def ensure_frozen(mutable: Mutable | Any, *, strict: bool = True, sample: Sample | None = None, retries: int = 1000) -> Any:
    """Ensure a mutable is frozen. Used when passing the mutable to a function which doesn't accept a mutable.

    If the argument is not a mutable, nothing happens.
    Otherwise, :meth:`~nni.mutable.Mutable.freeze` will be called if sample is given.
    If sample is None, :func:`ensure_frozen` will also try to fill the sample
    with the content in :class:`frozen_context`.
    Or else :meth:`~nni.mutable.Mutable.robust_default` will be called on the mutable.

    Parameters
    ----------
    mutable : nni.mutable.Mutable or any
        The mutable to freeze.
    strict
        Whether to raise an error if sample context is not provided and not found.
    sample
        The context to freeze the mutable with.
    retries
        Control the number of retries in case :meth:`~nni.mutable.Mutable.robust_default` is called.

    Examples
    --------
    >>> with frozen_context({'a': 2}):
    ...     ensure_frozen(Categorical([1, 2, 3], label='a'))
    2
    >>> ensure_frozen(Categorical([1, 2, 3]), strict=False)
    1
    >>> ensure_frozen(Categorical([1, 2, 3], label='a'), sample={'a': 2}, strict=False)
    2
    >>> ensure_frozen('anything', strict=False)
    'anything'
    """
    if not isinstance(mutable, Mutable):
        return mutable

    # If we're in a frozen context, just use the current.
    if sample is None:
        ctx = frozen_context.current()
        if ctx is not None:
            sample = ctx

    if sample is not None:
        # If we have a sample, we can just use it.
        # Use freeze here to detect potential label mismatch errors.
        try:
            return mutable.freeze(sample)
        except:
            _logger.error(
                'Failed to freeze mutable %s with sample %s. '
                'In NAS, please make sure to have registered it via add_mutable(). '
                'Otherwise, please make sure you are not inside a frozen_context.',
                mutable, sample)
            raise
    else:
        if retries < 0 or (_ENSURE_FROZEN_STRICT and strict):
            raise RuntimeError(
                f'No frozen context is found for {mutable!r}. Assuming no context. '
                'If you are using NAS, you are probably using `ensure_frozen` in forward, or outside the init of ModelSpace. '
                'Please avoid doing this as they will lead to erroneous results.'
            )

        # TODO: Currently only mutable parameters in NAS evaluator end up here.
        # It might cause consistency issues between multiple parameters without context.
        # I don't want to throw a warning here, but there should be a smarter way to do this.
        return mutable.robust_default(retries=retries)


class frozen_context(ContextStack):
    """
    Context manager to set a sample into context.
    Then the sample will be retrievable from an arbitrary level of function calls via :func:`current_frozen_context`.

    There are two use cases:

    1. Setting a global sample so that some modules can directly create the frozen version, rather than first-create-and-freeze.
    2. Sharing default / dry-run samples when the search space is dynamically created.

    The implementation is basically adding another layer of empty dict on top of a global stack.
    When retrieved, all dicts in the stack will be merged, from the bottom to the top.
    When updated, only the dict on the top will be updated.

    Parameters
    ----------
    sample
        The sample to be set into context.

    Returns
    -------
    Context manager that provides a frozen context.

    Examples
    --------
    ::
        def some_func():
            print(frozen_context.current()['learning_rate'])  # 0.1

        with frozen_context({'learning_rate': 0.1}):
            some_func()
    """

    def __init__(self, sample: Sample | None = None):
        super().__init__(_FROZEN_CONTEXT_KEY, sample or {})

    @staticmethod
    def top_context() -> frozen_context:
        return ContextStack.top(_FROZEN_CONTEXT_KEY)

    @staticmethod
    def current() -> dict | None:
        """Retrieve the current frozen context.
        If multiple layers have been found, they would be merged from bottom to top.

        Returns
        -------
        The sample in frozen context.
        If no sample is found, return none.
        """
        try:
            ContextStack.top(_FROZEN_CONTEXT_KEY)
            sample: Sample = {}
            for ctx in ContextStack.stack(_FROZEN_CONTEXT_KEY):
                if not isinstance(ctx, dict):
                    raise TypeError(f'Expect architecture to be a dict, found: {ctx}')
                sample.update(ctx)
            return sample
        except NoContextError:
            return None

    @staticmethod
    def update(sample: Sample) -> None:
        """
        Update the current dry run context.
        Only the topmost context will be updated.

        Parameters
        ----------
        sample
            The sample to be updated into context.
        """
        try:
            ctx = ContextStack.top(_FROZEN_CONTEXT_KEY)
            assert isinstance(ctx, dict)
            ctx.update(sample)
        except NoContextError:
            raise RuntimeError('No frozen context is found. Please use frozen_context() to create one.')

    @staticmethod
    @contextmanager
    def bypass():
        """
        Ignore the most recent :class:`frozen_context`.

        This is useful in creating a search space within a ``frozen_context()`` context.
        Under the hood, it only disables the most recent one frozen context, which means,
        if it's currently in a nested with-frozen-arch context, multiple ``bypass()`` contexts is required.

        Examples
        --------
        >>> with frozen_context(arch_dict):
        ...     with frozen_context.bypass():
        ...         model_space = ModelSpace()
        """

        NO_CONTEXT = '_no_ctx_'

        sample = NO_CONTEXT  # make linter happy
        try:
            try:
                sample = ContextStack.pop(_FROZEN_CONTEXT_KEY)
            except IndexError:
                # context unavailable
                sample = NO_CONTEXT
            yield
        finally:
            if sample is not NO_CONTEXT:
                ContextStack.push(_FROZEN_CONTEXT_KEY, sample)


class frozen_factory:
    """Create a factory object that invokes a function with a frozen context.

    Parameters
    ----------
    callable
        The function to be invoked.
    sample
        The sample to be used as the frozen context.

    Examples
    --------
    >>> factory = frozen_factory(ModelSpaceClass, {"choice1": 3})
    >>> model = factory(channels=16, classes=10)
    """

    # NOTE: mutations on ``init_args`` and ``init_kwargs`` themselves are not supported.

    def __init__(self, callable: Callable[..., Any], sample: Sample | frozen_context):  # pylint: disable=redefined-builtin
        self.callable = callable
        if not isinstance(sample, frozen_context):
            self.sample = frozen_context(sample)
        else:
            self.sample = sample

    def __call__(self, *init_args, **init_kwargs):
        with self.sample:
            return self.callable(*init_args, **init_kwargs)

    def __repr__(self):
        return f'frozen_factory(callable={self.callable}, arch={self.sample.value})'
