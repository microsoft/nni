# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['DeduplicationHelper', 'DuplicationError', 'RetrySamplingHelper']

import logging
from typing import Any, Type, TypeVar, Callable

from nni.mutable import SampleValidationError

_logger = logging.getLogger(__name__)

T = TypeVar('T')


def _to_hashable(obj):
    """Trick to make a dict saveable in a set."""
    if isinstance(obj, dict):
        return frozenset((k, _to_hashable(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return tuple(_to_hashable(v) for v in obj)
    return obj


class DuplicationError(SampleValidationError):
    """Exception raised when a sample is duplicated."""

    def __init__(self, sample):
        super().__init__(f'Duplicated sample found: {sample}')


class DeduplicationHelper:
    """Helper class to deduplicate samples.

    Different from the deduplication on the HPO side,
    this class simply checks if a sample has been tried before, and does nothing else.
    """

    def __init__(self, raise_on_dup: bool = False):
        self._history = set()
        self._raise_on_dup = raise_on_dup

    def dedup(self, sample: Any) -> bool:
        """
        If the new sample has not been seen before, it will be added to the history and return True.
        Otherwise, return False directly.

        If raise_on_dup is true, a :class:`DuplicationError` will be raised instead of returning False.
        """
        sample = _to_hashable(sample)
        if sample in self._history:
            _logger.debug('Duplicated sample found: %s', sample)
            if self._raise_on_dup:
                raise DuplicationError(sample)
            return False
        self._history.add(sample)
        return True

    def remove(self, sample: Any) -> None:
        """
        Remove a sample from the history.
        """
        self._history.remove(_to_hashable(sample))

    def reset(self):
        self._history = set()

    def state_dict(self):
        return {
            'dedup_history': list(self._history)
        }

    def load_state_dict(self, state_dict):
        self._history = set(state_dict['dedup_history'])


class RetrySamplingHelper:
    """Helper class to retry a function until it succeeds.

    Typical use case is to retry random sampling until a non-duplicate / valid sample is found.

    Parameters
    ----------
    retries
        Number of retries.
    exception_types
        Exception types to catch.
    raise_last
        Whether to raise the last exception if all retries failed.
    """

    def __init__(self,
                 retries: int = 500,
                 exception_types: tuple[Type[Exception]] = (SampleValidationError,),
                 raise_last: bool = False):
        self.retries = retries
        self.exception_types = exception_types
        self.raise_last = raise_last

    def retry(self, func: Callable[..., T], *args, **kwargs) -> T | None:
        for retry in range(self.retries):
            try:
                return func(*args, **kwargs)
            except self.exception_types as e:
                if retry in [0, 10, 100, 1000]:
                    _logger.debug('Sampling failed. %d retries so far. Exception caught: %r', retry, e)
                if retry >= self.retries - 1 and self.raise_last:
                    _logger.warning('Sampling failed after %d retries. Giving up and raising the last exception.', self.retries)
                    raise

        _logger.warning('Sampling failed after %d retires. Giving up and returning None.', self.retries)
        return None
