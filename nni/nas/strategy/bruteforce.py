# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['GridSearch', 'Random']

import logging
import warnings
from typing import Iterator, Any

from numpy.random import RandomState

from nni.mutable import Sample
from nni.nas.space import ExecutableModelSpace

from .base import Strategy
from .utils import DeduplicationHelper, RetrySamplingHelper

_logger = logging.getLogger(__name__)


class GridSearch(Strategy):
    """
    Traverse the search space and try all the possible combinations one by one.

    Parameters
    ----------
    shuffle
        Shuffle the order in a candidate list, so that they are tried in a random order.
        Currently, the implementation is a pseudo-random shuffle, which only shuffles the order of every 100 candidates.
    seed
        Random seed.
    """

    _shuffle_buffer_size: int = 100
    _granularity_patience: int = 3  # stop increasing granularity after this many times of no new sample found

    def __init__(self, *, shuffle: bool = True, seed: int | None = None, dedup: bool = True):
        super().__init__()

        self.shuffle = shuffle

        # Internal only:
        # Do not try the same configuration twice.
        # Turning it off might result in duplications when the space has an infinite size,
        # or the strategy tries to resume from a checkpoint,
        # but might improve memory efficiency in extreme cases.
        self._dedup = DeduplicationHelper() if dedup else None
        self._granularity = 1
        self._granularity_processed: int | None = None
        self._no_sample_found_counter = 0
        self._random_state = RandomState(seed)

    def extra_repr(self) -> str:
        return f'shuffle={self.shuffle}, dedup={self._dedup is not None}'

    def _grid_generator(self, model_space: ExecutableModelSpace) -> Iterator[ExecutableModelSpace]:
        if self._no_sample_found_counter >= self._granularity_patience:
            _logger.info('Patience already run out (%d > %d). Nothing to search.',
                         self._no_sample_found_counter, self._granularity_patience)
            return

        finite = self._space_validation(model_space)

        while True:
            new_sample_found = False
            for model in model_space.grid(granularity=self._granularity):
                if self._dedup is not None and not self._dedup.dedup(model.sample):
                    continue

                new_sample_found = True
                yield model

            if not new_sample_found:
                self._no_sample_found_counter += 1
                _logger.info('No new sample found when granularity is %d. Current patience: %d.',
                             self._granularity, self._no_sample_found_counter)
                if self._no_sample_found_counter >= self._granularity_patience:
                    _logger.info('No new sample found for %d times. Stop increasing granularity.',
                                 self._granularity_patience)
                    break
            else:
                self._no_sample_found_counter = 0

            if finite:
                _logger.info('Space is finite. Grid generation is complete.')
                break

            self._granularity += 1
            _logger.info('Space is infinite. Increasing granularity to %d.', self._granularity)

    def _run(self) -> None:
        generator = self._grid_generator(self.model_space)

        if self.shuffle:
            # Shuffle the order of every `_shuffle_buffer_size` candidates.
            shuffle_buffer = []
            generator_running = True

            # Already generated does not mean already submitted.
            # We need to keep track of the granularity actually processed,
            # to avoid skipping granularities when resuming.
            self._granularity_processed = self._granularity

            while generator_running:
                should_submit = False
                try:
                    next_model = next(generator)
                    shuffle_buffer.append(next_model)
                    if len(shuffle_buffer) == self._shuffle_buffer_size:
                        should_submit = True
                except StopIteration:
                    # Submit the final models.
                    should_submit = True
                    generator_running = False

                if should_submit:
                    # Submit models and clear the shuffle buffer.
                    self._random_state.shuffle(shuffle_buffer)
                    for model in shuffle_buffer:
                        if not self.wait_for_resource():
                            _logger.info('Budget exhausted, but search space is not exhausted.')
                            return
                        self.engine.submit_models(model)
                    shuffle_buffer = []

                    # Update granularity processed.
                    self._granularity_processed = self._granularity

        else:
            # Keep this in a separate branch because it's very simple.
            for model in generator:
                if not self.wait_for_resource():
                    _logger.info('Budget exhausted, but search space is not exhausted.')
                    return
                self.engine.submit_models(model)

    def _space_validation(self, model_space: ExecutableModelSpace) -> bool:
        """Check whether the space is supported by grid search.

        Return true if the space is finite, false if it's not.
        Raise error if it's not supported.
        """
        for mutable in model_space.simplify().values():
            # This method will raise error if grid is not implemented.
            if len(list(mutable.grid(granularity=1))) != len(list(mutable.grid(granularity=1 + self._granularity_patience))):
                return False
        return True

    def load_state_dict(self, state_dict: dict) -> None:
        self._granularity = state_dict['granularity']
        self._no_sample_found_counter = state_dict['no_sample_found_counter']
        self._random_state.set_state(state_dict['random_state'])
        _logger.info('Grid search will resume from granularity %d.', self._granularity)
        if self._dedup is not None:
            self._dedup.load_state_dict(state_dict)
        else:
            _logger.info('Grid search would possibly yield duplicate samples since dedup is turned off.')

    def state_dict(self) -> dict:
        result: dict[str, Any] = {'random_state': self._random_state.get_state()}
        if self._granularity_processed is None:
            result.update(granularity=self._granularity, no_sample_found_counter=self._no_sample_found_counter)
        else:
            result.update(granularity=self._granularity_processed, no_sample_found_counter=0)

        if self._dedup is not None:
            result.update(self._dedup.state_dict())
        return result


class Random(Strategy):
    """
    Random search on the search space.

    Parameters
    ----------
    dedup
        Do not try the same configuration twice.
    seed
        Random seed.
    """

    _duplicate_retry = 500

    def __init__(self, *, dedup: bool = True, seed: int | None = None, **kwargs):
        super().__init__()

        if 'variational' in kwargs or 'model_filter' in kwargs:
            warnings.warn('Variational and model filter are no longer supported in random search and will be removed in future releases.',
                          DeprecationWarning)

        self._dedup_helper = DeduplicationHelper(raise_on_dup=True) if dedup else None
        self._retry_helper = RetrySamplingHelper(self._duplicate_retry)

        self._random_state = RandomState(seed)

    def extra_repr(self) -> str:
        return f'dedup={self._dedup_helper is not None}'

    def random(self, model_space: ExecutableModelSpace) -> ExecutableModelSpace:
        """Generate a random model from the space."""
        sample: Sample = {}
        model = model_space.random(random_state=self._random_state, memo=sample)
        if self._dedup_helper is not None:
            self._dedup_helper.dedup(sample)
        return model

    def _run(self) -> None:
        while True:
            # Random search needs retry to:
            # 1. Generate new when dedup is on.
            # 2. Retry when the sample is invalid.
            model = self._retry_helper.retry(self.random, self.model_space)
            if model is None:
                _logger.info('Random generation has run out of patience. There is nothing to search. Exiting.')
                return

            if not self.wait_for_resource():
                break

            self.engine.submit_models(model)

    def load_state_dict(self, state_dict: dict) -> None:
        self._random_state.set_state(state_dict['random_state'])
        if self._dedup_helper is not None:
            self._dedup_helper.load_state_dict(state_dict)

    def state_dict(self) -> dict:
        dedup_state = self._dedup_helper.state_dict() if self._dedup_helper is not None else {}
        return {
            'random_state': self._random_state.get_state(),
            **dedup_state
        }
