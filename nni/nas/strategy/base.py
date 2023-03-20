# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Iterator, cast

from nni.nas.execution import ExecutionEngine
from nni.nas.space import ExecutableModelSpace, ModelStatus
from nni.typehint import TrialMetric

_logger = logging.getLogger(__name__)


class StrategyStatus(str, Enum):
    """Status of a strategy.

    A strategy is in one of the following statuses:

    - ``EMPTY``: The strategy is not initialized.
    - ``INITIALIZED``: The strategy is initialized (with a model space), but not started.
    - ``RUNNING``: The strategy is running.
    - ``SUCCEEDED``: The strategy has successfully ended.
    - ``INTERRUPTED``: The strategy is interrupted.
    - ``FAILED``: The strategy is stopped due to error.
    """
    EMPTY = 'empty'
    INITIALIZED = 'initialized'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    INTERRUPTED = 'interrupted'
    FAILED = 'failed'


class Strategy:
    """Base class for NAS strategies.

    To explore a space with a strategy, use::

        strategy = MyStrategy()
        strategy(model_space, engine)

    The strategy has a :meth:`run` method, that defines the process of exploring a NAS space.

    Strategy is stateful. It might store information of the current :meth:`initialize` and :meth:`run` as member attributes.
    We do not allow :meth:`run` a strategy twice with same, or different model spaces.

    Subclass should override :meth:`_initialize` and :meth:`_run`,
    as well as :meth:`state_dict` and :meth:`load_state_dict` for checkpointing.
    """

    def __init__(self, model_space: ExecutableModelSpace | None = None, engine: ExecutionEngine | None = None):
        self._engine: ExecutionEngine | None = None
        self._model_space: ExecutableModelSpace | None = None

        # Status is internal for now.
        self._status = StrategyStatus.EMPTY
        if engine is not None and model_space is not None:
            self.initialize(model_space, engine)
        elif engine is not None or model_space is not None:
            raise ValueError('Both engine and model_space should be provided, or both should be None.')

    @property
    def engine(self) -> ExecutionEngine:
        """Strategy should use :attr:`engine` to submit models, listen to metrics, and do budget / concurrency control.

        The engine is set by :meth:`set_engine`, either manually, or by a NAS experiment.

        The engine could be either a real engine, or a middleware that wraps a real engine.
        It doesn't make any difference because their interface are the same.

        See Also
        --------
        nni.nas.execution.ExecutionEngine
        """
        if self._engine is None:
            raise RuntimeError("Strategy is not attached to an engine.")
        return self._engine

    @property
    def model_space(self) -> ExecutableModelSpace:
        """The model space that strategy is currently exploring.

        It should be the same one as the input argument of :meth:`run`,
        but the property exists for convenience.

        See Also
        --------
        nni.nas.space.ExecutableModelSpace
        """
        if self._model_space is None:
            raise RuntimeError("Strategy is not attached to a model space.")
        return self._model_space

    def wait_for_resource(self) -> bool:
        while not self.engine.idle_worker_available():
            if not self.engine.budget_available():
                _logger.debug('No worker and budget is exhausted. Strategy should not submit new models.')
                return False

            time.sleep(1.)

        # Sometimes engine has workers but no budget.
        return self.engine.budget_available()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self):
        return ''

    def __call__(self, model_space: ExecutableModelSpace, engine: ExecutionEngine) -> None:
        """Explore the model space.

        This is a convenience method that calls :meth:`initialize`, and :meth:`run`, subsequently.
        """
        if not hasattr(self, '_status'):
            raise RuntimeError(f'Strategy {self.__class__.__name__} does not have _status. Maybe it forgets to call super().__init__?')
        self.initialize(model_space, engine)
        self.run()

    def initialize(self, model_space: ExecutableModelSpace, engine: ExecutionEngine) -> ExecutableModelSpace:
        """Initialize the strategy.

        This method should be called before :meth:`run` to initialize some states.

        Some strategies might even mutate the ``model_space``. They should return the mutated model space.

        :meth:`load_state_dict` can be called after :meth:`initialize` to restore the state of the strategy.

        Subclass override :meth:`_initialize` instead of this method.
        """
        if self._status != StrategyStatus.EMPTY:
            raise RuntimeError('Strategy has already been initialized.')
        self._model_space = model_space
        self._engine = engine
        model_space = self._initialize(model_space, engine)
        self._status = StrategyStatus.INITIALIZED
        return model_space

    def run(self) -> None:
        """Explore the model space.

        This should be the main part of a NAS experiment.
        Strategies decide how to explore the model space.
        They can submit models to :attr:`engine` for training and evaluation.

        The strategy doesn't have to wait for all the models it submits to finish training.

        The caller of :meth:`run` is responsible of setting the :attr:`engine` and :attr:`model_space` before calling :meth:`run`.

        Subclass override :meth:`_run` instead of this method.
        """
        try:
            if self._status == StrategyStatus.RUNNING:
                raise RuntimeError('Strategy is already running.')

            if self._status == StrategyStatus.INTERRUPTED:
                raise RuntimeError('Strategy is interrupted. Please resume by creating a new strategy and load_state_dict.')

            if self._status != StrategyStatus.INITIALIZED:
                raise RuntimeError('Strategy should not be called twice.')

            self._status = StrategyStatus.RUNNING

            # Explore the model space.
            self._run()
            # Strategy doesn't wait for the models it submitted.

            _logger.debug('Strategy has successfully finished.')
            self._status = StrategyStatus.SUCCEEDED
        except KeyboardInterrupt:
            _logger.warning('Strategy is interrupted.')
            self._status = StrategyStatus.INTERRUPTED
            raise
        except:
            _logger.error('Strategy failed to execute.')
            self._status = StrategyStatus.FAILED
            raise
        finally:
            try:
                self._cleanup()
            except:
                _logger.exception('Exception raised during strategy cleanup. Ignore.')

    def _initialize(self, model_space: ExecutableModelSpace, engine: ExecutionEngine) -> ExecutableModelSpace:
        """Implementation of :meth:`initialize`.

        In most cases, subclass should override this method instead of :meth:`initialize`,
        for strategy initialization.
        """
        _logger.debug('Strategy %r is initialized.', self)
        return model_space  # un-mutated

    def _run(self) -> None:
        """Implementation of :meth:`run`.

        In most cases, subclass should override this method instead of :meth:`run`,
        for strategy exploration.
        """
        raise NotImplementedError(f'Strategy {self} did not implement run().')

    def _cleanup(self) -> None:
        """Clean up the strategy.

        This method is called when :meth:`run` finishes.

        Subclass can optionally override this to unregister itself from the engine,
        so that it won't get erroneously notified when the engine turns to running models submitted by another strategy.
        Since strategy can't run twice, by design it shouldn't register the callbacks again.

        To make APIs like :meth:`list_models` continue to work, we generally don't recommend "unset" the engines here.
        """
        _logger.debug('Strategy %r cleaned up.', self)

    def list_models(self, sort: bool = True, limit: int | None = None) -> Iterator[ExecutableModelSpace]:
        """List all the models that is ever searched by the engine.

        A typical use case of this is to get the top-performing models produced during :meth:`run`.

        The default implementation uses :meth:`~nni.nas.execution.ExecutionEngine.list_models` to
        retrieve a list of models from the execution engine.

        Parameters
        ----------
        sort
            Whether to sort the models by their metric (in descending order).
            If sorted is true, only models with "Trained" status and non-``None`` metric are returned.
        limit
            Limit the number of models to return.

        Returns
        -------
        An iterator of models.
        """
        if self._status in (StrategyStatus.INITIALIZED, StrategyStatus.EMPTY):
            raise RuntimeError('Strategy has not been run.')

        if sort:
            models = [model for model in self.engine.list_models(status=ModelStatus.Trained) if model.metric is not None]
            if limit is not None and limit > len(models):
                _logger.warning('Only %d models are trained, but %d top models are requested.', len(models), limit)
            yield from sorted(models, key=lambda m: cast(TrialMetric, m.metric), reverse=True)[:limit]

        else:
            for i, model in enumerate(self.engine.list_models()):
                if limit is not None and i >= limit:
                    break
                yield model

    def state_dict(self) -> dict:
        """Dump the state of the strategy.

        This is used for checkpointing.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement `state_dict()`.')

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state of the strategy. This is used for loading checkpoints.

        The *state* of strategy is some variables that are related to the current exploration process.
        The loading is often done after :meth:`initialize` and before :meth:`run`.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement `load_state_dict()`.')


BaseStrategy = Strategy
