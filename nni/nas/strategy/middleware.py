# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['Chain', 'StrategyMiddleware', 'Filter', 'Deduplication', 'FailureHandler', 'MultipleEvaluation', 'MedianStop']

import copy
import logging
import warnings
from collections import defaultdict, deque
from typing import Iterable, Callable, Any, Iterator, List, cast
from typing_extensions import Literal

import numpy as np

from nni.nas.execution import ExecutionEngine, Middleware
from nni.nas.execution.event import ModelEventType, ModelEvent, TrainingEndEvent, IntermediateMetricEvent, FinalMetricEvent
from nni.nas.space import ExecutableModelSpace, ModelStatus

from nni.typehint import TrialMetric

from .base import Strategy
from .utils import DeduplicationHelper

_logger = logging.getLogger(__name__)


class Chain(Strategy):
    """Chain a :class:`~nni.nas.strategy.base.Strategy` (main strategy) with several :class:`StrategyMiddleware`.

    All the communications between strategy and execution engine will pass through the chain of middlewares.
    For example, when the strategy submits a model, it will be handled by the middleware,
    which decides whether to hand over to the next middleware, or to manipulate, or even block the model.
    The last middleware is connected to the real execution engine (which might be also guarded by a few middlewares).

    Parameters
    ----------
    strategy
        The main strategy.
        There can be exactly one strategy which is submitting models actively, which is therefore called **main strategy**.
    *middlewares
        A chain of middlewares. At least one.

    See Also
    --------
    StrategyMiddleware
    """

    def __init__(self, strategy: Strategy, *middlewares: StrategyMiddleware):
        if not middlewares:
            raise ValueError('Chain strategy requires at least one middleware.')
        if not isinstance(strategy, Strategy):
            raise ValueError('The first argument of Chain must be a instance of Strategy.')
        if not all(isinstance(m, StrategyMiddleware) for m in middlewares):
            raise ValueError('All arguments of Chain except the first must be instances of StrategyMiddleware.')
        self._strategy = strategy
        self._middlewares = list(middlewares)
        super().__init__()

    def __getitem__(self, index: int) -> Strategy | StrategyMiddleware:
        if not 0 <= index < len(self._middlewares) + 1:
            raise ValueError(f'index must be in the range of [0, {len(self._middlewares) + 1}), got {index}')
        if index == 0:
            return self._strategy
        else:
            return self._middlewares[index - 1]

    def _initialize(self, model_space: ExecutableModelSpace, engine: ExecutionEngine) -> ExecutableModelSpace:
        """Initialize a strategy chain, which includes the following steps:

        1. calling :meth:`StrategyMiddleware.set_engine` from bottom to top.
        2. initialize the main strategy.
        3. calling :meth:`StrategyMiddleware._initialize_model_space` from top to bottom.
        """
        for cur, nex in list(zip(self._middlewares, cast(List[ExecutionEngine], self._middlewares[1:]) + [engine]))[::-1]:
            cur.set_engine(nex)

        model_space = self._strategy.initialize(model_space, self._middlewares[0])

        for middleware in self._middlewares:
            model_space = middleware._initialize_model_space(model_space)

        return model_space

    def run(self) -> None:
        _logger.debug('Calling run of the main strategy: %s', self._strategy)
        try:
            return self._strategy.run()
        finally:
            # Sync the status
            self._status = self._strategy._status

            # Strategy has already cleaned up. Never mind it.
            # Cleanup the middlewares, in the reverse order of initialization.
            for middleware in self._middlewares:
                try:
                    middleware._cleanup()
                except:
                    _logger.exception('Error when calling unset_engine of %s. Ignore.', middleware)

    def list_models(self, sort: bool = True, limit: int | None = None) -> Iterator[ExecutableModelSpace]:
        """List the models.

        Note that ``sort = True`` by default will filter out unsuccessful models with no metrics.
        Turn it to false if you are interested in the full running history.

        See Also
        --------
        nni.nas.strategy.base.Strategy.list_models
        """
        return self._strategy.list_models(sort=sort, limit=limit)

    def load_state_dict(self, state_dict: dict) -> None:
        self._strategy.load_state_dict(state_dict['strategy'])
        for i, middleware in enumerate(self._middlewares):
            middleware.load_state_dict(state_dict['middlewares'][i])

    def state_dict(self) -> dict:
        return {
            'strategy': self._strategy.state_dict(),
            'middlewares': [middleware.state_dict() for middleware in self._middlewares]
        }

    def extra_repr(self):
        return '\n' + ',\n'.join([
            '  ' + repr(s) for s in cast(List[Any], [self._strategy]) + cast(List[Any], self._middlewares)
        ]) + '\n'


class StrategyMiddleware(Middleware):
    """:class:`StrategyMiddleware` intercepts the models,
    and *strategically* filters, mutates, or replicates them and submits them to the engine.
    It can also intercept the metrics reported by the engine, and manipulates them.

    :class:`StrategyMiddleware` is often used together with :class:`~nni.nas.strategy.Chain`,
    which chains a main strategy and a list of middlewares.
    When a model is created by the main strategy, it is passed to the middlewares in order,
    during which each middleware have access to the model, and pass it to the next middleware.
    The metric does quite the opposite, i.e., it is passed from the engine, through all the middlewares,
    and all the way back to the main strategy.

    We refer to the middleware closest to the main strategy as **upper-level middleware**,
    as it exists at the upper level of the calling stack.
    Conversely, we refer to the middleware closest to the engine as **lower-level middleware**.
    """

    def __init__(self, model_space: ExecutableModelSpace | None = None, engine: ExecutionEngine | None = None) -> None:
        super().__init__(engine)
        self._model_space: ExecutableModelSpace | None = model_space

        if engine is not None:
            self.set_engine(engine)
        if model_space is not None:
            self._initialize_model_space(model_space)

    @property
    def model_space(self) -> ExecutableModelSpace:
        """Model space is useful for the middleware to do advanced things, e.g., sample its own models.

        The model space is set by whoever uses the middleware, before the strategy starts to run.
        """
        if self._model_space is None:
            raise RuntimeError("Strategy middleware is not attached to a model space.")
        return self._model_space

    def set_engine(self, engine: ExecutionEngine) -> None:
        """Calls :meth:`_initialize_engine`.

        Subclass should NOT override this.
        They can encouraged to override :meth:`_initialize_engine` instead, for simplicity.
        """
        super().set_engine(engine)
        self._initialize_engine(engine)

    def _initialize_engine(self, engine: ExecutionEngine) -> None:
        """Initialize the underlying engine.

        Register some callbacks here if needed.
        """
        _logger.debug('Middleware %r initialized with engine: %r', self, engine)

    def _initialize_model_space(self, model_space: ExecutableModelSpace) -> ExecutableModelSpace:
        """Set the model space (and possibly mutate the model space).

        This method is called for each middleware / strategy, from the upper-level to lower-level.
        So lower-level will see the mutated model space by the upper-level middleware.
        """
        _logger.debug('Middleware %r initialized with model space: %r', self, model_space)
        self._model_space = model_space
        return model_space

    def _cleanup(self) -> None:
        """Subclass override this (instead of :meth:`shutdown`) to detach the callbacks.

        Semantically similar to :meth:`~nni.nas.strategy.base.Strategy._cleanup`.
        """
        _logger.debug('Middleware %r cleaned up.', self)

    # The following method has default implementation, because few middleware will need to override them.

    def idle_worker_available(self) -> bool:
        return self.engine.idle_worker_available()

    def budget_available(self) -> bool:
        return self.engine.budget_available()

    def shutdown(self) -> None:
        raise RuntimeError('Shutdown should never be called for :class:`StrategyMiddleware`.')


class Filter(StrategyMiddleware):
    """
    Block certain models from submitting.

    When models are submitted, they will go through the filter function, to check their validity.
    If the function returns true, the model will be submitted as usual.
    Otherwise, the model will be immediately marked as invalid (and optionally have a metric to penalize the strategy).

    We recommend to use this middleware to check certain constraints,
    or prevent the training of some bad models from happening.

    Parameters
    ----------
    filter_fn
        The filter function. Input argument is a :class:`~nni.nas.space.ExecutableModelSpace`.
        Returning ``True`` means the model is good to submit.
    metric_for_invalid
        When setting to be not None, the metric will be assigned to invalid models.
        Otherwise, no metric will be set.
    patience
        Number of continuous invalid models received until the middleware reports no budget.
    retain_history
        To faithfully record all the submitted models including the invalid ones.
        Setting this to false would lose the record of the invalid models, but will also be more memory-efficient.
        Note that the history can NOT be recovered upon :meth:`load_state_dict`.

    Examples
    --------
    With :class:`Filter`, it becomes a lot easier to have some customized controls for the built-in strategies.

    For example, if I have a fancy estimator that can tell whether a model's accuracy is above 90%,
    and I don't want any model below 90% submitted for training, I can do::

        def some_fancy_estimator(model) -> bool:
            # return True or False
            ...

        strategy = Chain(
            RegularizedEvolution(),
            Filter(some_fancy_estimator)
        )

    If the estimator returns false, the model will be immediately marked as invalid, and will not run.
    """

    def __init__(self, filter_fn: Callable[[ExecutableModelSpace], bool],
                 metric_for_invalid: TrialMetric | None = None,
                 patience: int = 1000,
                 retain_history: bool = True) -> None:
        super().__init__()
        self.filter_fn = filter_fn
        self.metric_for_invalid = metric_for_invalid
        self.patience = patience
        self.retain_history = retain_history

        self._filter_count_patience = 0
        self._filter_count_total = 0
        self._filter_models: list[ExecutableModelSpace] = []

    def extra_repr(self) -> str:
        rv = f'filter_fn={self.filter_fn}'
        if self.metric_for_invalid is not None:
            rv += f', metric_for_invalid={self.metric_for_invalid}'
        return rv

    def load_state_dict(self, state_dict: dict) -> None:
        self._filter_count_patience = state_dict['filter_count_patience']
        self._filter_count_total = state_dict['filter_count_total']
        if self.retain_history and self._filter_count_total:
            _logger.warning('Loading state for Filter does not recover previous filtered model history.')

    def state_dict(self) -> dict:
        return {
            'filter_count_patience': self._filter_count_patience,
            'filter_count_total': self._filter_count_total
        }

    def submit_models(self, *models: ExecutableModelSpace) -> None:
        to_submit = []
        for model in models:
            if self.filter_fn(model):
                # Valid
                self._filter_count_patience = 0
                to_submit.append(model)
            else:
                # Invalid
                if self.retain_history:
                    self._filter_models.append(model)

                self._filter_count_patience += 1
                self._filter_count_total += 1
                _logger.debug('Invalid model filtered out: %s', model)

                # Send a metric event to the strategy.
                if self.metric_for_invalid is not None:
                    self.dispatch_model_event(ModelEventType.FinalMetric, metric=self.metric_for_invalid, model=model)

                # The model finishes "training".
                self.dispatch_model_event(ModelEventType.TrainingEnd, status=ModelStatus.Invalid, model=model)

        if to_submit:
            self.engine.submit_models(*to_submit)

    def budget_available(self) -> bool:
        if self._filter_count_patience >= self.patience:
            _logger.info('Too many invalid models. Should stop.')
            return False
        return self.engine.budget_available()

    def list_models(self, status: ModelStatus | None = None) -> Iterable[ExecutableModelSpace]:
        if self._filter_count_total > 0 and not self.retain_history:
            warnings.warn('Filter middleware currently does not list models that are invalid when `retain_history` is set to False.')

        yield from self.engine.list_models(status)

        if self.retain_history:
            for model in self._filter_models:
                if status is None or model.status == status:
                    yield model

    def register_model_event_callback(self, event_type: ModelEventType, callback: Callable[[ModelEvent], None]) -> None:
        super().register_model_event_callback(event_type, callback)
        self.engine.register_model_event_callback(event_type, callback)

    def unregister_model_event_callback(self, event_type: ModelEventType, callback: Callable[[ModelEvent], None]) -> None:
        super().register_model_event_callback(event_type, callback)
        self.engine.unregister_model_event_callback(event_type, callback)


class Deduplication(StrategyMiddleware):
    """
    This middleware is able to deduplicate models that are submitted by strategies.

    When duplicated models are found, the middleware can be configured to,
    either mark the model as invalid, or find the metric of the model from history and "replay" the metrics.
    Regardless of which action is taken, the patience counter will always increase, and when it runs out,
    the middleware will say there is no more budget.

    Notice that some strategies have already provided deduplication on their own, e.g., :class:`~nni.nas.strategy.Random`.
    This class is to help those strategies who do NOT have the ability of deduplication.

    Parameters
    ----------
    action
        What to do when a duplicated model is found.
        ``invalid`` means to mark the model as invalid,
        while ``replay`` means to retrieve the metric of the previous same model from the engine.
    patience
        Number of continuous duplicated models received until the middleware reports no budget.
    retain_history
        To record all the duplicated models even if there are not submitted to the underlying engine.
        While turning this off might lose part of the submitted model history,
        it will also reduce the memory cost.
    """

    def __init__(self, action: Literal['invalid', 'replay'], patience: int = 1000, retain_history: bool = True) -> None:
        super().__init__()
        self.patience = patience
        self.action = action
        self.retain_history = retain_history

        self._dedup_helper = DeduplicationHelper()

        self._dup_count_patience = 0
        self._dup_count_total = 0
        self._dup_models: list[ExecutableModelSpace] = []

    def extra_repr(self) -> str:
        return f'action={self.action}'

    def load_state_dict(self, state_dict: dict) -> None:
        self._dedup_helper.load_state_dict(state_dict)
        self._dup_count_patience = state_dict['dup_count_patience']
        self._dup_count_total = state_dict['dup_count_total']
        if self.retain_history and self._dup_count_total:
            _logger.warning('Loading state for Deduplication does not recover previous deduplicated model history.')

    def state_dict(self) -> dict:
        return {
            'dup_count_patience': self._dup_count_patience,
            'dup_count_total': self._dup_count_total,
            **self._dedup_helper.state_dict()
        }

    def submit_models(self, *models: ExecutableModelSpace) -> None:
        to_submit = []
        for model in models:
            if self._dedup_helper.dedup(model.sample):
                # New model, submit.
                self._dup_count_patience = 0
                to_submit.append(model)
            else:
                _logger.debug('Duplicated model found: %s', model)
                if not self.handle_duplicate_model(model):
                    _logger.warning('Model is believed to be a duplicate but we did not find a same model in the history. '
                                    'The model will be treated as a new model: %s', model)
                    # But we didn't clear out the patience here.
                    # It might be a model from the resumed experiment.
                    to_submit.append(model)
                else:
                    if self.retain_history:
                        self._dup_models.append(model)
                    self._dup_count_patience += 1
                    self._dup_count_total += 1

        if to_submit:
            self.engine.submit_models(*to_submit)

    def list_models(self, status: ModelStatus | None = None) -> Iterable[ExecutableModelSpace]:
        if self._dup_count_total > 0 and not self.retain_history:
            warnings.warn('Filter middleware currently does not list models that are invalid.')

        yield from self.engine.list_models(status)

        if self.retain_history:
            for model in self._dup_models:
                if status is None or model.status == status:
                    yield model

    def handle_duplicate_model(self, model: ExecutableModelSpace) -> bool:
        if self.action == 'invalid':
            self.dispatch_model_event(ModelEventType.TrainingEnd, status=ModelStatus.Invalid, model=model)

        elif self.action == 'replay':
            found = False
            for retrieved_model in self.engine.list_models():
                if retrieved_model.sample == model.sample:
                    found = True
                    _logger.debug('Using metrics of retrieved model as the submitted model is a duplicate: %s', retrieved_model)

                    # Using the models stored metrics to create events for re-dispatch.
                    for metric in retrieved_model.metrics.intermediates:
                        self.dispatch_model_event(ModelEventType.IntermediateMetric, metric=metric, model=model)
                    self.dispatch_model_event(ModelEventType.FinalMetric, metric=retrieved_model.metric, model=model)
                    self.dispatch_model_event(ModelEventType.TrainingEnd, status=retrieved_model.status, model=model)
                    break

            if not found:
                _logger.warning('No previous model found in the engine history that matches the submitted model. This might be a bug. '
                                'This model will be submitted anyway.')
                return False

        else:
            raise ValueError('Invalid action: %s' % self.action)

        return True

    def budget_available(self) -> bool:
        if self._dup_count_patience >= self.patience:
            _logger.info('Too many duplicated models. Should stop.')
            return False
        return self.engine.budget_available()

    def register_model_event_callback(self, event_type: ModelEventType, callback: Callable[[ModelEvent], None]) -> None:
        super().register_model_event_callback(event_type, callback)
        self.engine.register_model_event_callback(event_type, callback)

    def unregister_model_event_callback(self, event_type: ModelEventType, callback: Callable[[ModelEvent], None]) -> None:
        super().unregister_model_event_callback(event_type, callback)
        self.engine.unregister_model_event_callback(event_type, callback)


class FailureHandler(StrategyMiddleware):
    """
    This middleware handles failed models.

    The handler supports two modes:

    - Retry mode: to re-submit the model to the engine, until the model succeeds or patience runs out.
    - Metric mode: to send a metric for the model, so that the strategy gets penalized for generating this model.

    "Failure" doesn't necessarily mean it has to be the "Failed" state.
    It can be other types such as "Invalid", or "Interrupted", etc.
    The middleware can thus be chained with other middlewares (e.g., :class:`Filter`),
    to retry (or put metrics) on invalid models::

        strategy = Chain(
            RegularizedEvolution(),
            FailureHandler(metric=-1.0, failure_types=(ModelStatus.Invalid, )),
            Filter(filter_fn=custom_constraint)
        )

    Parameters
    ----------
    metric
        The metric to send when the model failed. Implies metric mode.
    retry_patience
        Maximum number times of retires. Implies retry mode.
        ``metric`` and ``retry_patience`` can't be both set and can't be both unset.
        Exactly one of them must be set.
    failure_types
        A tuple of :class:`~nni.nas.space.ModelStatus`,
        indicating a set of status that are considered failure.
    retain_history
        Only has effect in retry mode.
        If set to ``True``, submitted models will be kept in a dedicated place, separated from retried models.
        Otherwise, :meth:`list_models` might return both submitted models and retried models.
    """

    def __init__(self, *, metric: TrialMetric | None = None, retry_patience: int | None = None,
                 failure_types: tuple[ModelStatus, ...] = (ModelStatus.Failed,),
                 retain_history: bool = True) -> None:
        super().__init__()
        self.metric = metric
        self.retry_patience = retry_patience
        self.retain_history = retain_history

        if self.metric is not None and self.retry_patience is not None:
            raise ValueError('Only one of metric and retry_patience can be specified.')
        elif self.metric is not None:
            self.action: Literal['metric', 'retry'] = 'metric'
        elif self.retry_patience is not None:
            self.action: Literal['metric', 'retry'] = 'retry'
        else:
            raise ValueError('Either metric or retry_patience must be specified.')

        if not isinstance(failure_types, tuple):
            raise TypeError('failure_types must be a tuple of ModelStatus.')
        self.failure_types = failure_types

        self._retry_count: dict[ExecutableModelSpace, int] = defaultdict(int)
        self._history: list[ExecutableModelSpace] = []

    def extra_repr(self) -> str:
        return f'action={self.action!r}, metric={self.metric}, retry_patience={self.retry_patience}'

    def load_state_dict(self, state_dict: dict) -> None:
        _logger.warning('Loading state for FailureHandler will not recover any states.')

    def state_dict(self) -> dict:
        return {}

    def _initialize_engine(self, engine: ExecutionEngine) -> None:
        engine.register_model_event_callback(ModelEventType.TrainingEnd, self.handle_failure)

    def _cleanup(self) -> None:
        self.engine.unregister_model_event_callback(ModelEventType.TrainingEnd, self.handle_failure)

    def submit_models(self, *models: ExecutableModelSpace) -> None:
        if self.retain_history:
            self._history.extend(models)
        return self.engine.submit_models(*models)

    def list_models(self, status: ModelStatus | None = None) -> Iterable[ExecutableModelSpace]:
        if self.retain_history:
            for model in self._history:
                if status is None or model.status == status:
                    yield model
        else:
            yield from self.engine.list_models(status=status)

    def handle_failure(self, event: TrainingEndEvent) -> None:
        """Handle a training end event. Do something if the model is failed.

        This callback only works if it's registered before other callbacks.
        In practice, it is, because the middlewares call `set_engine` bottom-up.
        """
        if event.status not in self.failure_types:
            if event.status != ModelStatus.Trained:
                _logger.warning('%s reports status %s. This is not a failure type.', event.model, event.status)
            return

        if self.action == 'metric':
            assert self.metric is not None

            if event.model.metric is not None:
                _logger.warning('%s failed, but it has a metric. Will send another metric of %f anyway.',
                                event.model, self.metric)
            self.dispatch_model_event(ModelEventType.FinalMetric, metric=self.metric, model=event.model)

        elif self.action == 'retry':
            assert self.retry_patience is not None

            if self._retry_count[event.model] >= self.retry_patience:
                _logger.info('%s failed %d times. Will not retry it any more. Mark as failure.',
                             event.model, self._retry_count[event.model])

            elif self.engine.budget_available():  # TODO: It'd be better to check idle worker and lock the resource here.
                self._retry_count[event.model] += 1
                _logger.debug('%s failed. Retrying. Attempt: %d', event.model, self._retry_count[event.model])

                # Maybe we should emit an event here?
                event.model.status = ModelStatus.Retrying

                # Clear its metrics.
                event.model.metrics.clear()

                # The rest of the callbacks shouldn't receive the event,
                # because for them, training didn't end.
                event.stop_propagation()

                self.engine.submit_models(event.model)

            else:
                _logger.info('Budget exhausted. Stop retrying although failed.')

    def register_model_event_callback(self, event_type: ModelEventType, callback: Callable[[ModelEvent], None]) -> None:
        super().register_model_event_callback(event_type, callback)
        self.engine.register_model_event_callback(event_type, callback)

    def unregister_model_event_callback(self, event_type: ModelEventType, callback: Callable[[ModelEvent], None]) -> None:
        super().unregister_model_event_callback(event_type, callback)
        self.engine.unregister_model_event_callback(event_type, callback)


class MultipleEvaluation(StrategyMiddleware):
    """
    Runs each model for multiple times, and use the averaged metric as the final result.

    This is useful in scenarios where model evaluation is unstable, with randomness (e.g., Reinforcement Learning).

    When models are submitted, replicas of the models will be created (via deepcopy). See :meth:`submit_models`.
    The intermediate metrics, final metric, as well as status will be reduced in their arriving order.
    For example, the first intermediate metric reported by all replicas will be gathered and averaged,
    to be the first intermediate metric of the submitted original model.
    Similar for final metric and status.
    The status is only considered successful when all the replicas have a successful status.
    Otherwise, the first unsuccessful status of replicas will be used as the status of the original model.

    Parameters
    ----------
    repeat
        How many times to evaluate each model.
    retain_history
        If ``True``, keep all the submitted original models in memory.
        Otherwise :meth:`list_models` will return the replicated models, which, on the other hand saves some memory.
    """

    def __init__(self, repeat: int, retain_history: bool = True) -> None:
        super().__init__()
        if repeat <= 1:
            raise ValueError('repeat must be greater than 1.')
        self.repeat = repeat
        self.retain_history = retain_history

        self._history: list[ExecutableModelSpace] = []
        self._model_to_repeats: dict[ExecutableModelSpace, list[ExecutableModelSpace]] = {}
        self._inverse_model_to_repeats: dict[ExecutableModelSpace, tuple[ExecutableModelSpace, int]] = {}

        # We don't send events until all the replicas of the same model all receive events.
        # (this is like the Tetris game.)
        def factory(): return [deque() for _ in range(self.repeat)]

        self._model_repeats_intermediate: dict[ExecutableModelSpace, list[deque[TrialMetric]]] = defaultdict(factory)
        self._model_repeats_final: dict[ExecutableModelSpace, list[deque[TrialMetric]]] = defaultdict(factory)
        self._model_repeats_status: dict[ExecutableModelSpace, list[deque[ModelStatus]]] = defaultdict(factory)

    def extra_repr(self) -> str:
        return f'repeat={self.repeat}'

    def load_state_dict(self, state_dict: dict) -> None:
        _logger.warning('Loading state for MultipleEvaluation will not recover any states. '
                        'Unfinished repeated evaluations as well as their original models are lost.')

    def state_dict(self) -> dict:
        return {}

    def _initialize_engine(self, engine: ExecutionEngine) -> None:
        engine.register_model_event_callback(ModelEventType.IntermediateMetric, self.handle_intermediate_metric)
        engine.register_model_event_callback(ModelEventType.FinalMetric, self.handle_final_metric)
        engine.register_model_event_callback(ModelEventType.TrainingEnd, self.handle_training_end)

    def _cleanup(self) -> None:
        self.engine.unregister_model_event_callback(ModelEventType.IntermediateMetric, self.handle_intermediate_metric)
        self.engine.unregister_model_event_callback(ModelEventType.FinalMetric, self.handle_final_metric)
        self.engine.unregister_model_event_callback(ModelEventType.TrainingEnd, self.handle_training_end)

    def submit_models(self, *models: ExecutableModelSpace) -> None:
        """Submit the models.

        The method will replicate the models by :attr:`repeat` number of times.
        If multiple models are submitted simultaneously, the models will be submitted replica by replica.
        For example, three models are submitted and they are repeated two times,
        the submitting order will be:
        model1, model2, model3, model1, model2, model3.

        Warnings
        --------
        - This method might exceed the budget of the underlying engine,
          even if the budget shows available when the strategy submits.
        - This method will ignore a model if the model's replicas is current running.
        """
        deduped_models = []

        for model in models:
            if model in self._model_to_repeats:
                _logger.warning('%s is already repeatedly running. New submission of the same model will be ignored.', model)
                # TODO: We can't submit the model directly. Otherwise the model's events will be very messy to handle.
                #       Probably another refactor is needed to deal with the "same model, different instance" problem.
            else:
                self._model_to_repeats[model] = [copy.deepcopy(model) for _ in range(self.repeat)]
                for i, m in enumerate(self._model_to_repeats[model]):
                    self._inverse_model_to_repeats[m] = (model, i)
                deduped_models.append(model)
                if self.retain_history:
                    self._history.append(model)

        if not deduped_models:
            return

        for i in range(self.repeat):
            _logger.debug('Submitting models for %d times. %d/%d...', self.repeat, i + 1, self.repeat)
            # TODO: check budget?
            self.engine.submit_models(*[self._model_to_repeats[model][i] for model in deduped_models])

    def list_models(self, status: ModelStatus | None = None) -> Iterable[ExecutableModelSpace]:
        if not self.retain_history:
            warnings.warn('MultipleEvaluation.list_models returns the replicated models when history is not retained.')
            yield from self.engine.list_models(status)
        else:
            for model in self._history:
                if status is None or model.status == status:
                    yield model

    def _average(self, collections: list[Any]) -> Any:
        if isinstance(collections[0], (int, float)):
            return sum(collections) / len(collections)
        elif isinstance(collections[0], dict):
            return {k: self._average([c[k] for c in collections]) for k in collections[0]}
        elif isinstance(collections[0], list):
            return [self._average([c[i] for c in collections]) for i in range(len(collections[0]))]
        else:
            raise ValueError(f'Cannot average {type(collections[0])}: {collections}')

    def handle_intermediate_metric(self, event: IntermediateMetricEvent) -> None:
        if event.model not in self._inverse_model_to_repeats:
            _logger.warning('%s is not in the list of models that are repeated. Ignoring it.', event.model)
            return

        original_model, repeat_index = self._inverse_model_to_repeats[event.model]

        # Put one block at `repeat_index`.
        self._model_repeats_intermediate[original_model][repeat_index].append(event.metric)

        # Let's cancel the Tetris.
        while all(len(q) > 0 for q in self._model_repeats_intermediate[original_model]):
            metrics = [q.popleft() for q in self._model_repeats_intermediate[original_model]]
            averaged = self._average(metrics)
            _logger.debug('Repeat middleware reduced intermediate metrics %s to %s.', metrics, averaged)
            self.dispatch_model_event(ModelEventType.IntermediateMetric, metric=averaged, model=original_model)

    # Same for final metric and status.
    def handle_final_metric(self, event: FinalMetricEvent) -> None:
        if event.model not in self._inverse_model_to_repeats:
            _logger.warning('%s is not in the list of models that are repeated. Ignoring it.', event.model)
            return

        original_model, repeat_index = self._inverse_model_to_repeats[event.model]
        self._model_repeats_final[original_model][repeat_index].append(event.metric)

        # There can be more than one final metric. Put them into queue.
        while all(len(q) > 0 for q in self._model_repeats_final[original_model]):
            metrics = [q.popleft() for q in self._model_repeats_final[original_model]]
            averaged = self._average(metrics)
            _logger.debug('Repeat middleware reduced final metric %s to %s.', metrics, averaged)
            self.dispatch_model_event(ModelEventType.FinalMetric, metric=averaged, model=original_model)

    def handle_training_end(self, event: TrainingEndEvent) -> None:
        if event.model not in self._inverse_model_to_repeats:
            _logger.warning('%s is not in the list of models that are repeated. Ignoring it.', event.model)
            return

        original_model, repeat_index = self._inverse_model_to_repeats[event.model]
        self._model_repeats_status[original_model][repeat_index].append(event.status)

        # If here because one model should have at most one ending status.
        if all(len(q) > 0 for q in self._model_repeats_status[original_model]):
            statuses = [q.popleft() for q in self._model_repeats_status[original_model]]

            if all(status == ModelStatus.Trained for status in statuses):
                # Successful is only when all the repeats are successful.
                status = ModelStatus.Trained
            else:
                # Otherwise find the first unsuccessful status.
                status = [status for status in statuses if status != ModelStatus.Trained][0]

            _logger.debug('Repeat middleware reduced status %s to %s.', statuses, status)
            self.dispatch_model_event(ModelEventType.TrainingEnd, status=status, model=original_model)

            self._clear_model(original_model)

    def _clear_model(self, model: ExecutableModelSpace):
        # Validation check. See if there are any more metrics.
        if any(len(q) > 0 for q in self._model_repeats_intermediate[model]):
            _logger.warning('Intermediate metrics are not empty after training end. Ignoring them: %s',
                            self._model_repeats_intermediate[model])
        if any(len(q) > 0 for q in self._model_repeats_final[model]):
            _logger.warning('Final metrics are not empty after training end. Ignoring them: %s',
                            self._model_repeats_final[model])
        if any(len(q) > 0 for q in self._model_repeats_status[model]):
            _logger.warning('Got left statuses after training end. This should not happen: %s',
                            self._model_repeats_status[model])

        # Cleanup.
        self._model_repeats_intermediate.pop(model)
        self._model_repeats_final.pop(model)
        self._model_repeats_status.pop(model)

        for m in self._model_to_repeats[model]:
            self._inverse_model_to_repeats.pop(m)
        self._model_to_repeats.pop(model)


class MedianStop(StrategyMiddleware):
    """Kill a running model when its best intermediate result so far is worse than
    the median of results of all completed models at the same number of intermediate reports.

    Follow the mechanism in :class:`~nni.hpo.medianstop_assessor.MedianstopAssessor` to stop trials.

    Warnings
    --------
    This only works theoretically.
    It can't be used because engine doesn't have the ability to kill a model currently.
    """

    def __init__(self) -> None:
        # Map from intermediate sequence ID to a list of intermediate results from all trials.
        self._intermediates_history: dict[int, list[float]] = defaultdict(list)
        _logger.warning('MedianStop is not tested against current engine implementations.')

    def load_state_dict(self, state_dict: dict) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def _initialize_engine(self, engine: ExecutionEngine) -> None:
        engine.register_model_event_callback(ModelEventType.IntermediateMetric, self.handle_intermediate)
        engine.register_model_event_callback(ModelEventType.TrainingEnd, self.handle_training_end)

    def handle_intermediate(self, event: IntermediateMetricEvent) -> None:
        # Get the metric of myself.
        best_in_history = max(event.metric, max(event.model.metrics.intermediates))

        # Get the median of all other trials.
        index = len(event.model.metrics.intermediates)
        median = np.median(self._intermediates_history[index])
        if best_in_history < median:
            _logger.info('%s is worse than median of all completed models. MedianStop will kill it.', event.model)
            self.engine.kill_model(event.model)  # type: ignore  # pylint: disable=no-member

    def handle_training_end(self, event: TrainingEndEvent) -> None:
        if event.status != ModelStatus.Trained:
            _logger.info('%s is not successfully trained. MedianStop will not consider it.', event.model)
            return

        for intermediate_id, intermediate_value in enumerate(event.model.metrics.intermediates):
            self._intermediates_history[intermediate_id].append(intermediate_value)
