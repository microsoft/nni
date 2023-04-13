# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Iterable, NewType, Callable, Type, overload

from nni.nas.space import ExecutableModelSpace, ModelStatus

from .event import ModelEvent, ModelEventType, FinalMetricEvent, IntermediateMetricEvent, TrainingEndEvent

__all__ = [
    'WorkerInfo', 'ExecutionEngine', 'Middleware',
]


WorkerInfo: Type[Any] = NewType('WorkerInfo', Any)
"""
Won't be needed in short-term. Left from v2.x era.

This describes the properties of a worker machine. (e.g. memory size)
"""

_logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    The abstract interface of execution engine.

    Execution engine is responsible for *executing* the submitted models.
    The engine has the freedom to choose the execution environment. For example,
    whether to execute it instantly in the current process, or send it to NNI training service (e.g., local / remote).
    It may also optimize the workloads with techniques like CSE, or even doing benchmark queries.

    Note that some engines might reply on certain model space formats.
    For example, some engines might require the model space to be a graph, to do certain optimizations.

    Every subclass of class:`ExecutableModelSpace` has its general logic (i.e., code) of execution
    defined in its class. But the interpretation of the logic depends on the engine itself.

    In synchronized use case, the strategy will have a loop to call `submit_models` and `wait_models` repeatedly,
    and will receive metrics from `ExecutableModelSpace` attributes.
    Execution engine could assume that strategy will only submit graph when there are available resources (for now).

    In asynchronized use case, the strategy will register a listener to receive events,
    while still using `submit_models` to train.

    There might be some util functions benefit all optimizing methods,
    but non-mandatory utils should not be covered in abstract interface.
    """

    def __init__(self) -> None:
        self._callbacks: dict[ModelEventType, list] = defaultdict(list)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ''

    def wait_models(self, *models: ExecutableModelSpace) -> None:
        """Wait for models to complete training (either success or failure).

        If no models are given, wait for all models to complete.
        """
        if not models:
            model_iterator = self.list_models()
        else:
            model_iterator = models

        while True:
            left_models = [g for g in model_iterator if not g.status.completed()]
            if not left_models:
                break
            time.sleep(1)

    def submit_models(self, *models: ExecutableModelSpace) -> None:
        """
        Submit models to NNI.

        This method is supposed to call something like `nni.Advisor.create_trial_job(graph_data)`.
        """
        raise NotImplementedError()

    def list_models(self, status: ModelStatus | None = None) -> Iterable[ExecutableModelSpace]:
        """
        Get all models submitted.

        If status is presented, only return models with the given status.

        Execution engine should store a copy of models that have been submitted and return a list of copies in this method.
        """
        raise NotImplementedError()

    def idle_worker_available(self) -> bool:
        """
        Return the number of idle workers.
        That is, the recommended number of models to submit currently.

        Strategy can respect / ignore the number. If strategy chooses to ignore,
        the engine doesn't guarantee anything about the newly-submitted model.

        NOTE: The return value was originally designed to be a list of :class:`WorkerInfo` objects.
        If no details are available, this may returns a list of "empty" objects, reporting the number of idle workers.
        However, :class:`WorkerInfo` is almost never used in practice. So we removed it for now to simplify the type-checking.
        """
        raise NotImplementedError()

    def budget_available(self) -> bool:
        """
        Return whether the engine still has available budget.

        Budget could be defined by the number of models, total duration, or energy consumption, etc.

        If the engine has already exhausted the budget, it will not accept any new models.

        NOTE: NNI has no definition of *budget* yet. Therefore this method only returns true or false.
        In future, we might change it to a concrete *budget*.
        """
        raise NotImplementedError()

    def register_model_event_callback(self, event_type: ModelEventType, callback: Callable[..., None]) -> None:
        """
        Register a callback to receive model event.

        Parameters
        ----------
        event_type
            The type of event that is to listen.
        callback
            The callback to receive the event.
            It receives a :class:`~nni.nas.execution.ModelEvent` object, and is expected to return nothing.
        """
        if not isinstance(event_type, ModelEventType):
            event_type = ModelEventType(event_type)
        self._callbacks[event_type].append(callback)

    def unregister_model_event_callback(self, event_type: ModelEventType, callback: Callable[..., None]) -> None:
        """
        Unregister a callback.

        Parameters
        ----------
        event_type
            The type of event that is to listen.
        callback
            The callback to receive the event.
            The event must have been registered before.
        """
        if not isinstance(event_type, ModelEventType):
            event_type = ModelEventType(event_type)
        self._callbacks[event_type].remove(callback)

    @overload
    def dispatch_model_event(self, event: ModelEventType, **kwargs: Any) -> None:
        ...

    @overload
    def dispatch_model_event(self, event: str, **kwargs: Any) -> None:
        ...

    @overload
    def dispatch_model_event(self, event: ModelEvent) -> None:
        ...

    def dispatch_model_event(self, event: ModelEvent | str | ModelEventType, **kwargs: Any) -> None:
        """
        Dispatch a model event to all callbacks. Invoke :meth:`default_callback` at the end.
        This is a utility method for subclass of :class:`ExecutionEngine` to dispatch (emit) events.

        If the engine intends to change the model status / metrics, and also notifies the listeners,
        they are supposed to construct a model event and call :meth:`dispatch_model_event`,
        rather than changing the status of metrics of the model directly.
        Only in this way, the listeners can properly receive the update,
        and even intercept the update before they actually take effect.

        The behavior of :meth:`default_callback` is defined by whoever "dispatches" the event
        (although it has a default implementation).
        """
        if not isinstance(event, ModelEvent):
            if not isinstance(event, ModelEventType):
                event = ModelEventType(event)
            if not isinstance(event, ModelEventType):
                raise TypeError(f'event must be a ModelEvent or a ModelEventType, but got {type(event)}')
            for event_class in ModelEvent.__subclasses__():
                if event_class.event_type == event:
                    event = event_class(**kwargs)
                    break
            else:
                raise ValueError(f'Unknown event type {event}')

        for callback in self._callbacks[event.event_type]:
            if not event._canceled:
                callback(event)
        if not event._canceled and not event._default_canceled:
            self.default_callback(event)

    def default_callback(self, event: ModelEvent) -> None:
        """
        Default callback that is called when a model has a new metric, or a new status.

        This callback is called after all callbacks registered by the user of this engine,
        if it's not canceled.

        The callback implements the most typical behavior of an event:

        - Update the metrics of the model if the event is a metric event.
        - Update the status of the model if the event is a status event.
        """
        if isinstance(event, FinalMetricEvent):
            if event.model.metrics.final is not None:
                _logger.warning(f'Final metric of model {event.model} already exists: {event.model.metrics.final}. '
                                'It will be overwritten.')
            event.model.metrics.final = event.metric
        elif isinstance(event, IntermediateMetricEvent):
            event.model.metrics.add_intermediate(event.metric)
        elif isinstance(event, TrainingEndEvent):
            event.model.status = event.status

    def shutdown(self) -> None:
        """
        Stop the engine.

        The engine will not accept new models, or handle callbacks after being shutdown.
        Anything after :meth:`shutdown` is called is considered undefined behavior.

        Since engine is ephemeral, there is no such thing as ``restart``.
        Creating another engine and load the state dict is encouraged instead.
        """
        raise NotImplementedError('Engine {self.__class__.__name__} did not implement `shutdown`.')

    def state_dict(self) -> dict:
        """Return the state of the engine.

        The state is used to resume the engine.
        """
        raise NotImplementedError(f'Engine {self.__class__.__name__} did not implement `state_dict`.')

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state of the engine.

        Symmetric to :meth:`state_dict`.
        """
        raise NotImplementedError(f'Engine {self.__class__.__name__} did not implement `load_state_dict`.')


class Middleware(ExecutionEngine):
    """
    A middleware that wraps another execution engine.
    It can be used to transform the submitted models before passing to the underlying engine.

    Middlewares sits between a strategy and a engine.
    There could be multiple middlewares chained.
    Some middlewares logically belong to the strategy side, for example model filters and early stopper.
    Others logically belong to the engine side, for example CSE and benchmarking.
    This class is designed mainly for the engine side.
    Strategy side should inherit another dedicated superclass.

    Implementing a middleware is similar to implementing an engine,
    but with the option of leveraging the ability of the underlying wrapped engine.
    Apart from the methods that would otherwise raise NotImplementedError if not implemented,
    we recommend override :meth:`set_engine` and :meth:`register_model_event_callback`.
    In :meth:`set_engine`, the middleware registers some callbacks by itself on the underlying engine,
    while in :meth:`register_model_event_callback`, the middleware decides what to do with the callbacks from the outside.
    There are basically two approaches to handle the callbacks:

    1. Register the callbacks directly on the underlying engine.
       Since callbacks in :meth:`set_engine` are registered before the callbacks from the outside,
       they can intercept the events and manipulates/stops them when needed.
    2. Keep the callbacks to itself.
       Register callbacks written by the middleware itself to the underlying engine,
       which creates brand new events and uses :meth:`dispatch_model_event` to invoke the callbacks from the outside.

    Some other (hacky) approaches might not be possible (e.g., wrap the callbacks with a closure).
    But they are not recommended.

    Middleware should be responsible for unregistering the callbacks at :meth:`shutdown`.

    Parameters
    ----------
    engine
        The underlying execution engine.
    """

    def __init__(self, engine: ExecutionEngine | None = None) -> None:
        super().__init__()
        self._engine: ExecutionEngine | None = None
        if engine is not None:
            self.set_engine(engine)

    @property
    def engine(self) -> ExecutionEngine:
        """The underlying execution engine (or another middleware)."""
        if self._engine is None:
            raise RuntimeError('Underlying engine is not set')
        return self._engine

    def set_engine(self, engine: ExecutionEngine) -> None:
        """
        Override this to do some initialization, e.g., register some callbacks.

        Engine can't be "unset" once set, because middlewares can be only binded once.
        To unregister the callbacks, override :meth:`shutdown`.

        Parameters
        ----------
        engine
            The underlying execution engine.
        """
        _logger.debug('Engine is set for middleware %s: %s', self, engine)
        if self._engine is not None:
            raise ValueError('Engine is already set.')
        self._engine = engine
