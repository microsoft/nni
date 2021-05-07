# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod, abstractclassmethod
from typing import Any, Iterable, NewType, List, Union

from ..graph import Model, MetricData

__all__ = [
    'GraphData', 'WorkerInfo',
    'AbstractGraphListener', 'AbstractExecutionEngine'
]


GraphData = NewType('GraphData', Any)
"""
A _serializable_ internal data type defined by execution engine.

Execution engine will submit this kind of data through NNI to worker machine, and train it there.

A `GraphData` object describes a (merged) executable graph.

This is trial's "hyper-parameter" in NNI's term and will be transfered in JSON format.

See `AbstractExecutionEngine` for details.
"""


WorkerInfo = NewType('WorkerInfo', Any)
"""
To be designed.  Discussion needed.

This describes the properties of a worker machine. (e.g. memory size)
"""


class AbstractGraphListener(ABC):
    """
    Abstract listener interface to receive graph events.

    Use `AbstractExecutionEngine.register_graph_listener()` to activate a listener.
    """

    @abstractmethod
    def on_metric(self, model: Model, metric: MetricData) -> None:
        """
        Reports the final metric of a graph.
        """
        raise NotImplementedError

    @abstractmethod
    def on_intermediate_metric(self, model: Model, metric: MetricData) -> None:
        """
        Reports the latest intermediate metric of a trainning graph.
        """
        pass

    @abstractmethod
    def on_training_end(self, model: Model, success: bool) -> None:
        """
        Reports either a graph is fully trained or the training process has failed.
        """
        pass


class AbstractExecutionEngine(ABC):
    """
    The abstract interface of execution engine.

    Most of these APIs are used by strategy, except `trial_execute_graph`, which is invoked by framework in trial.
    Strategy will get the singleton execution engine object through a global API,
    and use it in either sync or async manner.

    Execution engine is responsible for submitting (maybe-optimized) models to NNI,
    and assigning their metrics to the `Model` object after training.
    Execution engine is also responsible to launch the graph in trial process,
    because it's the only one who understands graph data, or "hyper-parameter" in NNI's term.

    Execution engine will leverage NNI Advisor APIs, which are yet open for discussion.

    In synchronized use case, the strategy will have a loop to call `submit_models` and `wait_models` repeatly,
    and will receive metrics from `Model` attributes.
    Execution engine could assume that strategy will only submit graph when there are availabe resources (for now).

    In asynchronized use case, the strategy will register a listener to receive events,
    while still using `submit_models` to train.

    There will be a `BaseExecutionEngine` subclass.
    Inner-graph optimizing is supposed to derive `BaseExecutionEngine`,
    while overrides `submit_models` and `trial_execute_graph`.
    cross-graph optimizing is supposed to derive `AbstractExectutionEngine` directly,
    because in this case APIs like `wait_graph` and `listener.on_training_end` will have unique logic.

    There might be some util functions benefit all optimizing methods,
    but non-mandatory utils should not be covered in abstract interface.
    """

    @abstractmethod
    def submit_models(self, *models: Model) -> None:
        """
        Submit models to NNI.

        This method is supposed to call something like `nni.Advisor.create_trial_job(graph_data)`.
        """
        raise NotImplementedError

    @abstractmethod
    def list_models(self) -> Iterable[Model]:
        """
        Get all models in submitted.

        Execution engine should store a copy of models that have been submitted and return a list of copies in this method.
        """
        raise NotImplementedError

    @abstractmethod
    def query_available_resource(self) -> Union[List[WorkerInfo], int]:
        """
        Returns information of all idle workers.
        If no details are available, this may returns a list of "empty" objects, reporting the number of idle workers.

        Could be left unimplemented for first iteration.
        """
        raise NotImplementedError

    @abstractmethod
    def budget_exhausted(self) -> bool:
        """
        Check whether user configured max trial number or max execution duration has been reached
        """
        raise NotImplementedError

    @abstractmethod
    def register_graph_listener(self, listener: AbstractGraphListener) -> None:
        """
        Register a listener to receive graph events.

        Could be left unimplemented for first iteration.
        """
        raise NotImplementedError

    @abstractclassmethod
    def trial_execute_graph(cls) -> MetricData:
        """
        Train graph and returns its metrics, in a separate trial process.

        Each call to `nni.Advisor.create_trial_job(graph_data)` will eventually invoke this method.

        Because this method will be invoked in trial process on training platform,
        it has different context from other methods and has no access to global variable or `self`.
        However util APIs like `.utils.experiment_config()` should still be available.
        """
        raise NotImplementedError
