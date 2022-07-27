# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['BaseGraphData', 'BaseExecutionEngine']

import logging
import os
import random
import string
from typing import Any, Dict, Iterable, List

from nni.experiment import rest

from nni.nas.execution.common import (
    AbstractExecutionEngine, AbstractGraphListener, RetiariiAdvisor, get_mutation_summary,
    Model, ModelStatus, MetricData, Evaluator,
    send_trial, receive_trial_parameters, get_advisor
)
from nni.nas.utils import import_
from .codegen import model_to_pytorch_script

_logger = logging.getLogger(__name__)

class BaseGraphData:
    """
    Data sent between strategy and trial, in graph-based execution engine.

    Attributes
    ----------
    model_script
        code of an instantiated PyTorch model
    evaluator
        training approach for model_script
    mutation_summary
        a dict of all the choices during mutations in the HPO search space format
    """
    def __init__(self, model_script: str, evaluator: Evaluator, mutation_summary: dict) -> None:
        self.model_script = model_script
        self.evaluator = evaluator
        self.mutation_summary = mutation_summary

    def dump(self) -> dict:
        return {
            'model_script': self.model_script,
            # engine needs to call dump here,
            # otherwise, evaluator will become binary
            # also, evaluator can be none in tests
            'evaluator': self.evaluator._dump() if self.evaluator is not None else None,
            'mutation_summary': self.mutation_summary
        }

    @staticmethod
    def load(data) -> 'BaseGraphData':
        return BaseGraphData(data['model_script'], Evaluator._load(data['evaluator']), data['mutation_summary'])


class BaseExecutionEngine(AbstractExecutionEngine):
    """
    The execution engine with no optimization at all.
    Resource management is implemented in this class.
    """

    def __init__(self, rest_port: int | None = None, rest_url_prefix: str | None = None) -> None:
        """
        Upon initialization, advisor callbacks need to be registered.
        Advisor will call the callbacks when the corresponding event has been triggered.
        Base execution engine will get those callbacks and broadcast them to graph listener.

        Parameters
        ----------
        rest_port
            The port of the experiment's rest server
        rest_url_prefix
            The url prefix of the experiment's rest entry
        """
        self.port = rest_port
        self.url_prefix = rest_url_prefix

        self._listeners: List[AbstractGraphListener] = []
        self._running_models: Dict[int, Model] = dict()
        self._history: List[Model] = []

        self.resources = 0

        # register advisor callbacks
        advisor: RetiariiAdvisor = get_advisor()
        advisor.register_callbacks({
            'send_trial': self._send_trial_callback,
            'request_trial_jobs': self._request_trial_jobs_callback,
            'trial_end': self._trial_end_callback,
            'intermediate_metric': self._intermediate_metric_callback,
            'final_metric': self._final_metric_callback
        })

    def submit_models(self, *models: Model) -> None:
        for model in models:
            data = self.pack_model_data(model)
            self._running_models[send_trial(data.dump())] = model
            self._history.append(model)

    def list_models(self) -> Iterable[Model]:
        return self._history

    def register_graph_listener(self, listener: AbstractGraphListener) -> None:
        self._listeners.append(listener)

    def _send_trial_callback(self, paramater: dict) -> None:
        if self.resources <= 0:
            # FIXME: should be a warning message here
            _logger.debug('There is no available resource, but trial is submitted.')
        self.resources -= 1
        _logger.debug('Resource used. Remaining: %d', self.resources)

    def _request_trial_jobs_callback(self, num_trials: int) -> None:
        self.resources += num_trials
        _logger.debug('New resource available. Remaining: %d', self.resources)

    def _trial_end_callback(self, trial_id: int, success: bool) -> None:
        model = self._running_models[trial_id]
        if success:
            model.status = ModelStatus.Trained
        else:
            model.status = ModelStatus.Failed
        for listener in self._listeners:
            listener.on_training_end(model, success)

    def _intermediate_metric_callback(self, trial_id: int, metrics: MetricData) -> None:
        model = self._running_models[trial_id]
        model.intermediate_metrics.append(metrics)
        for listener in self._listeners:
            listener.on_intermediate_metric(model, metrics)

    def _final_metric_callback(self, trial_id: int, metrics: MetricData) -> None:
        model = self._running_models[trial_id]
        model.metric = metrics
        for listener in self._listeners:
            listener.on_metric(model, metrics)

    def query_available_resource(self) -> int:
        return self.resources

    def budget_exhausted(self) -> bool:
        resp = rest.get(self.port, '/check-status', self.url_prefix)
        return resp['status'] == 'DONE'

    @classmethod
    def pack_model_data(cls, model: Model) -> Any:
        mutation_summary = get_mutation_summary(model)
        assert model.evaluator is not None, 'Model evaluator can not be None'
        return BaseGraphData(model_to_pytorch_script(model), model.evaluator, mutation_summary)  # type: ignore

    @classmethod
    def trial_execute_graph(cls) -> None:
        """
        Initialize the model, hand it over to trainer.
        """
        graph_data = BaseGraphData.load(receive_trial_parameters())
        random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        file_name = f'_generated_model/{random_str}.py'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as f:
            f.write(graph_data.model_script)
        model_cls = import_(f'_generated_model.{random_str}._model')
        graph_data.evaluator._execute(model_cls)
        os.remove(file_name)
