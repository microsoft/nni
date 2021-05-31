# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import random
import string
from typing import Any, Dict, Iterable, List

from .interface import AbstractExecutionEngine, AbstractGraphListener
from .. import codegen, utils
from ..graph import Model, ModelStatus, MetricData, Evaluator
from ..integration_api import send_trial, receive_trial_parameters, get_advisor

_logger = logging.getLogger(__name__)

class BaseGraphData:
    def __init__(self, model_script: str, evaluator: Evaluator) -> None:
        self.model_script = model_script
        self.evaluator = evaluator

    def dump(self) -> dict:
        return {
            'model_script': self.model_script,
            'evaluator': self.evaluator
        }

    @staticmethod
    def load(data) -> 'BaseGraphData':
        return BaseGraphData(data['model_script'], data['evaluator'])


class BaseExecutionEngine(AbstractExecutionEngine):
    """
    The execution engine with no optimization at all.
    Resource management is implemented in this class.
    """

    def __init__(self) -> None:
        """
        Upon initialization, advisor callbacks need to be registered.
        Advisor will call the callbacks when the corresponding event has been triggered.
        Base execution engine will get those callbacks and broadcast them to graph listener.
        """
        self._listeners: List[AbstractGraphListener] = []

        # register advisor callbacks
        advisor = get_advisor()
        advisor.send_trial_callback = self._send_trial_callback
        advisor.request_trial_jobs_callback = self._request_trial_jobs_callback
        advisor.trial_end_callback = self._trial_end_callback
        advisor.intermediate_metric_callback = self._intermediate_metric_callback
        advisor.final_metric_callback = self._final_metric_callback

        self._running_models: Dict[int, Model] = dict()
        self._history: List[Model] = []

        self.resources = 0

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
        advisor = get_advisor()
        return advisor.stopping

    @classmethod
    def pack_model_data(cls, model: Model) -> Any:
        return BaseGraphData(codegen.model_to_pytorch_script(model), model.evaluator)

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
        model_cls = utils.import_(f'_generated_model.{random_str}._model')
        graph_data.evaluator._execute(model_cls)
        os.remove(file_name)
