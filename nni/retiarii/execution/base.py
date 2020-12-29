import logging
from typing import Dict, Any, List

from .interface import AbstractExecutionEngine, AbstractGraphListener, WorkerInfo
from .. import codegen, utils
from ..graph import Model, ModelStatus, MetricData
from ..integration import send_trial, receive_trial_parameters, get_advisor

_logger = logging.getLogger(__name__)

class BaseGraphData:
    def __init__(self, model_script: str, training_module: str, training_kwargs: Dict[str, Any]) -> None:
        self.model_script = model_script
        self.training_module = training_module
        self.training_kwargs = training_kwargs

    def dump(self) -> dict:
        return {
            'model_script': self.model_script,
            'training_module': self.training_module,
            'training_kwargs': self.training_kwargs
        }

    @staticmethod
    def load(data):
        return BaseGraphData(data['model_script'], data['training_module'], data['training_kwargs'])


class BaseExecutionEngine(AbstractExecutionEngine):
    """
    The execution engine with no optimization at all.
    Resource management is yet to be implemented.
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

    def submit_models(self, *models: Model) -> None:
        for model in models:
            data = BaseGraphData(codegen.model_to_pytorch_script(model),
                                 model.training_config.module, model.training_config.kwargs)
            self._running_models[send_trial(data.dump())] = model

    def register_graph_listener(self, listener: AbstractGraphListener) -> None:
        self._listeners.append(listener)

    def _send_trial_callback(self, paramater: dict) -> None:
        for listener in self._listeners:
            _logger.warning('resources: %s', listener.resources)
            if not listener.has_available_resource():
                _logger.warning('There is no available resource, but trial is submitted.')
            listener.on_resource_used(1)
            _logger.warning('on_resource_used: %s', listener.resources)

    def _request_trial_jobs_callback(self, num_trials: int) -> None:
        for listener in self._listeners:
            listener.on_resource_available(1 * num_trials)
            _logger.warning('on_resource_available: %s', listener.resources)

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

    def query_available_resource(self) -> List[WorkerInfo]:
        raise NotImplementedError  # move the method from listener to here?

    @classmethod
    def trial_execute_graph(cls) -> None:
        """
        Initialize the model, hand it over to trainer.
        """
        graph_data = BaseGraphData.load(receive_trial_parameters())
        with open('_generated_model.py', 'w') as f:
            f.write(graph_data.model_script)
        trainer_cls = utils.import_(graph_data.training_module)
        model_cls = utils.import_('_generated_model._model')
        trainer_instance = trainer_cls(model=model_cls(), **graph_data.training_kwargs)
        trainer_instance.fit()
