import logging
from typing import List, Dict, Tuple

from .interface import AbstractExecutionEngine, AbstractGraphListener, WorkerInfo
from .. import codegen, utils
from ..graph import Model, ModelStatus, MetricData
from ..integration import send_trial, receive_trial_parameters, get_advisor
from .logical_optimizer.logical_plan import LogicalPlan, PhysicalDevice
from .logical_optimizer.opt_dedup_input import DedupInputOptimizer

from .base import BaseGraphData

_logger = logging.getLogger(__name__)


class CGOExecutionEngine(AbstractExecutionEngine):
    def __init__(self, n_model_per_graph=4) -> None:
        self._listeners: List[AbstractGraphListener] = []
        self._running_models: Dict[int, Model] = dict()
        self.logical_plan_counter = 0
        self.n_model_per_graph = n_model_per_graph
        self._optimizers = [DedupInputOptimizer()]
        self._original_models = {}
        self._original_model_to_multi_model = {}

        # register advisor callbacks
        advisor = get_advisor()
        advisor.send_trial_callback = self._send_trial_callback
        advisor.request_trial_jobs_callback = self._request_trial_jobs_callback
        advisor.trial_end_callback = self._trial_end_callback
        advisor.intermediate_metric_callback = self._intermediate_metric_callback
        advisor.final_metric_callback = self._final_metric_callback

    def add_optimizer(self, opt):
        self._optimizers.append(opt)

    def submit_models(self, *models: List[Model]) -> None:
        _logger.info('%d models are submitted', len(models))
        logical = self._build_logical(models)

        for opt in self._optimizers:
            opt.convert(logical)

        phy_models_and_placements = self._assemble(logical)
        for model, placement, grouped_models in phy_models_and_placements:
            data = BaseGraphData(codegen.model_to_pytorch_script(model, placement=placement),
                                 model.training_config.module, model.training_config.kwargs)
            for m in grouped_models:
                self._original_models[m.model_id] = m
                self._original_model_to_multi_model[m.model_id] = model
            self._running_models[send_trial(data.dump())] = model

        # for model in models:
        #     data = BaseGraphData(codegen.model_to_pytorch_script(model),
        #                          model.config['trainer_module'], model.config['trainer_kwargs'])
        #     self._running_models[send_trial(data.dump())] = model

    def _assemble(self, logical_plan: LogicalPlan) -> List[Tuple[Model, PhysicalDevice]]:
        # unique_models = set()
        # for node in logical_plan.graph.nodes:
        #     if node.graph.model not in unique_models:
        #         unique_models.add(node.graph.model)
        # return [m for m in unique_models]
        grouped_models: List[Dict[Model, PhysicalDevice]] = AssemblePolicy().group(logical_plan)
        phy_models_and_placements = []
        for multi_model in grouped_models:
            model, model_placement = logical_plan.assemble(multi_model)
            phy_models_and_placements.append((model, model_placement, multi_model.keys()))
        return phy_models_and_placements

    def _build_logical(self, models: List[Model]) -> LogicalPlan:
        logical_plan = LogicalPlan(plan_id=self.logical_plan_counter)
        for model in models:
            logical_plan.add_model(model)
        self.logical_plan_counter += 1
        return logical_plan

    def register_graph_listener(self, listener: AbstractGraphListener) -> None:
        self._listeners.append(listener)

    def _send_trial_callback(self, paramater: dict) -> None:
        for listener in self._listeners:
            listener.on_resource_used(0)  # FIXME: find the real resource id

    def _request_trial_jobs_callback(self, num_trials: int) -> None:
        for listener in self._listeners:
            listener.on_resource_available([0] * num_trials)  # FIXME: find the real resource id

    def _trial_end_callback(self, trial_id: int, success: bool) -> None:
        model = self._running_models[trial_id]
        if success:
            model.status = ModelStatus.Trained
        else:
            model.status = ModelStatus.Failed
        for model_id in self._original_model_to_multi_model:
            if self._original_model_to_multi_model[model_id] == model:
                original_model = self._original_models[model_id]
                if success:
                    original_model.status = ModelStatus.Trained
                else:
                    original_model.status = ModelStatus.Failed
                for listener in self._listeners:
                    listener.on_training_end(original_model, success)

    def _intermediate_metric_callback(self, trial_id: int, metrics: MetricData) -> None:
        # model = self._running_models[trial_id]
        merged_metrics = dict(metrics)
        for model_id in merged_metrics:
            int_model_id = int(model_id)
            self._original_models[int_model_id].intermediate_metrics.append(merged_metrics[model_id])
            # model.intermediate_metrics.append(metrics)
            for listener in self._listeners:
                listener.on_intermediate_metric(self._original_models[int_model_id], merged_metrics[model_id])

    def _final_metric_callback(self, trial_id: int, metrics: MetricData) -> None:
        merged_metrics = dict(metrics)
        for model_id in merged_metrics:
            int_model_id = int(model_id)
            self._original_models[int_model_id].intermediate_metrics.append(merged_metrics[model_id])
            # model.intermediate_metrics.append(metrics)
            for listener in self._listeners:
                listener.on_metric(self._original_models[int_model_id], merged_metrics[model_id])

    def query_available_resource(self) -> List[WorkerInfo]:
        raise NotImplementedError  # move the method from listener to here?

    @classmethod
    def trial_execute_graph(cls) -> None:
        """
        Initialize the model, hand it over to trainer.
        """
        graph_data = BaseGraphData.load(receive_trial_parameters())
        _logger.info('CGO_ENGINE trial parameters received')
        with open('_generated_model.py', 'w') as f:
            f.write(graph_data.model_script)
        # with open('_debug_graph_data.json', 'w') as f:
        #     json.dump(graph_data.dump(), f)
        trainer_cls = utils.import_(graph_data.training_module)
        model_cls = utils.import_(f"_generated_model.{graph_data.training_kwargs['model_cls']}")
        trainer_instance = trainer_cls(model_cls(), graph_data.training_kwargs)
        trainer_instance.fit()


class AssemblePolicy:
    @staticmethod
    def group(logical_plan):
        group_model = {}
        for idx, m in enumerate(logical_plan.models):
            group_model[m] = PhysicalDevice('server', f'cuda:{idx}')
        return [group_model]
