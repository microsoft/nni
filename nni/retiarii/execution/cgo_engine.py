# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
from nni.retiarii.evaluator.pytorch.cgo_evaluator import MultiModelSupervisedLearningModule
import os
import random
import string
from typing import Iterable, List, Dict, Tuple

from .interface import AbstractExecutionEngine, AbstractGraphListener, WorkerInfo
from .. import codegen, utils
from ..graph import Model, ModelStatus, MetricData, Node
from ..integration_api import send_trial, receive_trial_parameters, get_advisor
from .logical_optimizer.logical_plan import LogicalPlan, PhysicalDevice, AbstractLogicalNode
from .logical_optimizer.opt_dedup_input import DedupInputOptimizer
from ..evaluator.pytorch.lightning import Lightning

from .base import BaseGraphData

_logger = logging.getLogger(__name__)


class CGOExecutionEngine(AbstractExecutionEngine):
    def __init__(self, available_devices=None) -> None:
        self._listeners: List[AbstractGraphListener] = []
        self._running_models: Dict[int, Model] = dict()
        self.logical_plan_counter = 0
        self.available_devices = available_devices if available_devices else []
        self._optimizers = [DedupInputOptimizer()]
        self._original_models = {}
        self._original_model_to_multi_model = {}
        self._trial_to_original_models = {}

        self.resources = 0

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
            data = BaseGraphData(codegen.model_to_pytorch_script(model, placement=placement), model.evaluator)
            trial_id = send_trial(data.dump())
            self._trial_to_original_models[trial_id] = []
            for m in grouped_models:
                self._original_models[m.model_id] = m
                self._original_model_to_multi_model[m.model_id] = model
                self._trial_to_original_models[trial_id].append(m.model_id)
            self._running_models[trial_id] = model

        # for model in models:
        #     data = BaseGraphData(codegen.model_to_pytorch_script(model),
        #                          model.config['trainer_module'], model.config['trainer_kwargs'])
        #     self._running_models[send_trial(data.dump())] = model

    def list_models(self) -> Iterable[Model]:
        raise NotImplementedError

    def _assemble(self, logical_plan: LogicalPlan) -> List[Tuple[Model, PhysicalDevice, List[Model]]]:
        # unique_models = set()
        # for node in logical_plan.graph.nodes:
        #     if node.graph.model not in unique_models:
        #         unique_models.add(node.graph.model)
        # return [m for m in unique_models]
        grouped_models: List[Dict[Model, PhysicalDevice]] = AssemblePolicy().group(logical_plan, self.available_devices)
        phy_models_and_placements = []
        for multi_model in grouped_models:
            model, model_placement = logical_plan.assemble(multi_model)
            assert(isinstance(model.evaluator, Lightning))
            assert(isinstance(model.evaluator.module, MultiModelSupervisedLearningModule))
            # replace the module with a new instance whose n_models is set
            # n_models must be set in __init__, otherwise it cannot be captured by serialize_cls
            new_module_init_params = model.evaluator.module._init_parameters.copy()
            new_module_init_params['n_models'] = len(multi_model)
            new_module = MultiModelSupervisedLearningModule(**new_module_init_params)
            model.evaluator.module = new_module
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
        if self.resources <= 0:
            _logger.warning('There is no available resource, but trial is submitted.')
        print("_send_trial_callback", paramater)
        # self.resources -= paramater['training_kwargs']['n_model']
        _logger.info('on_resource_used: %d', self.resources)

    def _request_trial_jobs_callback(self, num_trials: int) -> None:
        self.resources += num_trials
        _logger.info('on_resource_available: %d', self.resources)

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
        merged_metrics = {}
        for idx, _ in enumerate(metrics):
            merged_metrics[self._trial_to_original_models[trial_id][idx]] = metrics[idx]
        for model_id in merged_metrics:
            self._original_models[model_id].intermediate_metrics.append(merged_metrics[model_id])
            # model.intermediate_metrics.append(metrics)
            for listener in self._listeners:
                listener.on_intermediate_metric(self._original_models[model_id], merged_metrics[model_id])

    def _final_metric_callback(self, trial_id: int, metrics: MetricData) -> None:
        _logger.error(metrics)

        if isinstance(metrics, float):
            self._listeners[0].on_metric(self._running_models[trial_id], metrics)
        else:
            merged_metrics = {}
            for idx, _ in enumerate(metrics):
                merged_metrics[self._trial_to_original_models[trial_id][idx]] = metrics[idx]
            for model_id in merged_metrics:
                self._original_models[model_id].intermediate_metrics.append(merged_metrics[model_id])
                # model.intermediate_metrics.append(metrics)
                for listener in self._listeners:
                    listener.on_metric(self._original_models[model_id], merged_metrics[model_id])

    def query_available_resource(self) -> List[WorkerInfo]:
        raise NotImplementedError  # move the method from listener to here?

    def budget_exhausted(self) -> bool:
        raise NotImplementedError

    @classmethod
    def trial_execute_graph(cls) -> None:
        """
        Initialize the model, hand it over to trainer.
        """
        graph_data = BaseGraphData.load(receive_trial_parameters())
        _logger.info('CGO_ENGINE trial parameters received')
        random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        file_name = f'_generated_model/{random_str}.py'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as f:
            f.write(graph_data.model_script)
        # with open('_debug_graph_data.json', 'w') as f:
        #     json.dump(graph_data.dump(), f)
        print("graph_data", graph_data)
        trainer_instance = graph_data.evaluator  # utils.import_(graph_data.evaluator)
        model_cls = utils.import_(f'_generated_model.{random_str}._model')
        # trainer_instance.set_model(model_cls())
        # trainer_instance = trainer_cls(model_cls(), graph_data.training_kwargs)
        trainer_instance.fit(model_cls())


class AssemblePolicy:
    @staticmethod
    def _is_related_node(model: Model, node: Node):
        if isinstance(node, AbstractLogicalNode):
            if model in node.related_models:
                return True
        else:
            if model == node.graph.model:
                return True
        return False

    @staticmethod
    def _check_graph_connectivity(model: Model,
                                  group_model: Dict[Model, PhysicalDevice],
                                  logical_plan: LogicalPlan) -> bool:
        for edge in logical_plan.logical_graph.edges:
            if AssemblePolicy._is_related_node(model, edge.head) or \
                    AssemblePolicy._is_related_node(model, edge.tail):
                for grouped_model in group_model:
                    if AssemblePolicy._is_related_node(grouped_model, edge.head) or \
                            AssemblePolicy._is_related_node(grouped_model, edge.tail):
                        return True
        return False

    @staticmethod
    def _check_evaluator(new_model: Model, group_model: Dict[Model, PhysicalDevice]) -> bool:
        if not (isinstance(new_model.evaluator, Lightning)
                and isinstance(new_model.evaluator.module, MultiModelSupervisedLearningModule)):
            return False
        for m in group_model:
            if not m.evaluator == new_model.evaluator:
                return False
        return True

    @staticmethod
    def group(logical_plan, available_devices):
        # TODO: Packing multiple model in one GPU
        # Currently, we only support one model per GPU
        all_grouped_models = []
        group_model = {}
        assert(len(available_devices) > 0)  # There should be at least 1 device, set in CGO_DEVICES
        for idx, m in enumerate(logical_plan.models):
            # models in one group should
            # (1) not use more GPUs than available_devices
            # (2) be connected in the logical plan (independent models should be assembled in multiple groups)
            # (3) use same MultiModelSupervisedLearningModule
            if len(group_model) > 0 and \
                (AssemblePolicy._check_graph_connectivity(m, group_model, logical_plan) == False or
                    AssemblePolicy._check_evaluator(m, group_model) == False):
                all_grouped_models.append(group_model)
                group_model = {}
            group_model[m] = PhysicalDevice('server', available_devices[idx % len(available_devices)])
            if len(group_model) == len(available_devices) or \
                    idx == len(logical_plan.models) - 1:
                all_grouped_models.append(group_model)
                group_model = {}
        return all_grouped_models
