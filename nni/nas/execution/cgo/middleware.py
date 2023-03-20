# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['CrossGraphOptimization']

import logging
import time
import threading
from collections.abc import Iterable
from typing import List, Dict, Tuple, cast

from nni.common.device import GPUDevice, Device
from nni.experiment.config.training_services import RemoteConfig
from nni.nas.space import GraphModelSpace, Node, ModelStatus
from nni.nas.execution.engine import Middleware, ExecutionEngine
from nni.nas.execution.event import ModelEventType, IntermediateMetricEvent, FinalMetricEvent, TrainingEndEvent
from nni.typehint import TrialMetric

from .logical_optimizer.logical_plan import LogicalPlan, AbstractLogicalNode
from .logical_optimizer.opt_dedup_input import DedupInputOptimizer

_logger = logging.getLogger(__name__)


class CrossGraphOptimization(Middleware):
    """
    The execution engine middleware of Cross-Graph Optimization (CGO).
    It's a technique that merges multiple models into one model for training speedup.
    See `Retiarii paper <https://www.usenix.org/system/files/osdi20-zhang_quanlu.pdf>`__ for details.

    Currently, :class:`CrossGraphOptimization` is only a prototype.
    It's not fully tested, and also, comes with a bunch of constraints on the model space and evaluator:

    - The models must be in the format of :class:`~nni.nas.space.GraphModelSpace`.
    - The evaluator has to be a :class:`~nni.nas.evaluator.pytorch.Lightning` evaluator.
    - The ``lightning_module`` argument of the evaluator must be an instance of
      :class:`~nni.nas.execution.cgo.evaluator.MultiModelSupervisedLearningModule`.
    - The ``trainer`` argument of the evaluator must be an instance of
      :class:`~nni.nas.execution.cgo.evaluator.MultiModelTrainer`.

    There are also a number of limitations:

    - CGO doesn't support stop and resume a checkpoint.
    - Only remote training service is supported.
    - All model history are stored in memory. The experiment might not scale well.

    Parameters
    ----------
    remote_config
        The remote training service config.
    max_concurrency
        The maximum number of trials to run concurrently.
    batch_waiting_time
        Seconds to wait for each batch of trial submission.
        The trials within one batch could apply cross-graph optimization.
    """

    def __init__(self, remote_config: RemoteConfig,
                 max_concurrency: int | None = None,
                 batch_waiting_time: int = 60) -> None:
        super().__init__()

        _logger.warning('Cross graph optimization is an experimental feature. Usages are subject to change.')

        self._history: List[GraphModelSpace] = []
        self._running_models: Dict[int, GraphModelSpace] = {}

        self.logical_plan_counter = 0
        self.available_devices: List[Device] = []
        self.max_concurrency: int | None = max_concurrency

        devices = self._construct_devices(remote_config)
        for device in devices:
            self.available_devices.append(device)
        self.all_devices = self.available_devices.copy()

        self._batch_waiting_time = batch_waiting_time  # seconds to wait for all models in a batch to do cross-graph optimization
        self._optimizers = [DedupInputOptimizer()]
        self._original_models: Dict[int, GraphModelSpace] = {}
        self._original_model_to_multi_model: Dict[int, GraphModelSpace] = {}
        self._trial_to_original_models: Dict[int, List[int]] = {}
        self._trial_used_devices: Dict[int, List[Device]] = {}

        self._queuing_models: List[Tuple[float, GraphModelSpace]] = []
        self._models_to_retry: List[GraphModelSpace] = []
        self._queue_lock = threading.Lock()

        self._stopped = False
        self._consumer_thread = threading.Thread(target=self._consume_models)
        self._consumer_thread.start()

    def _construct_devices(self, training_service):
        devices = []
        if hasattr(training_service, 'machine_list'):
            for machine in cast(RemoteConfig, training_service).machine_list:
                assert machine.gpu_indices is not None, \
                    'gpu_indices must be set in RemoteMachineConfig for CGO execution engine'
                assert isinstance(machine.gpu_indices, list), 'gpu_indices must be a list'
                for gpu_idx in machine.gpu_indices:
                    devices.append(GPUDevice(machine.host, gpu_idx))
        return devices

    def shutdown(self):
        self._stopped = True
        self._consumer_thread.join()

        if self._engine is None:
            _logger.warning('Underlying engine is not set. Skip shutdown.')

        else:
            self.engine.unregister_model_event_callback(ModelEventType.TrainingEnd, self._training_end_callback)
            self.engine.unregister_model_event_callback(ModelEventType.FinalMetric, self._final_metric_callback)
            self.engine.unregister_model_event_callback(ModelEventType.IntermediateMetric, self._intermediate_metric_callback)

            self.engine.shutdown()

    def load_state_dict(self, state_dict: dict) -> None:
        _logger.info('Cross graph optimization does not preserve any states by itself. Loading the state of inner engine: %s', self.engine)
        return self.engine.load_state_dict(state_dict)

    def state_dict(self) -> dict:
        return self.engine.state_dict()

    def set_engine(self, engine: ExecutionEngine) -> None:
        super().set_engine(engine)
        self.engine.register_model_event_callback(ModelEventType.TrainingEnd, self._training_end_callback)
        self.engine.register_model_event_callback(ModelEventType.FinalMetric, self._final_metric_callback)
        self.engine.register_model_event_callback(ModelEventType.IntermediateMetric, self._intermediate_metric_callback)

    def add_optimizer(self, opt):
        self._optimizers.append(opt)

    def submit_models(self, *models: GraphModelSpace) -> None:
        if any(not isinstance(model, GraphModelSpace) for model in models):
            raise TypeError('Cross graph optimization only supports GraphModelSpace.')
        curr_time = time.time()
        _logger.info('%d models are submitted.', len(models))
        with self._queue_lock:
            self._queuing_models.extend([(curr_time, _) for _ in models])
            self._history.extend(models)

    def _submit_retry_models(self, models: List[GraphModelSpace]) -> None:
        _logger.info('%d models are retried.', len(models))
        with self._queue_lock:
            self._models_to_retry.extend(models)

    def _consume_models(self):
        # a thread to monitor self._models_to_retry and self._queuing_models to consume them in batch
        while not self._stopped:
            # retrying jobs should be first scheduled.
            while self._models_to_retry:
                with self._queue_lock:
                    # Get next model and lock the resource.
                    if len(self.available_devices) > 0:
                        m = self._models_to_retry[0]
                        self._models_to_retry = self._models_to_retry[1:]
                        m = self._schedule_models_in_batch(m)
                    else:
                        break

                # submit the single model to avoid cross-graph optimization.
                self.engine.submit_models(*m)

                time.sleep(1)

            # Submit merged models
            merged_models = []

            with self._queue_lock:
                curr_time = time.time()

                num_models_to_submit = len(self.available_devices)
                if self.max_concurrency is not None:
                    num_models_to_submit = min(num_models_to_submit, self.max_concurrency)

                if self._queuing_models and curr_time - self._queuing_models[0][0] >= self._batch_waiting_time:
                    num_models_to_submit = min(num_models_to_submit, len(self._queuing_models))
                    if num_models_to_submit > 0:
                        merged_models = list(self._schedule_models_in_batch(*[_[1] for _ in self._queuing_models[:num_models_to_submit]]))
                        self._queuing_models = self._queuing_models[num_models_to_submit:]
                        _logger.debug('Scheduled %d models in batch.', num_models_to_submit)

            # Outside lock to avoid deadlock.
            if merged_models:
                self.engine.submit_models(*merged_models)

            time.sleep(1)

    def _schedule_models_in_batch(self, *models: GraphModelSpace) -> Iterable[GraphModelSpace]:
        _logger.info('%d models are scheduled in batch.', len(models))
        _logger.debug('Scheduled model ids: %s', [m.model_id for m in models])
        for model in models:
            model.status = ModelStatus.Training
        logical = self._build_logical(list(models))

        for opt in self._optimizers:
            opt.convert(logical)

        for model, grouped_models in self._assemble(logical):
            assert model.placement is not None
            _logger.debug('Created grouped model %d. Original model ids: %s', model.model_id, [m.model_id for m in grouped_models])

            # unique non-cpu devices used by the trial
            self._trial_used_devices[model.model_id] = list(set([_ for _ in model.placement.values() if isinstance(_, GPUDevice)]))
            _logger.debug('Model %d uses devices: %s', model.model_id, self._trial_used_devices[model.model_id])

            # currently, it is impossible for search strategy to submit models more than the number of available devices
            for used_device in self._trial_used_devices[model.model_id]:
                self.available_devices.remove(used_device)  # used_device must be in self.available_devices
            self._running_models[model.model_id] = model

            self._trial_to_original_models[model.model_id] = []
            for m in grouped_models:
                self._original_models[m.model_id] = m
                self._original_model_to_multi_model[m.model_id] = model
                self._trial_to_original_models[model.model_id].append(m.model_id)

            yield model

    def list_models(self) -> Iterable[GraphModelSpace]:
        return self._history

    def idle_worker_available(self) -> bool:
        # the _queuing_models need to use available_devices first
        with self._queue_lock:
            available_for_more_models = len(self.available_devices) - len(self._queuing_models) - len(self._models_to_retry)
        return bool(available_for_more_models)

    def budget_available(self) -> bool:
        return self.engine.budget_available()

    def _assemble(self, logical_plan: LogicalPlan) -> Iterable[Tuple[GraphModelSpace, List[GraphModelSpace]]]:
        """
        Return the assembled models as a list of tuple.
        Each tuple contains the assembled model, the device placement of graph nodes, and the original models.
        """
        grouped_models: List[Dict[GraphModelSpace, Device]] = []

        # try to use the available_devices first so that it can be launched as early as possible
        # if free devices are not enough to assemble all models in one trial, try all devices
        if len(self.available_devices) > 0:
            grouped_models = AssemblePolicy().group(logical_plan, self.available_devices)

        if len(self.available_devices) == 0 or len(grouped_models) > 1:
            grouped_models: List[Dict[GraphModelSpace, Device]] = AssemblePolicy().group(logical_plan, self.all_devices)

        for multi_model in grouped_models:
            model, model_placement = logical_plan.assemble(multi_model)
            assert isinstance(model, GraphModelSpace), 'Assembled model must be a GraphModelSpace.'

            from nni.nas.evaluator.pytorch import Lightning
            from .evaluator import MultiModelLightningModule, MultiModelTrainer

            if not isinstance(model.evaluator, Lightning):
                raise TypeError('Cross-graph optimization only supports pytorch lighting as evaluator.')
            if not isinstance(model.evaluator.module, MultiModelLightningModule):
                raise TypeError('Cross-graph optimization only support MultiModelLightningModule')
            if not isinstance(model.evaluator.trainer, MultiModelTrainer):
                raise TypeError('Cross-graph optimization only support MultiModelTrainer')

            # Set n_models of the lightning module.
            model.evaluator.module.n_models = len(multi_model)
            model.status = ModelStatus.Frozen
            model.placement = model_placement
            model.metrics.strict = False

            yield model, list(multi_model.keys())

    def _build_logical(self, models: List[GraphModelSpace]) -> LogicalPlan:
        assert len(models) > 0
        logical_plan = LogicalPlan(model_cls=models[0].__class__, plan_id=self.logical_plan_counter)
        for model in models:
            logical_plan.add_model(model)
        self.logical_plan_counter += 1
        return logical_plan

    def _training_end_callback(self, event: TrainingEndEvent) -> None:
        model = cast(GraphModelSpace, event.model)
        _logger.debug(f'Training end for merged model {model.model_id}.')
        model = self._running_models[model.model_id]
        models_to_retry = []
        for model_id in self._original_model_to_multi_model:
            if self._original_model_to_multi_model[model_id] == model:
                original_model = self._original_models[model_id]
                if model.status == ModelStatus.Trained:
                    self.dispatch_model_event(TrainingEndEvent(original_model, ModelStatus.Trained))
                else:
                    # the failed models in a multi-model will be retried one by one w/o CGO
                    if len(self._trial_to_original_models[model.model_id]) > 1:
                        # TODO: should the listeners be notified?
                        original_model.status = ModelStatus.Frozen
                        original_model.metrics.clear()
                        models_to_retry.append(original_model)
                    else:
                        self.dispatch_model_event(TrainingEndEvent(original_model, ModelStatus.Failed))

        if len(models_to_retry) > 0:
            self._submit_retry_models(models_to_retry)

        self.available_devices.extend(self._trial_used_devices[model.model_id])
        self.available_devices = sorted(list(set(self.available_devices)))
        del self._running_models[model.model_id]

    def _intermediate_metric_callback(self, event: IntermediateMetricEvent) -> None:
        model = cast(GraphModelSpace, event.model)
        metrics = cast(List[TrialMetric], event.metric)
        _logger.debug(f'Received intermediate metrics for merged model {model.model_id}: {metrics}')
        if not isinstance(metrics, Iterable):
            raise TypeError('Intermediate metrics must be a list of TrialMetric.')
        if len(metrics) != len(self._trial_to_original_models[model.model_id]):
            raise ValueError('Number of intermediate metrics must be equal to number of original models.')

        merged_metrics: Dict[int, TrialMetric] = {}
        for idx, _ in enumerate(metrics):
            merged_metrics[self._trial_to_original_models[model.model_id][idx]] = metrics[idx]
        for model_id in merged_metrics:
            self.dispatch_model_event(IntermediateMetricEvent(self._original_models[model_id], merged_metrics[model_id]))

    def _final_metric_callback(self, event: FinalMetricEvent) -> None:
        model = cast(GraphModelSpace, event.model)
        metrics = cast(List[TrialMetric], event.metric)
        _logger.debug(f'Received final metrics for merged model {model.model_id}: {metrics}')
        if not isinstance(metrics, Iterable):
            raise TypeError('Final metrics must be a list of TrialMetric.')
        if len(metrics) != len(self._trial_to_original_models[model.model_id]):
            raise ValueError('Number of final metrics must be equal to number of original models.')

        merged_metrics: Dict[int, TrialMetric] = {}
        for idx, _ in enumerate(metrics):
            merged_metrics[self._trial_to_original_models[model.model_id][idx]] = metrics[idx]

        _logger.debug(f'Mapped to metrics of original models: {merged_metrics}')

        for model_id in merged_metrics:
            self.dispatch_model_event(FinalMetricEvent(self._original_models[model_id], merged_metrics[model_id]))


class AssemblePolicy:
    @staticmethod
    def _is_related_node(model: GraphModelSpace, node: Node):
        if isinstance(node, AbstractLogicalNode):
            if model in node.related_models:
                return True
        else:
            if model == node.graph.model:
                return True
        return False

    @staticmethod
    def _check_graph_connectivity(model: GraphModelSpace,
                                  group_model: Dict[GraphModelSpace, Device],
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
    def _check_evaluator(new_model: GraphModelSpace, group_model: Dict[GraphModelSpace, Device]) -> bool:
        from nni.nas.evaluator.pytorch import Lightning
        from .evaluator import MultiModelLightningModule, MultiModelTrainer

        if not (isinstance(new_model.evaluator, Lightning)
                and isinstance(new_model.evaluator.module, MultiModelLightningModule)
                and isinstance(new_model.evaluator.trainer, MultiModelTrainer)):
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
            group_model[m] = available_devices[idx % len(available_devices)]
            if len(group_model) == len(available_devices) or \
                    idx == len(logical_plan.models) - 1:
                all_grouped_models.append(group_model)
                group_model = {}
        return all_grouped_models
