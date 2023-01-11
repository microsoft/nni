# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Dict, List, overload

import torch
from torch.optim import Adam

from .scheduled_pruner import ScheduledPruner
from .tools import is_active_target, generate_sparsity, sum_metric
from ..base.target_space import TargetType
from ..base.wrapper import ModuleWrapper
from ..utils import Evaluator

MOVEMENT_SCORE_PNAME = '{}_mvp_score'
_logger = logging.getLogger(__name__)


class MovementPruner(ScheduledPruner):
    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, warmup_step: int,
                 cooldown_begin_step: int, regular_scale: float = 1.):
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, warmup_step: int,
                 cooldown_begin_step: int, regular_scale: float = 1., existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, warmup_step: int,
                 cooldown_begin_step: int, regular_scale: float = 1., existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model, config_list, evaluator, existed_wrappers)
        assert 0 <= warmup_step < cooldown_begin_step
        self.warmup_step = warmup_step
        self.cooldown_begin_step = cooldown_begin_step
        self.regular_scale = regular_scale
        self._init_sparse_goals()
        self._set_apply_method()

        self.interval_steps = 1
        self.total_times = (self.cooldown_begin_step - self.warmup_step) // self.interval_steps
        self._remaining_times: int
        self.scores: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

    def _set_apply_method(self):
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                if target_space.apply_method == 'mul':
                    target_space.apply_method = 'movement_mul'
                if target_space.apply_method == 'add':
                    target_space.apply_method = 'movement_add'

    def _register_movement_scores(self):
        for module_name, ts in self._target_spaces.items():
            for target_name, target_space in ts.items():
                if is_active_target(target_space):
                    # TODO: add input / output
                    if target_space.type is TargetType.PARAMETER:
                        target_space._wrapper.register_parameter(MOVEMENT_SCORE_PNAME.format(target_name),
                                                                 torch.nn.Parameter(torch.zeros_like(target_space.target)))
                        score = target_space._get_wrapper_attr(MOVEMENT_SCORE_PNAME.format(target_name))
                        self.scores[module_name][target_name] = score
                    else:
                        raise NotImplementedError()

    def _register_scores_optimization(self, evaluator: Evaluator):
        scores = []
        for _, v in self.scores.items():
            for _, s in v.items():
                scores.append(s)

        if not scores:
            return

        params = [{"params": scores}]
        optimizer = Adam(params, 1e-2)

        def optimizer_task():
            optimizer.step()
            optimizer.zero_grad()

        evaluator.patch_optimizer_step(before_step_tasks=[optimizer_task], after_step_tasks=[])

    def _patch_loss(self, evaluator: Evaluator):

        def loss_patch(original_loss, batch):
            reg_loss = 0.
            count = 0
            for _, ts in self._target_spaces.items():
                for target_name, target_space in ts.items():
                    score: torch.Tensor = getattr(target_space._wrapper, MOVEMENT_SCORE_PNAME.format(target_name), None)
                    if target_space.sparse_threshold is not None and score is not None:
                        reg_loss += torch.norm(torch.sigmoid(score), p=1) / score.numel()
                        count += 1
            ratio = max(0., min(1., 1 - self._remaining_times / self.total_times ** 3))
            if count > 0:
                reg_loss = self.regular_scale * ratio * reg_loss / count
            return original_loss + reg_loss

        evaluator.patch_loss(loss_patch)

    def _register_trigger(self, evaluator: Evaluator):
        self._current_step = 0
        self._iterial_step = 0
        self._remaining_times = self.total_times

        def optimizer_task():
            self._current_step += 1
            if self.warmup_step < self._current_step <= self.cooldown_begin_step:
                self._iterial_step += 1
                if self._iterial_step == self.interval_steps:
                    self._remaining_times -= 1
                    self.update_sparse_goals(self.total_times - self._remaining_times)
                    debug_msg = f'{self.__class__.__name__} generate masks, remaining times {self._remaining_times}'
                    _logger.debug(debug_msg)
                    if self._remaining_times > 0:
                        self._iterial_step = 0
            if self.warmup_step < self._current_step:
                self.update_masks(self.generate_masks())

        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def update_sparse_goals(self, current_times: int):
        ratio = max(0., min(1., 1 - (1 - current_times / self.total_times) ** 3))
        self._update_sparse_goals_by_ratio(ratio)

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        data = defaultdict(dict)
        for module_name, ts in self._target_spaces.items():
            for target_name, target_space in ts.items():
                score: torch.Tensor = getattr(target_space._wrapper, MOVEMENT_SCORE_PNAME.format(target_name), None)
                if score is not None:
                    data[module_name][target_name] = score.clone().detach()
        return data

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        return sum_metric(data=data, target_spaces=self._target_spaces)

    def _generate_sparsity(self, metrics: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        return generate_sparsity(metrics=metrics, target_spaces=self._target_spaces)

    def compress(self, max_steps: int):
        assert max_steps >= self.cooldown_begin_step
        self.evaluator.bind_model(self.bound_model, self._get_param_names_map())
        self.compress_fuse(self.evaluator)
        self.evaluator.train(max_steps)
        self.evaluator.unbind_model()
        return self.bound_model, self.get_masks()

    def compress_fuse(self, evaluator: Evaluator):
        self._update_sparse_goals_by_ratio(0.)
        self._register_movement_scores()
        self._patch_loss(evaluator)
        self._register_scores_optimization(evaluator)
        self._register_trigger(evaluator)
