# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Dict, List, overload

import torch
from torch.optim import Adam

from .scheduled_pruner import ScheduledPruner
from .tools import is_active_target, generate_sparsity
from ..base.compressor import Compressor
from ..base.target_space import TargetType
from ..base.wrapper import ModuleWrapper
from ..utils import Evaluator, _EVALUATOR_DOCSTRING

MOVEMENT_SCORE_PNAME = '{}_mvp_score'
_logger = logging.getLogger(__name__)


class MovementPruner(ScheduledPruner):
    __doc__ = r"""
    Movement pruner is an implementation of movement pruning.
    This is a "fine-pruning" algorithm, which means the masks may change during each fine-tuning step.
    Each weight element will be scored by the opposite of the sum of the product of weight and its gradient during each step.
    This means the weight elements moving towards zero will accumulate negative scores,
    the weight elements moving away from zero will accumulate positive scores.
    The weight elements with low scores will be masked during inference.

    The following figure from the paper shows the weight pruning by movement pruning.

    .. image:: ../../../img/movement_pruning.png
        :target: ../../../img/movement_pruning.png
        :alt:

    For more details, please refer to `Movement Pruning: Adaptive Sparsity by Fine-Tuning <https://arxiv.org/abs/2005.07683>`__.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        A list of dict, each dict configure which module need to be pruned, and how to prune.
        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.
    evaluator
        {evaluator_docstring}
    warmup_step
        The total `optimizer.step()` number before start pruning for warm up.
        Make sure ``warmup_step`` is smaller than ``cooldown_begin_step``.
    cooldown_begin_step
        The number of steps at which sparsity stops growing, note that the sparsity stop growing doesn't mean masks not changed.
        The sparse ratio or sparse threshold after each `optimizer.step()` is::

            final_sparse * (1 - (1 - (current_step - warm_up_step) / (cool_down_beginning_step - warm_up_step)) ** 3)
    regular_scale
        A scale factor used to control the movement score regular loss.
        This factor only works on pruning target controlled by ``sparse_threshold``,
        the pruning target controlled by ``sparse_ratio`` will not be regularized.

    Examples
    --------
        Please refer to
        :githublink:`examples/tutorials/new_pruning_bert_glue.py <examples/tutorials/new_pruning_bert_glue.py>`.
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

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
        self.evaluator: Evaluator
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

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], warmup_step: int,
                        cooldown_begin_step: int, regular_scale: float = 1., evaluator: Evaluator | None = None):
        return super().from_compressor(compressor, new_config_list, warmup_step=warmup_step, cooldown_begin_step=cooldown_begin_step,
                                       regular_scale=regular_scale, evaluator=evaluator)

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
                        # TODO: here using a shrinked score to save memory, but need to test the speed.
                        score_val = torch.zeros_like(target_space.target)  # type: ignore
                        if target_space._scaler is not None:
                            score_val = target_space._scaler.shrink(score_val, keepdim=True)
                        target_space._wrapper.register_parameter(MOVEMENT_SCORE_PNAME.format(target_name),
                                                                 torch.nn.Parameter(score_val))
                        score = target_space._get_wrapper_attr(MOVEMENT_SCORE_PNAME.format(target_name))
                        self.scores[module_name][target_name] = score
                    else:
                        raise NotImplementedError()

    def _register_scores_optimization(self, evaluator: Evaluator):
        scores = []
        for _, target_scores in self.scores.items():
            for _, score in target_scores.items():
                scores.append(score)

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
            for module_name, target_scores in self.scores.items():
                for target_name, score in target_scores.items():
                    target_space = self._target_spaces[module_name][target_name]
                    if target_space.sparse_threshold is not None:
                        reg_loss += torch.norm(score.sigmoid(), p=1) / score.numel()  # type: ignore
                        count += 1
            ratio = max(0., min(1., 1 - (self._remaining_times / self.total_times) ** 3))
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
                score: torch.Tensor = getattr(target_space._wrapper, MOVEMENT_SCORE_PNAME.format(target_name), None)  # type: ignore
                if score is not None:
                    data[module_name][target_name] = score.clone().detach()
        return data

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        metrics = defaultdict(dict)
        for module_name, td in data.items():
            for target_name, target_data in td.items():
                if self._target_spaces[module_name][target_name].sparse_threshold is not None:
                    metrics[module_name][target_name] = target_data.sigmoid()
                else:
                    metrics[module_name][target_name] = target_data
        return metrics

    def _generate_sparsity(self, metrics: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        return generate_sparsity(metrics=metrics, target_spaces=self._target_spaces)

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        self._fusion_compress(max_steps, max_epochs)

    def _fuse_preprocess(self, evaluator: Evaluator):
        self._update_sparse_goals_by_ratio(0.)
        self._register_movement_scores()
        self._patch_loss(evaluator)
        self._register_scores_optimization(evaluator)
        self._register_trigger(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator):
        pass

    def compress(self, max_steps: int | None, max_epochs: int | None):
        if max_steps is not None:
            assert max_steps >= self.cooldown_begin_step
        else:
            warn_msg = \
                f'Using epochs number as training duration, please make sure the total training steps larger than `cooldown_begin_step`.'
            _logger.warning(warn_msg)
        return super().compress(max_steps, max_epochs)
