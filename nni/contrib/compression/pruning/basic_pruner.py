# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
import functools
from typing import Callable, Dict, List, overload

import torch

from .tools.common import _MASKS, _METRICS, _TARGET_SPACES
from .tools.calculate_metrics import norm_metrics
from .tools.collect_data import active_sparse_targets_filter
from .tools.sparse_gen import generate_sparsity
from ..base.compressor import Pruner
from ..base.target_space import TargetType, PruningTargetSpace
from ..base.wrapper import ModuleWrapper
from ..utils.evaluator import Evaluator, TensorHook


class _NormPruner(Pruner):
    p: int | str

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict]):
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None,
                 existed_wrapper: Dict[str, ModuleWrapper] | None = None, *args, **kwargs):
        super().__init__(model=model, config_list=config_list, evaluator=evaluator,
                         existed_wrapper=existed_wrapper, *args, **kwargs)

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return active_sparse_targets_filter(self._target_spaces)

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> _METRICS:
        return norm_metrics(p=self.p, data=data, target_spaces=self._target_spaces)

    def _generate_sparsity(self, metrics: _METRICS) -> _MASKS:
        return generate_sparsity(metrics, self._target_spaces)


class LevelPruner(_NormPruner):
    p = 1

    def _set_default_sparse_granularity(self, target_space: PruningTargetSpace):
        return None


class L1NormPruner(_NormPruner):
    p = 1


class L2NormPruner(_NormPruner):
    p = 2


class TaylorFOWeightPruner(Pruner):
    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int):
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int,
                 existed_wrapper: Dict[str, ModuleWrapper] | None = None, *args, **kwargs):
        super().__init__(model=model, config_list=config_list, evaluator=evaluator, existed_wrapper=existed_wrapper, *args, **kwargs)
        self.training_steps = training_steps

        # trigger masks generation when self.current_step == self.training_steps
        self.current_step = 0
        # save all target hooks with format {module_name: {target_name: hook}}
        self.hooks: Dict[str, Dict[str, TensorHook]] = defaultdict(dict)
        # handle the generated masks
        self._masks = None
        self._target_spaces: _TARGET_SPACES

    def register_hooks(self):
        # a factory function, return a tensor hook function for target
        def collector(buffer: List, target: torch.Tensor) -> Callable[[torch.Tensor], None]:
            assert len(buffer) == 0, 'Buffer pass to taylor pruner collector is not empty.'

            def collect_taylor(grad: torch.Tensor):
                if len(buffer) == 0:
                    buffer.append(torch.zeros_like(grad))
                if self.current_step < self.training_steps:
                    buffer[0] += (target.detach() * grad.detach()).pow(2)
            return collect_taylor

        hook_list = []
        for module_name, ts in self._target_spaces.items():
            for target_name, target_space in ts.items():
                if target_space.type is TargetType.PARAMETER:
                    hook = TensorHook(target_space.target, target_name, functools.partial(collector, target=target_space.target))
                    hook_list.append(hook)
                    self.hooks[module_name][target_name] = hook
        self.evaluator.register_hooks(hook_list)

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        data = defaultdict(dict)
        for module_name, hooks in self.hooks.items():
            for target_name, hook in hooks.items():
                if hook.buffer[0] > 0:
                    data[module_name][target_name] = hook.buffer[1] / hook.buffer[0]
        return data

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> _METRICS:
        return norm_metrics(p=1, data=data, target_spaces=self._target_spaces)

    def _generate_sparsity(self, metrics: _METRICS) -> _MASKS:
        return generate_sparsity(metrics, self._target_spaces)

    def register_trigger(self):
        def optimizer_task():
            self.current_step += 1
            if self.current_step == self.training_steps:
                self._masks = self.generate_masks()
                self.update_masks(self._masks)

        self.evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def compress(self):
        self.evaluator.bind_model(self.bound_model, self._get_param_names_map())
        self.register_hooks()
        self.register_trigger()
        self.evaluator.train(self.training_steps)
        self.evaluator.unbind_model()
        return self.bound_model, self._masks
