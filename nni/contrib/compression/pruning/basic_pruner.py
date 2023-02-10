# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
import functools
import logging
from typing import Callable, Dict, List, Literal, overload

import torch

from .tools import _DATA, _METRICS, _MASKS, active_sparse_targets_filter, norm_metrics, generate_sparsity, is_active_target
from ..base.compressor import Compressor, Pruner
from ..base.target_space import TargetType, PruningTargetSpace
from ..base.wrapper import ModuleWrapper
from ..utils.evaluator import Evaluator, TensorHook

_logger = logging.getLogger(__name__)


class _NormPruner(Pruner):
    p: int | str

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict]):
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None,
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None,
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model, config_list, evaluator, existed_wrappers)

        # `skip_first_step`, `interval_steps` and `total_times` are used by `register_trigger`.
        # `skip_first_step` controls if generating masks at the first step.
        # `interval_steps` is the optimize step interval for generating masks.
        # `total_times` is the total generation times of masks.
        self.first_step_gen = False
        self.interval_steps = -1
        self.total_times: int | Literal['unlimited'] = 1

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], evaluator: Evaluator | None = None):
        return super().from_compressor(compressor, new_config_list, evaluator=evaluator)

    def _collect_data(self) -> _DATA:
        return active_sparse_targets_filter(self._target_spaces)

    def _calculate_metrics(self, data: _DATA) -> _METRICS:
        return norm_metrics(p=self.p, data=data, target_spaces=self._target_spaces)

    def _generate_sparsity(self, metrics: _METRICS) -> _MASKS:
        return generate_sparsity(metrics, self._target_spaces)

    def _register_trigger(self, evaluator: Evaluator):
        self._current_step = 0
        self._is_first_step = True
        self._remaining_times = self.total_times

        def optimizer_task():
            self._current_step += 1
            if (self._is_first_step and self.first_step_gen) or self._current_step == self.interval_steps:
                masks = self.generate_masks()
                self.update_masks(masks)
                if isinstance(self._remaining_times, int):
                    self._remaining_times -= 1
                debug_msg = f'{self.__class__.__name__} generate masks, remaining times {self._remaining_times}'
                _logger.debug(debug_msg)
            if self._current_step == self.interval_steps and (self._remaining_times == 'unlimited' or self._remaining_times > 0):  # type: ignore
                self._current_step = 0
            if self._is_first_step:
                self._is_first_step = False

        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def compress_fuse(self, evaluator: Evaluator):
        self._register_trigger(evaluator)


class LevelPruner(_NormPruner):
    """
    This is a basic pruner, and in some papers called it magnitude pruning or fine-grained pruning.
    It will mask the smallest magnitude weights in each specified layer by a saprsity ratio configured in the config list.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        A list of dict, each dict configure which module need to be pruned, and how to prune.

    Examples
    --------
        TODO
    """
    p = 1

    def _set_default_sparse_granularity(self, target_space: PruningTargetSpace):
        return None


class L1NormPruner(_NormPruner):
    """
    L1 norm pruner computes the l1 norm of the layer weight on the first dimension,
    then prune the weight blocks on this dimension with smaller l1 norm values.
    i.e., compute the l1 norm of the filters in convolution layer as metric values,
    compute the l1 norm of the weight by rows in linear layer as metric values.

    For more details, please refer to `PRUNING FILTERS FOR EFFICIENT CONVNETS <https://arxiv.org/abs/1608.08710>`__.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        A list of dict, each dict configure which module need to be pruned, and how to prune.

    Examples
    --------
        TODO
    """
    p = 1


class L2NormPruner(_NormPruner):
    """
    L2 norm pruner is a variant of L1 norm pruner.
    The only different between L2 norm pruner and L1 norm pruner is
    L2 norm pruner prunes the weight with the smallest L2 norm of the weights.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        A list of dict, each dict configure which module need to be pruned, and how to prune.

    Examples
    --------
        TODO
    """
    p = 2


class TaylorFOWeightPruner(Pruner):
    """
    Taylor FO weight pruner is a pruner which prunes on the first weight dimension by default,
    based on estimated importance calculated from the first order taylor expansion on weights to achieve a preset level of network sparsity.
    The estimated importance is defined as the paper
    `Importance Estimation for Neural Network Pruning <http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf>`__.

    :math:`\widehat{\mathcal{I}}_{\mathcal{S}}^{(1)}(\mathbf{W}) \triangleq \sum_{s \in \mathcal{S}} \mathcal{I}_{s}^{(1)}(\mathbf{W})=\sum_{s \in \mathcal{S}}\left(g_{s} w_{s}\right)^{2}`

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        A list of dict, each dict configure which module need to be pruned, and how to prune.
    evaluator
        TODO: {evaluator_docstring}
    training_steps
        The step number used to collect gradients, the masks will be generated after training_steps training.

    Examples
    --------
        TODO
    """

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int):
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int,
                 existed_wrappers: Dict[str, ModuleWrapper]):
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, training_steps: int,
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model=model, config_list=config_list, evaluator=evaluator,
                         existed_wrappers=existed_wrappers)
        self.evaluator: Evaluator
        self.training_steps = training_steps

        # trigger masks generation when self._current_step == self.training_steps
        self._current_step = 0
        # save all target hooks with format {module_name: {target_name: hook}}
        self.hooks: Dict[str, Dict[str, TensorHook]] = defaultdict(dict)

        # `interval_steps` and `total_times` are used by `register_trigger`.
        # `interval_steps` is the optimize step interval for generating masks.
        # `total_times` is the total generation times of masks.
        self.interval_steps = training_steps
        self.total_times: int | Literal['unlimited'] = 1

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], training_steps: int, evaluator: Evaluator | None = None):
        return super().from_compressor(compressor, new_config_list, training_steps=training_steps, evaluator=evaluator)

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        data = defaultdict(dict)
        for module_name, hooks in self.hooks.items():
            for target_name, hook in hooks.items():
                if len(hook.buffer) > 0:
                    data[module_name][target_name] = hook.buffer[0] / self.training_steps
        return data

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> _METRICS:
        return norm_metrics(p=1, data=data, target_spaces=self._target_spaces)

    def _generate_sparsity(self, metrics: _METRICS) -> _MASKS:
        return generate_sparsity(metrics, self._target_spaces)

    def _register_hooks(self, evaluator: Evaluator):
        def collector(buffer: List, target: torch.Tensor) -> Callable[[torch.Tensor], None]:
            # a factory function, return a tensor hook function for target
            assert len(buffer) == 0, 'Buffer pass to taylor pruner collector is not empty.'

            def collect_taylor(grad: torch.Tensor):
                if len(buffer) == 0:
                    buffer.append(torch.zeros_like(grad))
                if self._current_step < self.training_steps:
                    buffer[0] += (target.detach() * grad.detach()).pow(2)
            return collect_taylor

        hook_list = []
        for module_name, ts in self._target_spaces.items():
            for target_name, target_space in ts.items():
                if is_active_target(target_space):
                    # TODO: add input/output
                    if target_space.type is TargetType.PARAMETER:
                        hook = TensorHook(target_space.target, target_name, functools.partial(collector, target=target_space.target))  # type: ignore
                        hook_list.append(hook)
                        self.hooks[module_name][target_name] = hook
                    else:
                        raise NotImplementedError()
        evaluator.register_hooks(hook_list)

    def _register_trigger(self, evaluator: Evaluator):
        assert self.interval_steps >= self.training_steps or self.interval_steps < 0
        self._remaining_times = self.total_times

        def optimizer_task():
            self._current_step += 1
            if self._current_step == self.training_steps:
                masks = self.generate_masks()
                self.update_masks(masks)
                if isinstance(self._remaining_times, int):
                    self._remaining_times -= 1
                debug_msg = f'{self.__class__.__name__} generate masks, remaining times {self._remaining_times}'
                _logger.debug(debug_msg)
            if self._current_step == self.interval_steps and (self._remaining_times == 'unlimited' or self._remaining_times > 0):  # type: ignore
                self._current_step = 0
                for _, hooks in self.hooks.items():
                    for _, hook in hooks.items():
                        hook.buffer.clear()

        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def compress(self):
        self.evaluator.bind_model(self.bound_model, self._get_param_names_map())
        self.compress_fuse(self.evaluator)
        self.evaluator.train(self.training_steps)
        self.evaluator.unbind_model()
        return self.bound_model, self.get_masks()

    def compress_fuse(self, evaluator: Evaluator):
        self._register_hooks(evaluator)
        self._register_trigger(evaluator)
