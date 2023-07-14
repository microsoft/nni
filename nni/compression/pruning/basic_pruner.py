# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Tuple, overload

import torch

from .tools import _DATA, _METRICS, _MASKS, active_sparse_targets_filter, norm_metrics, fpgm_metrics, generate_sparsity
from ..base.compressor import Compressor, Pruner
from ..base.target_space import PruningTargetSpace
from ..base.wrapper import ModuleWrapper
from ..utils.evaluator import Evaluator

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
        self.interval_steps = -1
        self.total_times: int | Literal['unlimited'] = 1
        # here is a reserved interface for potential iterative pruning needs,
        # first_step_gen controls if the masks generated on the first step.
        self.first_step_gen = False

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
            if self._current_step == self.interval_steps and \
                (self._remaining_times == 'unlimited' or self._remaining_times > 0):  # type: ignore
                self._current_step = 0
            if self._is_first_step:
                self._is_first_step = False

        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def _single_compress(self, max_steps: None, max_epochs: None):
        assert max_steps is None and max_epochs is None, \
            f'{self.__class__.__name__} do not support training aware pruning under single compress mode.'
        masks = self.generate_masks()
        self.update_masks(masks)

    def _fuse_preprocess(self, evaluator: Evaluator):
        self._register_trigger(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator):
        pass

    @overload
    def compress(self) -> Tuple[torch.nn.Module, _MASKS]:
        ...

    @overload
    def compress(self, max_steps: int | None, max_epochs: int | None) -> Tuple[torch.nn.Module, _MASKS]:
        ...

    def compress(self, max_steps: int | None = None, max_epochs: int | None = None):
        return super().compress(max_steps, max_epochs)


class LevelPruner(_NormPruner):
    """
    This is a basic pruner, and in some papers it is called magnitude pruning or fine-grained pruning.
    It will mask the smallest magnitude weights in each specified layer by a saprsity ratio configured in the config list.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        A list of dict, each dict configure which module need to be pruned, and how to prune.
        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.
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
        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.

    Examples
    --------
        Please refer to
        :githublink:`examples/compression/pruning/norm_pruning.py <examples/compression/pruning/norm_pruning.py>`.
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
        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.

    Examples
    --------
        Please refer to
        :githublink:`examples/compression/pruning/norm_pruning.py <examples/compression/pruning/norm_pruning.py>`.
    """
    p = 2


class FPGMPruner(_NormPruner):
    r"""
    FPGM pruner prunes the blocks of the weight with the smallest geometric median.
    FPGM chooses the weight blocks with the most replaceable contribution.

    For more details, please refer to
    `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration <https://arxiv.org/abs/1811.00250>`__.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        A list of dict, each dict configure which module need to be pruned, and how to prune.
        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.

    Examples
    --------
        Please refer to
        :githublink:`examples/compression/pruning/norm_pruning.py <examples/compression/pruning/norm_pruning.py>`.
    """
    p = 2

    def _calculate_metrics(self, data: _DATA) -> _METRICS:
        return fpgm_metrics(p=self.p, data=data, target_spaces=self._target_spaces)
