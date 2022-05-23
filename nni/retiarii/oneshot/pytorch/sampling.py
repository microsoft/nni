# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Experimental version of sampling-based one-shot implementation."""

from __future__ import annotations
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from .base_lightning import BaseOneShotLightningModule, MutationHook, no_default_hook
from .supermodule.sampling import (
    PathSamplingInput, PathSamplingLayer, MixedOpPathSamplingPolicy,
    PathSamplingCell, PathSamplingRepeat
)
from .supermodule.operation import NATIVE_MIXED_OPERATIONS
from .enas import ReinforceController, ReinforceField


class RandomSamplingLightningModule(BaseOneShotLightningModule):
    _random_note = """
    Random Sampling NAS Algorithm.
    In each epoch, model parameters are trained after a uniformly random sampling of each choice.
    Notably, the exporting result is **also a random sample** of the search space.

    Parameters
    ----------
    {{module_params}}
    {base_params}
    """.format(base_params=BaseOneShotLightningModule._mutation_hooks_note)

    __doc__ = _random_note.format(
        module_params=BaseOneShotLightningModule._inner_module_note,
    )

    # turn on automatic optimization because nothing interesting is going on here.
    @property
    def automatic_optimization(self) -> bool:
        return True

    def default_mutation_hooks(self) -> list[MutationHook]:
        """Replace modules with differentiable versions"""
        hooks = [
            PathSamplingLayer.mutate,
            PathSamplingInput.mutate,
            PathSamplingRepeat.mutate,
            PathSamplingCell.mutate,
        ]
        hooks += [operation.mutate for operation in NATIVE_MIXED_OPERATIONS]
        hooks.append(no_default_hook)
        return hooks

    def mutate_kwargs(self):
        """Use path sampling strategy for mixed-operations."""
        return {
            'mixed_op_sampling': MixedOpPathSamplingPolicy
        }

    def training_step(self, batch, batch_idx):
        self.resample()
        return self.model.training_step(batch, batch_idx)


class EnasLightningModule(RandomSamplingLightningModule):
    _enas_note = """
    The implementation of ENAS :cite:p:`pham2018efficient`. There are 2 steps in an epoch.
    Firstly, training model parameters.
    Secondly, training ENAS RL agent. The agent will produce a sample of model architecture to get the best reward.

    {{module_notes}}

    Parameters
    ----------
    {{module_params}}
    {base_params}
    ctrl_kwargs : dict
        Optional kwargs that will be passed to :class:`ReinforceController`.
    entropy_weight : float
        Weight of sample entropy loss.
    skip_weight : float
        Weight of skip penalty loss.
    baseline_decay : float
        Decay factor of baseline. New baseline will be equal to ``baseline_decay * baseline_old + reward * (1 - baseline_decay)``.
    ctrl_steps_aggregate : int
        Number of steps that will be aggregated into one mini-batch for RL controller.
    ctrl_grad_clip : float
        Gradient clipping value of controller.
    reward_metric_name : str or None
        The name of the metric which is treated as reward.
        This will be not effective when there's only one metric returned from evaluator.
        If there are multiple, it will find the metric with key name ``reward_metric_name``,
        which is "default" by default.
        Otherwise it raises an exception indicating multiple metrics are found.
    """.format(base_params=BaseOneShotLightningModule._mutation_hooks_note)

    __doc__ = _enas_note.format(
        module_notes='``ENASModule`` should be trained with :class:`nni.retiarii.oneshot.utils.ConcatenateTrainValDataloader`.',
        module_params=BaseOneShotLightningModule._inner_module_note,
    )

    @property
    def automatic_optimization(self) -> bool:
        return False

    def __init__(self,
                 inner_module: pl.LightningModule,
                 *,
                 ctrl_kwargs: dict[str, Any] | None = None,
                 entropy_weight: float = 1e-4,
                 skip_weight: float = .8,
                 baseline_decay: float = .999,
                 ctrl_steps_aggregate: float = 20,
                 ctrl_grad_clip: float = 0,
                 reward_metric_name: str | None = None,
                 mutation_hooks: list[MutationHook] | None = None):
        super().__init__(inner_module, mutation_hooks)

        # convert parameter spec to legacy ReinforceField
        # this part will be refactored
        self.nas_fields: list[ReinforceField] = []
        for name, param_spec in self.search_space_spec().items():
            if param_spec.chosen_size not in (1, None):
                raise ValueError('ENAS does not support n_chosen to be values other than 1 or None.')
            self.nas_fields.append(ReinforceField(name, param_spec.size, param_spec.chosen_size == 1))
        self.controller = ReinforceController(self.nas_fields, **(ctrl_kwargs or {}))

        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.baseline = 0.
        self.ctrl_steps_aggregate = ctrl_steps_aggregate
        self.ctrl_grad_clip = ctrl_grad_clip
        self.reward_metric_name = reward_metric_name

    def configure_architecture_optimizers(self):
        return optim.Adam(self.controller.parameters(), lr=3.5e-4)

    def training_step(self, batch_packed, batch_idx):
        batch, mode = batch_packed

        if mode == 'train':
            # train model params
            with torch.no_grad():
                self.resample()
            self.call_weight_optimizers('zero_grad')
            step_output = self.model.training_step(batch, batch_idx)
            w_step_loss = step_output['loss'] \
                if isinstance(step_output, dict) else step_output
            self.manual_backward(w_step_loss)
            self.call_weight_optimizers('step')

        else:
            # train ENAS agent
            arc_opt = self.architecture_optimizers()
            if not isinstance(arc_opt, optim.Optimizer):
                raise TypeError(f'Expect arc_opt to be a single Optimizer, but found: {arc_opt}')
            arc_opt.zero_grad()
            self.resample()

            step_output = self.model.validation_step(batch, batch_idx)

            # use the default metric of self.model as reward function
            if len(self.trainer.callback_metrics) == 1:
                _, metric = next(iter(self.trainer.callback_metrics.items()))
            else:
                metric_name = self.reward_metric_name or 'default'
                if metric_name not in self.trainer.callback_metrics:
                    raise KeyError(f'Model reported metrics should contain a ``{metric_name}`` key but '
                                   f'found multiple (or zero) metrics without default: {list(self.trainer.callback_metrics.keys())}. '
                                   f'Try to use self.log to report metrics with the specified key ``{metric_name}`` in validation_step, '
                                   'and remember to set on_step=True.')
                metric = self.trainer.callback_metrics[metric_name]
            reward: float = metric.item()

            if self.entropy_weight:
                reward = reward + self.entropy_weight * self.controller.sample_entropy.item()  # type: ignore
            self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
            rnn_step_loss = self.controller.sample_log_prob * (reward - self.baseline)
            if self.skip_weight:
                rnn_step_loss = rnn_step_loss + self.skip_weight * self.controller.sample_skip_penalty

            rnn_step_loss = rnn_step_loss / self.ctrl_steps_aggregate
            self.manual_backward(rnn_step_loss)

            if (batch_idx + 1) % self.ctrl_steps_aggregate == 0:
                if self.ctrl_grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.controller.parameters(), self.ctrl_grad_clip)
                arc_opt.step()
                arc_opt.zero_grad()

        return step_output

    def resample(self):
        """Resample the architecture with ENAS controller."""
        sample = self.controller.resample()
        result = self._interpret_controller_sampling_result(sample)
        for module in self.nas_modules:
            module.resample(memo=result)
        return result

    def export(self):
        """Run one more inference of ENAS controller."""
        self.controller.eval()
        with torch.no_grad():
            return self._interpret_controller_sampling_result(self.controller.resample())

    def _interpret_controller_sampling_result(self, sample: dict[str, int]) -> dict[str, Any]:
        """Convert ``{label: index}`` to ``{label: name}``"""
        space_spec = self.search_space_spec()
        for key in list(sample.keys()):
            sample[key] = space_spec[key].values[sample[key]]
        return sample
