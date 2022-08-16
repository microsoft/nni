# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Experimental version of sampling-based one-shot implementation."""

from __future__ import annotations
import warnings
from typing import Any, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from .base_lightning import MANUAL_OPTIMIZATION_NOTE, BaseOneShotLightningModule, MutationHook, no_default_hook
from .supermodule.operation import NATIVE_MIXED_OPERATIONS, NATIVE_SUPPORTED_OP_NAMES
from .supermodule.sampling import (
    PathSamplingInput, PathSamplingLayer, MixedOpPathSamplingPolicy,
    PathSamplingCell, PathSamplingRepeat
)
from .enas import ReinforceController, ReinforceField


class RandomSamplingLightningModule(BaseOneShotLightningModule):
    _random_note = """
    Train a super-net with uniform path sampling. See `reference <https://arxiv.org/abs/1904.00420>`__.

    In each epoch, model parameters are trained after a uniformly random sampling of each choice.
    Notably, the exporting result is **also a random sample** of the search space.

    The supported mutation primitives of RandomOneShot are:

    * :class:`nni.retiarii.nn.pytorch.LayerChoice`.
    * :class:`nni.retiarii.nn.pytorch.InputChoice`.
    * :class:`nni.retiarii.nn.pytorch.ValueChoice` (only when used in {supported_ops}).
    * :class:`nni.retiarii.nn.pytorch.Repeat`.
    * :class:`nni.retiarii.nn.pytorch.Cell`.
    * :class:`nni.retiarii.nn.pytorch.NasBench201Cell`.

    This strategy assumes inner evaluator has set
    `automatic optimization <https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html>`__ to true.

    Parameters
    ----------
    {{module_params}}
    {base_params}
    """.format(
        base_params=BaseOneShotLightningModule._mutation_hooks_note,
        supported_ops=', '.join(NATIVE_SUPPORTED_OP_NAMES)
    )

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

    def training_step(self, *args, **kwargs):
        self.resample()
        return self.model.training_step(*args, **kwargs)

    def export(self) -> dict[str, Any]:
        """
        Export of Random one-shot. It will return an arbitrary architecture.
        """
        warnings.warn(
            'Direct export from RandomOneShot returns an arbitrary architecture. '
            'Sampling the best architecture from this trained supernet is another search process. '
            'Users need to do another search based on the checkpoint of the one-shot strategy.',
            UserWarning
        )
        return super().export()


class EnasLightningModule(RandomSamplingLightningModule):
    _enas_note = """
    RL controller learns to generate the best network on a super-net. See `ENAS paper <https://arxiv.org/abs/1802.03268>`__.

    There are 2 steps in an epoch.

    - Firstly, training model parameters.
    - Secondly, training ENAS RL agent. The agent will produce a sample of model architecture to get the best reward.

    .. note::

       ENAS requires the evaluator to report metrics via ``self.log`` in its ``validation_step``.
       See explanation of ``reward_metric_name`` for details.

    The supported mutation primitives of ENAS are:

    * :class:`nni.retiarii.nn.pytorch.LayerChoice`.
    * :class:`nni.retiarii.nn.pytorch.InputChoice`.
    * :class:`nni.retiarii.nn.pytorch.ValueChoice` (only when used in {supported_ops}).
    * :class:`nni.retiarii.nn.pytorch.Repeat`.
    * :class:`nni.retiarii.nn.pytorch.Cell`.
    * :class:`nni.retiarii.nn.pytorch.NasBench201Cell`.

    {{module_notes}}

    {optimization_note}

    Parameters
    ----------
    {{module_params}}
    {base_params}
    ctrl_kwargs : dict
        Optional kwargs that will be passed to :class:`~nni.retiarii.oneshot.pytorch.enas.ReinforceController`.
    entropy_weight : float
        Weight of sample entropy loss in RL.
    skip_weight : float
        Weight of skip penalty loss. See :class:`~nni.retiarii.oneshot.pytorch.enas.ReinforceController` for details.
    baseline_decay : float
        Decay factor of reward baseline, which is used to normalize the reward in RL.
        At each step, the new reward baseline will be equal to ``baseline_decay * baseline_old + reward * (1 - baseline_decay)``.
    ctrl_steps_aggregate : int
        Number of steps for which the gradients will be accumulated,
        before updating the weights of RL controller.
    ctrl_grad_clip : float
        Gradient clipping value of controller.
    log_prob_every_n_step : int
        Log the probability of choices every N steps. Useful for visualization and debugging.
    reward_metric_name : str or None
        The name of the metric which is treated as reward.
        This will be not effective when there's only one metric returned from evaluator.
        If there are multiple, by default, it will find the metric with key name ``default``.
        If reward_metric_name is specified, it will find reward_metric_name.
        Otherwise it raises an exception indicating multiple metrics are found.
    """.format(
        base_params=BaseOneShotLightningModule._mutation_hooks_note,
        supported_ops=', '.join(NATIVE_SUPPORTED_OP_NAMES),
        optimization_note=MANUAL_OPTIMIZATION_NOTE
    )

    __doc__ = _enas_note.format(
        module_notes='``ENASModule`` should be trained with :class:`nni.retiarii.oneshot.pytorch.dataloader.ConcatLoader`.',
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
                 log_prob_every_n_step: int = 10,
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
        self.log_prob_every_n_step = log_prob_every_n_step
        self.reward_metric_name = reward_metric_name

    def configure_architecture_optimizers(self):
        return optim.Adam(self.controller.parameters(), lr=3.5e-4)

    def training_step(self, batch_packed, batch_idx):
        # The received batch is a tuple of (data, "train" | "val")
        batch, mode = batch_packed

        if mode == 'train':
            # train model params
            with torch.no_grad():
                self.resample()
            step_output = self.model.training_step(batch, batch_idx)
            w_step_loss = step_output['loss'] if isinstance(step_output, dict) else step_output
            self.advance_optimization(w_step_loss, batch_idx)

        else:
            # train ENAS agent

            # Run a sample to retrieve the reward
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
                                   f'Please try to set ``reward_metric_name`` to be one of the keys listed above. '
                                   f'If it is not working use self.log to report metrics with the specified key ``{metric_name}`` '
                                   'in validation_step, and remember to set on_step=True.')
                metric = self.trainer.callback_metrics[metric_name]
            reward: float = metric.item()

            # Compute the loss and run back propagation
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

                # Update the controller and zero out its gradients
                arc_opt = cast(optim.Optimizer, self.architecture_optimizers())
                arc_opt.step()
                arc_opt.zero_grad()

        self.advance_lr_schedulers(batch_idx)

        if (batch_idx + 1) % self.log_prob_every_n_step == 0:
            with torch.no_grad():
                self.log_dict({'prob/' + k: v for k, v in self.export_probs().items()})

        return step_output

    def on_train_epoch_start(self):
        # Always zero out the gradients of ENAS controller at the beginning of epochs.
        arc_opt = self.architecture_optimizers()
        if not isinstance(arc_opt, optim.Optimizer):
            raise TypeError(f'Expect arc_opt to be a single Optimizer, but found: {arc_opt}')
        arc_opt.zero_grad()

        return self.model.on_train_epoch_start()

    def resample(self):
        """Resample the architecture with ENAS controller."""
        sample = self.controller.resample()
        result = self._interpret_controller_sampling_result(sample)
        for module in self.nas_modules:
            module.resample(memo=result)
        return result

    def export_probs(self):
        """Export the probability from ENAS controller directly."""
        sample = self.controller.resample(return_prob=True)
        result = self._interpret_controller_probability_result(sample)
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

    def _interpret_controller_probability_result(self, sample: dict[str, list[float]]) -> dict[str, Any]:
        """Convert ``{label: [prob1, prob2, prob3]} to ``{label/choice: prob}``"""
        space_spec = self.search_space_spec()
        result = {}
        for key in list(sample.keys()):
            if len(space_spec[key].values) != len(sample[key]):
                raise ValueError(f'Expect {space_spec[key].values} to be of the same length as {sample[key]}')
            for value, weight in zip(space_spec[key].values, sample[key]):
                result[f'{key}/{value}'] = weight
        return result
