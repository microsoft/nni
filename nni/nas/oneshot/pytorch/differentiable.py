# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Experimental version of differentiable one-shot implementation."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.optim as optim

from .base_lightning import BaseOneShotLightningModule, MANUAL_OPTIMIZATION_NOTE, MutationHook, no_default_hook
from .supermodule.differentiable import (
    DifferentiableMixedLayer, DifferentiableMixedInput,
    MixedOpDifferentiablePolicy, GumbelSoftmax,
    DifferentiableMixedCell, DifferentiableMixedRepeat
)
from .supermodule.proxyless import ProxylessMixedInput, ProxylessMixedLayer
from .supermodule.operation import NATIVE_MIXED_OPERATIONS, NATIVE_SUPPORTED_OP_NAMES


class DartsLightningModule(BaseOneShotLightningModule):
    _darts_note = """
    Continuous relaxation of the architecture representation, allowing efficient search of the architecture using gradient descent.
    `Reference <https://arxiv.org/abs/1806.09055>`__.

    DARTS algorithm is one of the most fundamental one-shot algorithm.
    DARTS repeats iterations, where each iteration consists of 2 training phases.
    The phase 1 is architecture step, in which model parameters are frozen and the architecture parameters are trained.
    The phase 2 is model step, in which architecture parameters are frozen and model parameters are trained.
    In both phases, ``training_step`` of the Lightning evaluator will be used.

    The current implementation corresponds to DARTS (1st order) in paper.
    Second order (unrolled 2nd-order derivatives) is not supported yet.

    .. versionadded:: 2.8

       Supports searching for ValueChoices on operations, with the technique described in
       `FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions <https://arxiv.org/abs/2004.05565>`__.
       One difference is that, in DARTS, we are using Softmax instead of GumbelSoftmax.

    The supported mutation primitives of DARTS are:

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
    arc_learning_rate : float
        Learning rate for architecture optimizer. Default: 3.0e-4
    gradient_clip_val : float
        Clip gradients before optimizing models at each step. Default: None
    """.format(
        base_params=BaseOneShotLightningModule._mutation_hooks_note,
        supported_ops=', '.join(NATIVE_SUPPORTED_OP_NAMES),
        optimization_note=MANUAL_OPTIMIZATION_NOTE
    )

    __doc__ = _darts_note.format(
        module_notes='The DARTS Module should be trained with :class:`pytorch_lightning.trainer.supporters.CombinedLoader`.',
        module_params=BaseOneShotLightningModule._inner_module_note,
    )

    def default_mutation_hooks(self) -> list[MutationHook]:
        """Replace modules with differentiable versions"""
        hooks = [
            DifferentiableMixedLayer.mutate,
            DifferentiableMixedInput.mutate,
            DifferentiableMixedCell.mutate,
            DifferentiableMixedRepeat.mutate,
        ]
        hooks += [operation.mutate for operation in NATIVE_MIXED_OPERATIONS]
        hooks.append(no_default_hook)
        return hooks

    def mutate_kwargs(self):
        """Use differentiable strategy for mixed operations."""
        return {
            'mixed_op_sampling': MixedOpDifferentiablePolicy
        }

    def __init__(self, inner_module: pl.LightningModule,
                 mutation_hooks: list[MutationHook] | None = None,
                 arc_learning_rate: float = 3.0E-4,
                 gradient_clip_val: float | None = None):
        self.arc_learning_rate = arc_learning_rate
        self.gradient_clip_val = gradient_clip_val
        super().__init__(inner_module, mutation_hooks=mutation_hooks)

    def training_step(self, batch, batch_idx):
        # grad manually
        arc_optim = self.architecture_optimizers()
        if not isinstance(arc_optim, optim.Optimizer):
            raise TypeError(f'Expect arc_optim to be a single Optimizer, but found: {arc_optim}')

        # DARTS strategy makes sure that ``train`` and ``val`` must be in the batch
        trn_batch = batch['train']
        val_batch = batch['val']

        # phase 1: architecture step
        # The _resample hook is kept for some darts-based NAS methods like proxyless.
        # See code of those methods for details.
        self.resample()
        arc_optim.zero_grad()
        arc_step_loss = self.model.training_step(val_batch, 2 * batch_idx)
        if isinstance(arc_step_loss, dict):
            arc_step_loss = arc_step_loss['loss']
        self.manual_backward(arc_step_loss)
        arc_optim.step()

        # phase 2: model step
        self.resample()
        loss_and_metrics = self.model.training_step(trn_batch, 2 * batch_idx + 1)
        w_step_loss = loss_and_metrics['loss'] if isinstance(loss_and_metrics, dict) else loss_and_metrics
        self.advance_optimization(w_step_loss, batch_idx, self.gradient_clip_val)

        # Update learning rates
        self.advance_lr_schedulers(batch_idx)

        self.log_dict({'prob/' + k: v for k, v in self.export_probs().items()})

        return loss_and_metrics

    def configure_architecture_optimizers(self):
        # The alpha in DartsXXXChoices are the architecture parameters of DARTS. They share one optimizer.
        ctrl_params = []
        for m in self.nas_modules:
            ctrl_params += list(m.parameters(arch=True))  # type: ignore
        # Follow the hyper-parameters used in
        # https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/architect.py#L17
        params = list(set(ctrl_params))
        if not params:
            raise ValueError('No architecture parameters found. Nothing to search.')
        ctrl_optim = torch.optim.Adam(params, 3.e-4, betas=(0.5, 0.999), weight_decay=1.0E-3)
        return ctrl_optim


class ProxylessLightningModule(DartsLightningModule):
    _proxyless_note = """
    A low-memory-consuming optimized version of differentiable architecture search. See `reference <https://arxiv.org/abs/1812.00332>`__.

    This is a DARTS-based method that resamples the architecture to reduce memory consumption.
    Essentially, it samples one path on forward,
    and implements its own backward to update the architecture parameters based on only one path.

    The supported mutation primitives of Proxyless are:

    * :class:`nni.retiarii.nn.pytorch.LayerChoice`.
    * :class:`nni.retiarii.nn.pytorch.InputChoice`.

    {{module_notes}}

    {optimization_note}

    Parameters
    ----------
    {{module_params}}
    {base_params}
    arc_learning_rate : float
        Learning rate for architecture optimizer. Default: 3.0e-4
    gradient_clip_val : float
        Clip gradients before optimizing models at each step. Default: None
    """.format(
        base_params=BaseOneShotLightningModule._mutation_hooks_note,
        optimization_note=MANUAL_OPTIMIZATION_NOTE
    )

    __doc__ = _proxyless_note.format(
        module_notes='This module should be trained with :class:`pytorch_lightning.trainer.supporters.CombinedLoader`.',
        module_params=BaseOneShotLightningModule._inner_module_note,
    )

    def default_mutation_hooks(self) -> list[MutationHook]:
        """Replace modules with gumbel-differentiable versions"""
        hooks = [
            ProxylessMixedLayer.mutate,
            ProxylessMixedInput.mutate,
            no_default_hook,
        ]
        # FIXME: no support for mixed operation currently
        return hooks


class GumbelDartsLightningModule(DartsLightningModule):
    _gumbel_darts_note = """
    Choose the best block by using Gumbel Softmax random sampling and differentiable training.
    See `FBNet <https://arxiv.org/abs/1812.03443>`__ and `SNAS <https://arxiv.org/abs/1812.09926>`__.

    This is a DARTS-based method that uses gumbel-softmax to simulate one-hot distribution.
    Essentially, it tries to mimick the behavior of sampling one path on forward by gradually
    cool down the temperature, aiming to bridge the gap between differentiable architecture weights and
    discretization of architectures.

    .. versionadded:: 2.8
    
       Supports searching for ValueChoices on operations, with the technique described in
       `FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions <https://arxiv.org/abs/2004.05565>`__.

    The supported mutation primitives of GumbelDARTS are:

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
    gumbel_temperature : float
        The initial temperature used in gumbel-softmax.
    use_temp_anneal : bool
        If true, a linear annealing will be applied to ``gumbel_temperature``.
        Otherwise, run at a fixed temperature. See `SNAS <https://arxiv.org/abs/1812.09926>`__ for details.
        Default is false.
    min_temp : float
        The minimal temperature for annealing. No need to set this if you set ``use_temp_anneal`` False.
    arc_learning_rate : float
        Learning rate for architecture optimizer. Default: 3.0e-4
    gradient_clip_val : float
        Clip gradients before optimizing models at each step. Default: None
    """.format(
        base_params=BaseOneShotLightningModule._mutation_hooks_note,
        supported_ops=', '.join(NATIVE_SUPPORTED_OP_NAMES),
        optimization_note=MANUAL_OPTIMIZATION_NOTE
    )

    def mutate_kwargs(self):
        """Use gumbel softmax."""
        return {
            'mixed_op_sampling': MixedOpDifferentiablePolicy,
            'softmax': GumbelSoftmax(),
        }

    def __init__(self, inner_module,
                 mutation_hooks: list[MutationHook] | None = None,
                 arc_learning_rate: float = 3.0e-4,
                 gradient_clip_val: float | None = None,
                 gumbel_temperature: float = 1.,
                 use_temp_anneal: bool = False,
                 min_temp: float = .33):
        super().__init__(inner_module, mutation_hooks, arc_learning_rate=arc_learning_rate, gradient_clip_val=gradient_clip_val)
        self.temp = gumbel_temperature
        self.init_temp = gumbel_temperature
        self.use_temp_anneal = use_temp_anneal
        self.min_temp = min_temp

    def on_train_epoch_start(self):
        if self.use_temp_anneal:
            self.temp = (1 - self.trainer.current_epoch / self.trainer.max_epochs) * (self.init_temp - self.min_temp) + self.min_temp
            self.temp = max(self.temp, self.min_temp)

        self.log('gumbel_temperature', self.temp)

        for module in self.nas_modules:
            if hasattr(module, '_softmax') and isinstance(module, GumbelSoftmax):
                module._softmax.tau = self.temp  # type: ignore

        return self.model.on_train_epoch_start()
