# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Experimental version of differentiable one-shot implementation."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.optim as optim

from .base_lightning import BaseOneShotLightningModule, MutationHook, no_default_hook
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

    The current implementation is for DARTS in first order. Second order (unrolled) is not supported yet.

    *New in v2.8*: Supports searching for ValueChoices on operations, with the technique described in
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

    Parameters
    ----------
    {{module_params}}
    {base_params}
    arc_learning_rate : float
        Learning rate for architecture optimizer. Default: 3.0e-4
    """.format(
        base_params=BaseOneShotLightningModule._mutation_hooks_note,
        supported_ops=', '.join(NATIVE_SUPPORTED_OP_NAMES)
    )

    __doc__ = _darts_note.format(
        module_notes='The DARTS Module should be trained with :class:`nni.retiarii.oneshot.utils.InterleavedTrainValDataLoader`.',
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
                 arc_learning_rate: float = 3.0E-4):
        self.arc_learning_rate = arc_learning_rate
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
        self.finalize_grad()
        arc_optim.step()

        # phase 2: model step
        self.resample()
        self.call_weight_optimizers('zero_grad')
        loss_and_metrics = self.model.training_step(trn_batch, 2 * batch_idx + 1)
        w_step_loss = loss_and_metrics['loss'] \
            if isinstance(loss_and_metrics, dict) else loss_and_metrics
        self.manual_backward(w_step_loss)
        self.call_weight_optimizers('step')

        self.call_lr_schedulers(batch_idx)

        return loss_and_metrics

    def finalize_grad(self):
        # Note: This hook is currently kept for Proxyless NAS.
        pass

    def configure_architecture_optimizers(self):
        # The alpha in DartsXXXChoices are the architecture parameters of DARTS. They share one optimizer.
        ctrl_params = []
        for m in self.nas_modules:
            ctrl_params += list(m.parameters(arch=True))  # type: ignore
        ctrl_optim = torch.optim.Adam(list(set(ctrl_params)), 3.e-4, betas=(0.5, 0.999),
                                      weight_decay=1.0E-3)
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

    Parameters
    ----------
    {{module_params}}
    {base_params}
    arc_learning_rate : float
        Learning rate for architecture optimizer. Default: 3.0e-4
    """.format(base_params=BaseOneShotLightningModule._mutation_hooks_note)

    __doc__ = _proxyless_note.format(
        module_notes='This module should be trained with :class:`nni.retiarii.oneshot.pytorch.utils.InterleavedTrainValDataLoader`.',
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

    def finalize_grad(self):
        for m in self.nas_modules:
            m.finalize_grad()  # type: ignore


class GumbelDartsLightningModule(DartsLightningModule):
    _gumbel_darts_note = """
    Choose the best block by using Gumbel Softmax random sampling and differentiable training.
    See `FBNet <https://arxiv.org/abs/1812.03443>`__ and `SNAS <https://arxiv.org/abs/1812.09926>`__.

    This is a DARTS-based method that uses gumbel-softmax to simulate one-hot distribution.
    Essentially, it samples one path on forward,
    and implements its own backward to update the architecture parameters based on only one path.

    *New in v2.8*: Supports searching for ValueChoices on operations, with the technique described in
    `FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions <https://arxiv.org/abs/2004.05565>`__.

    The supported mutation primitives of GumbelDARTS are:

    * :class:`nni.retiarii.nn.pytorch.LayerChoice`.
    * :class:`nni.retiarii.nn.pytorch.InputChoice`.
    * :class:`nni.retiarii.nn.pytorch.ValueChoice` (only when used in {supported_ops}).
    * :class:`nni.retiarii.nn.pytorch.Repeat`.
    * :class:`nni.retiarii.nn.pytorch.Cell`.
    * :class:`nni.retiarii.nn.pytorch.NasBench201Cell`.

    {{module_notes}}

    Parameters
    ----------
    {{module_params}}
    {base_params}
    gumbel_temperature : float
        The initial temperature used in gumbel-softmax.
    use_temp_anneal : bool
        If true, a linear annealing will be applied to ``gumbel_temperature``.
        Otherwise, run at a fixed temperature. See `SNAS <https://arxiv.org/abs/1812.09926>`__ for details.
    min_temp : float
        The minimal temperature for annealing. No need to set this if you set ``use_temp_anneal`` False.
    arc_learning_rate : float
        Learning rate for architecture optimizer. Default: 3.0e-4
    """.format(
        base_params=BaseOneShotLightningModule._mutation_hooks_note,
        supported_ops=', '.join(NATIVE_SUPPORTED_OP_NAMES)
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
                 gumbel_temperature: float = 1.,
                 use_temp_anneal: bool = False,
                 min_temp: float = .33):
        super().__init__(inner_module, mutation_hooks, arc_learning_rate=arc_learning_rate)
        self.temp = gumbel_temperature
        self.init_temp = gumbel_temperature
        self.use_temp_anneal = use_temp_anneal
        self.min_temp = min_temp

    def on_train_epoch_end(self):
        if self.use_temp_anneal:
            self.temp = (1 - self.trainer.current_epoch / self.trainer.max_epochs) * (self.init_temp - self.min_temp) + self.min_temp
            self.temp = max(self.temp, self.min_temp)

        for module in self.nas_modules:
            if hasattr(module, '_softmax'):
                module._softmax.temp = self.temp  # type: ignore

        return self.model.on_train_epoch_end()
