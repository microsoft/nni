# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import warnings
from typing import Any, Iterable, List, cast, TYPE_CHECKING

import torch.optim as optim
import torch.nn as nn
from torch.optim import Optimizer

import nni.nas.nn.pytorch as nas_nn
from nni.nas.evaluator.pytorch import LightningModule, Trainer
from nni.mutable import Sample
from .supermodule.base import BaseSuperNetModule

if TYPE_CHECKING:
    from pytorch_lightning.core.optimizer import LightningOptimizer

__all__ = [
    'BaseSuperNetModule',
    'BaseOneShotLightningModule',
]


class BaseOneShotLightningModule(LightningModule):

    _inner_module_note = """inner_module : pytorch_lightning.LightningModule
        It's a `LightningModule <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`__
        that defines computations, train/val loops, optimizers in a single class.
        When used in NNI, the ``inner_module`` is the combination of instances of evaluator + base model
        (to be precise, a base model wrapped with LightningModule in evaluator).
    """

    __doc__ = """
    The base class for all one-shot NAS modules.

    :class:`BaseOneShotLightningModule` is implemented as a subclass of :class:`~nni.nas.evaluator.pytorch.Lightning`,
    to be make it deceptively look like a lightning module to the trainer.
    It's actually a wrapper of the lightning module in evaluator.
    The composition of different lightning modules is as follows::

        BaseOneShotLightningModule       <- Current class (one-shot logics)
            |_ evaluator.LightningModule <- Part of evaluator (basic training logics)
                |_ user's model          <- Model space, transformed to a supernet by current class.

    The base class implemented several essential utilities,
    such as preprocessing user's model, redirecting lightning hooks for user's model,
    configuring optimizers and exporting NAS result are implemented in this class.

    Attributes
    ----------
    training_module
        PyTorch lightning module, which defines the training recipe (the lightning module part in evaluator).

    Parameters
    ----------
    """ + _inner_module_note

    trainer: Trainer

    @property
    def automatic_optimization(self) -> bool:
        return False

    def __init__(self, training_module: LightningModule):
        super().__init__()
        self.training_module = training_module

    def supernet_modules(self) -> Iterable[BaseSuperNetModule]:
        """Return all supernet modules in the model space."""
        for module in self.modules():
            if isinstance(module, BaseSuperNetModule):
                yield module

    @property
    def model(self) -> nas_nn.ModelSpace:
        """Return the model space defined by the user.

        The model space is not guaranteed to have been transformed into a one-shot supernet.
        For instance, when ``__init__`` hasn't completed, the model space will still be the original one.
        """
        model = self.training_module.model
        if not isinstance(model, nas_nn.ModelSpace):
            raise TypeError(f'The model is expected to be a valid PyTorch model space, but got {type(model)}')
        return model

    def set_model(self, model: nn.Module) -> None:
        """Set the model space to be searched."""
        self.training_module.set_model(model)

    def resample(self) -> Sample:
        """Trigger the resample for each :meth:`supernet_modules`.
        Sometimes (e.g., in differentiable cases), it does nothing.

        Returns
        -------
        dict
            Sampled architecture.
        """
        result = {}
        for module in self.supernet_modules():
            result.update(module.resample(memo=result))
        return result

    def export(self) -> Sample:
        """
        Export the NAS result, ideally the best choice of each :meth:`supernet_modules`.
        You may implement an ``export`` method for your customized :meth:`supernet_modules`.

        Returns
        --------
        dict
            Keys are labels of mutables, and values are the choice indices of them.
        """
        result = {}
        for module in self.supernet_modules():
            result.update(module.export(memo=result))
        return result

    def export_probs(self) -> Sample:
        """
        Export the probability of every choice in the search space got chosen.

        .. note:: If such method of some modules is not implemented, they will be simply ignored.

        Returns
        -------
        dict
            In most cases, keys are labels of the mutables, while values are a dict,
            whose key is the choice and value is the probability of it being chosen.
        """
        result = {}
        for module in self.supernet_modules():
            try:
                result.update(module.export_probs(memo=result))
            except NotImplementedError:
                warnings.warn(
                    'Some super-modules you have used did not implement export_probs. You might find some logs are missing.',
                    UserWarning
                )
        return result

    def log_probs(self, probs: Sample) -> None:
        """
        Write the probability of every choice to the logger.
        (nothing related to log-probability stuff).

        Parameters
        ----------
        probs
            The result of :meth:`export_probs`.
        """
        # Flatten the probabilities and write to all the loggers.
        # According to my test, this works better than `add_scalars` and `add_histogram`.

        self.log_dict({
            f'prob/{label}/{value}': logit
            for label, dist in probs.items()
            for value, logit in dist.items()
        })

    def forward(self, x):
        return self.training_module(x)

    def configure_optimizers(self) -> Any:
        """
        Transparently configure optimizers for the inner model,
        unless one-shot algorithm has its own optimizer (via :meth:`configure_architecture_optimizers`),
        in which case, the optimizer will be appended to the list.

        The return value is still one of the 6 types defined in PyTorch-Lightning.
        """
        arch_optimizers = self.configure_architecture_optimizers() or []
        if not arch_optimizers:  # no architecture optimizer available
            return self.training_module.configure_optimizers()

        if isinstance(arch_optimizers, optim.Optimizer):
            arch_optimizers = [arch_optimizers]

        # Set the flag to True so that they can differ from other optimizers
        for optimizer in arch_optimizers:
            optimizer.is_arch_optimizer = True  # type: ignore

        optim_conf: Any = self.training_module.configure_optimizers()

        optim_conf = self.postprocess_weight_optimizers(optim_conf)

        # 0. optimizer is none
        if optim_conf is None:
            return arch_optimizers
        # 1. single optimizer
        if isinstance(optim_conf, Optimizer):
            return [optim_conf] + arch_optimizers
        # 2. two lists, optimizer + lr schedulers
        if (
            isinstance(optim_conf, (list, tuple))
            and len(optim_conf) == 2
            and isinstance(optim_conf[0], list)
            and all(isinstance(opt, Optimizer) for opt in optim_conf[0])
        ):
            return list(optim_conf[0]) + arch_optimizers, optim_conf[1]
        # 3. single dictionary
        if isinstance(optim_conf, dict):
            return [optim_conf] + [{'optimizer': optimizer} for optimizer in arch_optimizers]
        # 4. multiple dictionaries
        if isinstance(optim_conf, (list, tuple)) and all(isinstance(d, dict) for d in optim_conf):
            return list(optim_conf) + [{'optimizer': optimizer} for optimizer in arch_optimizers]
        # 5. single list or tuple, multiple optimizer
        if isinstance(optim_conf, (list, tuple)) and all(isinstance(opt, Optimizer) for opt in optim_conf):
            return list(optim_conf) + arch_optimizers
        # unknown configuration
        warnings.warn('Unknown optimizer configuration. Architecture optimizers will be ignored. Strategy might fail.', UserWarning)

        return optim_conf

    def setup(self, stage: str = cast(str, None)):  # add default value to be backward-compatible
        # redirect the access to trainer/log to this module
        # but note that we might be missing other attributes,
        # which could potentially be a problem
        self.training_module.trainer = self.trainer  # type: ignore
        self.training_module.log = self.log

        # Reset the optimizer progress (only once at the very beginning)
        self._optimizer_progress = 0

        return self.training_module.setup(stage)

    def teardown(self, stage: str = cast(str, None)):
        return self.training_module.teardown(stage)

    def postprocess_weight_optimizers(self, optimizers: Any) -> Any:
        """
        Some subclasss need to modify the original optimizers. This is where it should be done.
        For example, differentiable algorithms might not want the architecture weights to be inside the weight optimizers.

        Returns
        -------
        By default, it return the original object.
        """
        return optimizers

    def configure_architecture_optimizers(self) -> list[optim.Optimizer] | optim.Optimizer | None:
        """
        Hook kept for subclasses. A specific NAS method inheriting this base class should return its architecture optimizers here
        if architecture parameters are needed. Note that lr schedulers are not supported now for architecture_optimizers.

        Returns
        -------
        Optimizers used by a specific NAS algorithm. Return None if no architecture optimizers are needed.
        """
        return None

    def advance_optimization(
        self,
        loss: Any,
        batch_idx: int,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None
    ):
        """
        Run the optimizer defined in evaluators, when manual optimization is turned on.

        Call this method when the model should be optimized.
        To keep it as neat as possible, we only implement the basic ``zero_grad``, ``backward``, ``grad_clip``, and ``step`` here.
        Many hooks and pre/post-processing are omitted.
        Inherit this method if you need more advanced behavior.

        The full optimizer step could be found
        `here <https://github.com/Lightning-AI/lightning/blob/0e531283/src/pytorch_lightning/loops/optimization/optimizer_loop.py>`__.
        We only implement part of the optimizer loop here.

        Parameters
        ----------
        batch_idx: int
            The current batch index.
        """
        if self.automatic_optimization:
            raise ValueError('This method should not be used when automatic optimization is turned on.')

        # Has to be optimizers() here (to get LightningOptimizer)
        # instead of trainer.optimizers (raw optimizers),
        # because otherwise optim_progress is incorrect.
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        # Filter out optimizers for architecture parameters.
        optimizers = cast(List[Optimizer], [opt for opt in optimizers if not getattr(opt, 'is_arch_optimizer', False)])

        if hasattr(self.trainer, 'optimizer_frequencies'):  # lightning < 2
            self._legacy_advance_optimization(loss, batch_idx, optimizers, gradient_clip_val, gradient_clip_algorithm)
        else:
            if not self.training_module.automatic_optimization:
                raise ValueError('Evaluator module with manual optimization is not compatible with one-shot algorithms.')
            if len(optimizers) != 1:
                raise ValueError('More than one optimizer returned by evaluator. This is not supported in NAS.')
            optimizer = optimizers[0]

            # There should be many before/after hooks called here, but they are omitted in this implementation.
            # 1. zero gradient
            self.training_module.optimizer_zero_grad(self.trainer.current_epoch, batch_idx, optimizer)
            # 2. backward
            self.manual_backward(loss)
            # 3. grad clip
            self.training_module.configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
            # 4. optimizer step
            self.training_module.optimizer_step(self.trainer.current_epoch, batch_idx, optimizer)

        self._optimizer_progress += 1

    def _legacy_advance_optimization(
        self,
        loss: Any,
        batch_idx: int,
        optimizers: list[Optimizer],
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None
    ):
        """:meth:`advance_optimization` for Lightning 1.x."""

        if self.trainer.optimizer_frequencies:  # type: ignore
            warnings.warn('optimizer_frequencies is not supported in NAS. It will be ignored.', UserWarning)

        opt_idx = self._optimizer_progress % len(optimizers)
        optimizer = cast(Optimizer, optimizers[opt_idx])  # LightningOptimizer has the same interface as Optimizer.

        # There should be many before/after hooks called here, but they are omitted in this implementation.
        # 1. zero gradient
        self.training_module.optimizer_zero_grad(self.trainer.current_epoch, batch_idx, optimizer, opt_idx)  # type: ignore
        # 2. backward
        self.manual_backward(loss)
        # 3. grad clip
        self.training_module.configure_gradient_clipping(optimizer, opt_idx, gradient_clip_val, gradient_clip_algorithm)  # type: ignore
        # 4. optimizer step
        self.training_module.optimizer_step(self.trainer.current_epoch, batch_idx, optimizer, opt_idx)  # type: ignore

    def advance_lr_schedulers(self, batch_idx: int):
        """
        Advance the learning rates, when manual optimization is turned on.

        The full implementation is
        `here <https://github.com/Lightning-AI/lightning/blob/0e531283/src/pytorch_lightning/loops/epoch/training_epoch_loop.py>`__.
        We only include a partial implementation here.
        Advanced features like Reduce-lr-on-plateau are not supported.
        """
        if self.automatic_optimization:
            raise ValueError('This method should not be used when automatic optimization is turned on.')

        self._advance_lr_schedulers_impl(batch_idx, 'step')
        if self.trainer.is_last_batch:
            self._advance_lr_schedulers_impl(batch_idx, 'epoch')

    def _advance_lr_schedulers_impl(self, batch_idx: int, interval: str):
        current_idx = batch_idx if interval == 'step' else self.trainer.current_epoch
        current_idx += 1  # account for both batch and epoch starts from 0

        try:
            # lightning >= 1.6
            for config in self.trainer.lr_scheduler_configs:
                if hasattr(config, 'opt_idx'):
                    # lightning < 2.0
                    scheduler, opt_idx = config.scheduler, config.opt_idx  # type: ignore
                else:
                    scheduler, opt_idx = config.scheduler, None
                if config.reduce_on_plateau:
                    warnings.warn('Reduce-lr-on-plateau is not supported in NAS. It will be ignored.', UserWarning)
                if config.interval == interval and current_idx % config.frequency == 0:
                    if opt_idx is not None:
                        self.training_module.lr_scheduler_step(cast(Any, scheduler), cast(int, opt_idx), None)  # type: ignore
                    else:
                        self.training_module.lr_scheduler_step(cast(Any, scheduler), None)
        except AttributeError:
            # lightning < 1.6
            for lr_scheduler in self.trainer.lr_schedulers:  # type: ignore
                if lr_scheduler['reduce_on_plateau']:
                    warnings.warn('Reduce-lr-on-plateau is not supported in NAS. It will be ignored.', UserWarning)
                if lr_scheduler['interval'] == interval and current_idx % lr_scheduler['frequency']:
                    lr_scheduler['scheduler'].step()

    def architecture_optimizers(self) -> list[LightningOptimizer] | LightningOptimizer | None:
        """
        Get the optimizers configured in :meth:`configure_architecture_optimizers`.

        Return type would be LightningOptimizer or list of LightningOptimizer.
        """
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        optimizers = [opt for opt in optimizers if getattr(opt, 'is_arch_optimizer', False)]
        if not optimizers:
            return None
        if len(optimizers) == 1:
            return optimizers[0]
        return optimizers  # type: ignore

    # The following methods redirects the callbacks to inner module.
    # It's not the complete list though.
    # More methods can be added if needed.

    def on_train_start(self):
        return self.training_module.on_train_start()

    def on_train_end(self):
        return self.training_module.on_train_end()

    def on_validation_start(self):
        return self.training_module.on_validation_start()

    def on_validation_end(self):
        return self.training_module.on_validation_end()

    def on_fit_start(self):
        return self.training_module.on_fit_start()

    def on_fit_end(self):
        return self.training_module.on_fit_end()

    def on_train_batch_start(self, batch, batch_idx, *args, **kwargs):
        return self.training_module.on_train_batch_start(batch, batch_idx, *args, **kwargs)

    def on_train_batch_end(self, outputs, batch, batch_idx, *args, **kwargs):
        return self.training_module.on_train_batch_end(outputs, batch, batch_idx, *args, **kwargs)

    def on_train_epoch_start(self):
        return self.training_module.on_train_epoch_start()

    def on_train_epoch_end(self):
        return self.training_module.on_train_epoch_end()

    def on_before_backward(self, loss):
        return self.training_module.on_before_backward(loss)

    def on_after_backward(self):
        return self.training_module.on_after_backward()
