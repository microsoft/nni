# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import warnings
from itertools import chain
from typing import Callable, Any, Dict, Union, Tuple, cast

import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
from torch.optim import Optimizer

import nni.nas.nn.pytorch as nas_nn
from nni.common.hpo_utils import ParameterSpec
from nni.common.serializer import is_traceable
from nni.nas.nn.pytorch.choice import ValueChoiceX
from .supermodule.base import BaseSuperNetModule

__all__ = [
    'MANUAL_OPTIMIZATION_NOTE',
    'MutationHook',
    'BaseSuperNetModule',
    'BaseOneShotLightningModule',
    'traverse_and_mutate_submodules',
    'no_default_hook'
]


MANUAL_OPTIMIZATION_NOTE = """
    .. warning::

        The strategy, under the hood, creates a Lightning module that wraps the Lightning module defined in evaluator,
        and enables `Manual optimization <https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html>`_,
        although we assume **the inner evaluator has enabled automatic optimization**.
        We call the optimizers and schedulers configured in evaluator, following the definition in Lightning at best effort,
        but we make no guarantee that the behaviors are exactly same as automatic optimization.
        We call :meth:`~BaseSuperNetModule.advance_optimization` and :meth:`~BaseSuperNetModule.advance_lr_schedulers`
        to invoke the optimizers and schedulers configured in evaluators.
        Moreover, some advanced features like gradient clipping will not be supported.
        If you encounter any issues, please contact us by `creating an issue <https://github.com/microsoft/nni/issues>`_.

"""


MutationHook = Callable[[nn.Module, str, Dict[str, Any], Dict[str, Any]], Union[nn.Module, bool, Tuple[nn.Module, bool]]]


def traverse_and_mutate_submodules(
    root_module: nn.Module, hooks: list[MutationHook], mutate_kwargs: dict[str, Any], topdown: bool = True
) -> list[BaseSuperNetModule]:
    """
    Traverse the module-tree of ``root_module``, and call ``hooks`` on every tree node.

    Parameters
    ----------
    root_module : nn.Module
        User-defined model space.
        Since this method is called in the ``__init__`` of :class:`BaseOneShotLightningModule`,
        it's usually a ``pytorch_lightning.LightningModule``.
        The mutation will be in-place on ``root_module``.
    hooks : list[MutationHook]
        List of mutation hooks. See :class:`BaseOneShotLightningModule` for how to write hooks.
        When a hook returns an module, the module will be replaced (mutated) to the new module.
    mutate_kwargs : dict
        Extra keyword arguments passed to hooks.
    topdown : bool, default = False
        If topdown is true, hooks are first called, before traversing its sub-module (i.e., pre-order DFS).
        Otherwise, sub-modules are first traversed, before calling hooks on this node (i.e., post-order DFS).

    Returns
    ----------
    modules : dict[str, nn.Module]
        The replace result.
    """
    memo = {}

    module_list = []
    def apply(m):
        # Need to call list() here because the loop body might replace some children in-place.
        for name, child in list(m.named_children()):
            # post-order DFS
            if not topdown:
                apply(child)

            mutate_result = None

            for hook in hooks:
                hook_suggest = hook(child, name, memo, mutate_kwargs)

                # parse the mutate result
                if isinstance(hook_suggest, tuple):
                    hook_suggest, suppress = hook_suggest
                elif hook_suggest is True:
                    hook_suggest, suppress = None, True
                elif not hook_suggest:  # none / false
                    hook_suggest, suppress = None, False
                elif isinstance(hook_suggest, nn.Module):
                    suppress = True
                else:
                    raise TypeError(f'Mutation hook returned {hook_suggest} of unsupported type: {type(hook_suggest)}.')

                if hook_suggest is not None:
                    if not isinstance(hook_suggest, BaseSuperNetModule):
                        warnings.warn("Mutation hook didn't return a BaseSuperNetModule. It will be ignored in hooked module list.",
                                      RuntimeWarning)
                    setattr(m, name, hook_suggest)

                    mutate_result = hook_suggest

                # if suppress, no further mutation hooks are called
                if suppress:
                    break

            if isinstance(mutate_result, BaseSuperNetModule):
                # Replace child with the mutate result, and DFS this one
                child = mutate_result
                module_list.append(mutate_result)

            # pre-order DFS
            if topdown:
                apply(child)

    apply(root_module)

    return module_list


def no_default_hook(module: nn.Module, name: str, memo: dict[str, Any], mutate_kwargs: dict[str, Any]) -> bool:
    """Add this hook at the end of your hook list to raise error for unsupported mutation primitives."""

    # Forward IS NOT supernet
    primitive_list = (
        nas_nn.LayerChoice,
        nas_nn.InputChoice,
        nas_nn.Repeat,
        # nas_nn.NasBench101Cell,
        # nas_nn.ValueChoice,       # could be false positive
        # nas_nn.Cell,              # later
        # nas_nn.NasBench201Cell,   # forward = supernet
    )

    if isinstance(module, primitive_list):
        raise TypeError(f'{type(module).__name__} is not supported')

    if isinstance(module, nas_nn.Cell) and module.merge_op != 'all':
        # need output_node_indices, which depends on super-net
        raise TypeError(f'Cell with merge_op `{module.merge_op}` is not supported')

    if is_traceable(module):
        # check whether there is a value-choice in its arguments
        has_valuechoice = False
        for arg in chain(cast(list, module.trace_args), cast(dict, module.trace_kwargs).values()):
            if isinstance(arg, ValueChoiceX):
                has_valuechoice = True
                break

        if has_valuechoice:
            raise TypeError(f'`basic_unit` {type(module).__name__} with value choice in its arguments is not supported. '
                            'Please try to remove `basic_unit` to see if that works, or support this type with value choice manually.')

    return True  # suppress all other hooks


class BaseOneShotLightningModule(pl.LightningModule):

    _mutation_hooks_note = """mutation_hooks : list[MutationHook]
        Extra mutation hooks to support customized mutation on primitives other than built-ins.

        Mutation hooks are callable that inputs an Module and returns a
        :class:`~nni.retiarii.oneshot.pytorch.supermodule.base.BaseSuperNetModule`.
        They are invoked in :func:`~nni.retiarii.oneshot.pytorch.base_lightning.traverse_and_mutate_submodules`, on each submodules.
        For each submodule, the hook list are invoked subsequently,
        the later hooks can see the result from previous hooks.
        The modules that are processed by ``mutation_hooks`` will be replaced by the returned module,
        stored in :attr:`nas_modules`, and be the focus of the NAS algorithm.

        The hook list will be appended by ``default_mutation_hooks`` in each one-shot module.

        To be more specific, the input arguments are four arguments:

        1. a module that might be processed,
        2. name of the module in its parent module,
        3. a memo dict whose usage depends on the particular algorithm.
        4. keyword arguments (configurations).

        Note that the memo should be read/written by hooks.
        There won't be any hooks called on root module.

        The returned arguments can be also one of the three kinds:

        1. tuple of: :class:`~nni.retiarii.oneshot.pytorch.supermodule.base.BaseSuperNetModule` or None, and boolean,
        2. boolean,
        3. :class:`~nni.retiarii.oneshot.pytorch.supermodule.base.BaseSuperNetModule` or None.

        The boolean value is ``suppress`` indicates whether the following hooks should be called.
        When it's true, it suppresses the subsequent hooks, and they will never be invoked.
        Without boolean value specified, it's assumed to be false.
        If a none value appears on the place of
        :class:`~nni.retiarii.oneshot.pytorch.supermodule.base.BaseSuperNetModule`,
        it means the hook suggests to
        keep the module unchanged, and nothing will happen.

        An example of mutation hook is given in :func:`~nni.retiarii.oneshot.pytorch.base_lightning.no_default_hook`.
        However it's recommended to implement mutation hooks by deriving
        :class:`~nni.retiarii.oneshot.pytorch.supermodule.base.BaseSuperNetModule`,
        and add its classmethod ``mutate`` to this list.
    """

    _inner_module_note = """inner_module : pytorch_lightning.LightningModule
        It's a `LightningModule <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html>`__
        that defines computations, train/val loops, optimizers in a single class.
        When used in NNI, the ``inner_module`` is the combination of instances of evaluator + base model
        (to be precise, a base model wrapped with LightningModule in evaluator).
    """

    __doc__ = """
    The base class for all one-shot NAS modules.

    In NNI, we try to separate the "search" part and "training" part in one-shot NAS.
    The "training" part is defined with evaluator interface (has to be lightning evaluator interface to work with oneshot).
    Since the lightning evaluator has already broken down the training into minimal building blocks,
    we can re-assemble them after combining them with the "search" part of a particular algorithm.

    After the re-assembling, this module has defined all the search + training. The experiment can use a lightning trainer
    (which is another part in the evaluator) to train this module, so as to complete the search process.

    Essential function such as preprocessing user's model, redirecting lightning hooks for user's model,
    configuring optimizers and exporting NAS result are implemented in this class.

    Attributes
    ----------
    nas_modules : list[BaseSuperNetModule]
        Modules that have been mutated, which the search algorithms should care about.
    model : pl.LightningModule
        PyTorch lightning module. A model space with training recipe defined (wrapped by LightningModule in evaluator).

    Parameters
    ----------
    """ + _inner_module_note + _mutation_hooks_note

    trainer: pl.Trainer

    @property
    def automatic_optimization(self) -> bool:
        return False

    def default_mutation_hooks(self) -> list[MutationHook]:
        """Override this to define class-default mutation hooks."""
        return [no_default_hook]

    def mutate_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments passed to mutation hooks. Usually algo-specific."""
        return {}

    def __init__(self, model: pl.LightningModule, mutation_hooks: list[MutationHook] | None = None):
        super().__init__()
        assert isinstance(model, pl.LightningModule)
        self.model = model

        # append the default hooks
        mutation_hooks = (mutation_hooks or []) + self.default_mutation_hooks()

        # traverse the model, calling hooks on every submodule
        self.nas_modules: list[BaseSuperNetModule] = traverse_and_mutate_submodules(
            self.model, mutation_hooks, self.mutate_kwargs(), topdown=True)

    def search_space_spec(self) -> dict[str, ParameterSpec]:
        """Get the search space specification from :attr:`nas_modules`.

        Returns
        -------
        dict
            Key is the name of the choice, value is the corresponding :class:`ParameterSpec`.
        """
        result = {}
        for module in self.nas_modules:
            result.update(module.search_space_spec())
        return result

    def resample(self, memo=None) -> dict[str, Any]:
        """Trigger the resample for each :attr:`nas_modules`.
        Sometimes (e.g., in differentiable cases), it does nothing.

        Parameters
        ----------
        memo : dict[str, Any]
            Used to ensure the consistency of samples with the same label.

        Returns
        -------
        dict
            Sampled architecture.
        """
        result = memo or {}
        for module in self.nas_modules:
            result.update(module.resample(memo=result))
        return result

    def export(self) -> dict[str, Any]:
        """
        Export the NAS result, ideally the best choice of each :attr:`nas_modules`.
        You may implement an ``export`` method for your customized :attr:`nas_modules`.

        Returns
        --------
        dict
            Keys are names of ``nas_modules``, and values are the choice indices of them.
        """
        result = {}
        for module in self.nas_modules:
            result.update(module.export(memo=result))
        return result

    def export_probs(self) -> dict[str, Any]:
        """
        Export the probability of every choice in the search space got chosen.

        .. note:: If such method of some modules is not implemented, they will be simply ignored.

        Returns
        -------
        dict
            In most cases, keys are names of ``nas_modules`` suffixed with ``/`` and choice name.
            Values are the probability / logits depending on the implementation.
        """
        result = {}
        for module in self.nas_modules:
            try:
                result.update(module.export_probs(memo=result))
            except NotImplementedError:
                warnings.warn(
                    'Some super-modules you have used did not implement export_probs. You might find some logs are missing.',
                    UserWarning
                )
        return result

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> Any:
        """
        Transparently configure optimizers for the inner model,
        unless one-shot algorithm has its own optimizer (via :meth:`configure_architecture_optimizers`),
        in which case, the optimizer will be appended to the list.

        The return value is still one of the 6 types defined in PyTorch-Lightning.
        """
        arch_optimizers = self.configure_architecture_optimizers() or []
        if not arch_optimizers:  # no architecture optimizer available
            return self.model.configure_optimizers()

        if isinstance(arch_optimizers, optim.Optimizer):
            arch_optimizers = [arch_optimizers]

        # Set the flag to True so that they can differ from other optimizers
        for optimizer in arch_optimizers:
            optimizer.is_arch_optimizer = True  # type: ignore

        optim_conf: Any = self.model.configure_optimizers()

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

    def setup(self, stage=None):
        # redirect the access to trainer/log to this module
        # but note that we might be missing other attributes,
        # which could potentially be a problem
        self.model.trainer = self.trainer  # type: ignore
        self.model.log = self.log

        # Reset the optimizer progress (only once at the very beginning)
        self._optimizer_progress = 0

        return self.model.setup(stage)

    def teardown(self, stage=None):
        return self.model.teardown(stage)

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

        if self.trainer.optimizer_frequencies:
            warnings.warn('optimizer_frequencies is not supported in NAS. It will be ignored.', UserWarning)

        # Filter out optimizers for architecture parameters
        optimizers = [opt for opt in self.trainer.optimizers if not getattr(opt, 'is_arch_optimizer', False)]

        opt_idx = self._optimizer_progress % len(optimizers)
        optimizer = optimizers[opt_idx]

        # There should be many before/after hooks called here, but they are omitted in this implementation.
        # 1. zero gradient
        self.model.optimizer_zero_grad(self.trainer.current_epoch, batch_idx, optimizer, opt_idx)
        # 2. backward
        self.manual_backward(loss)
        # 3. grad clip
        self.model.configure_gradient_clipping(optimizer, opt_idx, gradient_clip_val, gradient_clip_algorithm)
        # 4. optimizer step
        self.model.optimizer_step(self.trainer.current_epoch, batch_idx, optimizer, opt_idx)

        self._optimizer_progress += 1

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
                scheduler, opt_idx = config.scheduler, config.opt_idx
                if config.reduce_on_plateau:
                    warnings.warn('Reduce-lr-on-plateau is not supported in NAS. It will be ignored.', UserWarning)
                if config.interval == interval and current_idx % config.frequency == 0:
                    self.model.lr_scheduler_step(cast(Any, scheduler), cast(int, opt_idx), None)
        except AttributeError:
            # lightning < 1.6
            for lr_scheduler in self.trainer.lr_schedulers:
                if lr_scheduler['reduce_on_plateau']:
                    warnings.warn('Reduce-lr-on-plateau is not supported in NAS. It will be ignored.', UserWarning)
                if lr_scheduler['interval'] == interval and current_idx % lr_scheduler['frequency']:
                    lr_scheduler['scheduler'].step()

    def architecture_optimizers(self) -> list[Optimizer] | Optimizer | None:
        """
        Get the optimizers configured in :meth:`configure_architecture_optimizers`.
        """
        optimizers = [opt for opt in self.trainer.optimizers if getattr(opt, 'is_arch_optimizer', False)]
        if not optimizers:
            return None
        if len(optimizers) == 1:
            return optimizers[0]
        return optimizers

    # The following methods redirects the callbacks to inner module.
    # It's not the complete list though.
    # More methods can be added if needed.

    def on_train_start(self):
        return self.model.on_train_start()

    def on_train_end(self):
        return self.model.on_train_end()

    def on_fit_start(self):
        return self.model.on_fit_start()

    def on_fit_end(self):
        return self.model.on_fit_end()

    def on_train_batch_start(self, batch, batch_idx, *args, **kwargs):
        return self.model.on_train_batch_start(batch, batch_idx, *args, **kwargs)

    def on_train_batch_end(self, outputs, batch, batch_idx, *args, **kwargs):
        return self.model.on_train_batch_end(outputs, batch, batch_idx, *args, **kwargs)

    def on_train_epoch_start(self):
        return self.model.on_train_epoch_start()

    def on_train_epoch_end(self):
        return self.model.on_train_epoch_end()

    def on_before_backward(self, loss):
        return self.model.on_before_backward(loss)

    def on_after_backward(self):
        return self.model.on_after_backward()
