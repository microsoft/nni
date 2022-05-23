# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import warnings
from itertools import chain
from typing import Callable, Any, Dict, Union, Tuple, List, cast

import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
from torch.optim import Optimizer

from torch.optim.lr_scheduler import _LRScheduler

import nni.retiarii.nn.pytorch as nas_nn
from nni.common.hpo_utils import ParameterSpec
from nni.common.serializer import is_traceable
from nni.retiarii.nn.pytorch.api import ValueChoiceX
from nni.typehint import Literal
from .supermodule.base import BaseSuperNetModule

__all__ = ['MutationHook', 'BaseSuperNetModule', 'BaseOneShotLightningModule', 'traverse_and_mutate_submodules']


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
        nas_nn.NasBench101Cell,
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
        Mutation hooks are callable that inputs an Module and returns a :class:`BaseSuperNetModule`.
        They are invoked in :meth:`traverse_and_mutate_submodules`, on each submodules.
        For each submodule, the hook list are invoked subsequently,
        the later hooks can see the result from previous hooks.
        The modules that are processed by ``mutation_hooks`` will be replaced by the returned module,
        stored in ``nas_modules``, and be the focus of the NAS algorithm.

        The hook list will be appended by ``default_mutation_hooks`` in each one-shot module.

        To be more specific, the input arguments are four arguments:

        #. a module that might be processed,
        #. name of the module in its parent module,
        #. a memo dict whose usage depends on the particular algorithm.
        #. keyword arguments (configurations).

        Note that the memo should be read/written by hooks.
        There won't be any hooks called on root module.
        The returned arguments can be also one of the three kinds:

        #. tuple of: :class:`BaseSuperNetModule` or None, and boolean,
        #. boolean,
        #. :class:`BaseSuperNetModule` or None.

        The boolean value is ``suppress`` indicates whether the folliwng hooks should be called.
        When it's true, it suppresses the subsequent hooks, and they will never be invoked.
        Without boolean value specified, it's assumed to be false.
        If a none value appears on the place of :class:`BaseSuperNetModule`, it means the hook suggests to
        keep the module unchanged, and nothing will happen.
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
        """Get the search space specification from ``nas_module``.

        Returns
        -------
        dict
            Key is the name of the choice, value is the corresponding :class:`ParameterSpec`.
        """
        result = {}
        for module in self.nas_modules:
            result.update(module.search_space_spec())
        return result

    def resample(self) -> dict[str, Any]:
        """Trigger the resample for each ``nas_module``.
        Sometimes (e.g., in differentiable cases), it does nothing.

        Returns
        -------
        dict
            Sampled architecture.
        """
        result = {}
        for module in self.nas_modules:
            result.update(module.resample(memo=result))
        return result

    def export(self) -> dict[str, Any]:
        """
        Export the NAS result, ideally the best choice of each ``nas_module``.
        You may implement an ``export`` method for your customized ``nas_module``.

        Returns
        --------
        dict
            Keys are names of ``nas_modules``, and values are the choice indices of them.
        """
        result = {}
        for module in self.nas_modules:
            result.update(module.export(memo=result))
        return result

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """This is the implementation of what happens in training loops of one-shot algos.
        It usually calls ``self.model.training_step`` which implements the real training recipe of the users' model.
        """
        return self.model.training_step(batch, batch_idx)

    def configure_optimizers(self):
        """
        Combine architecture optimizers and user's model optimizers.
        You can overwrite configure_architecture_optimizers if architecture optimizers are needed in your NAS algorithm.
        For now ``self.model`` is tested against :class:`nni.retiarii.evaluator.pytorch.lightning._SupervisedLearningModule`
        and it only returns 1 optimizer.
        But for extendibility, codes for other return value types are also implemented.
        """
        # pylint: disable=assignment-from-none
        arc_optimizers = self.configure_architecture_optimizers()
        if arc_optimizers is None:
            return self.model.configure_optimizers()

        if isinstance(arc_optimizers, optim.Optimizer):
            arc_optimizers = [arc_optimizers]
        self.arc_optim_count = len(arc_optimizers)

        # FIXME: this part uses non-official lightning API.
        # The return values ``frequency`` and ``monitor`` are ignored because lightning requires
        # ``len(optimizers) == len(frequency)``, and gradient backword is handled manually.
        # For data structure of variables below, please see pytorch lightning docs of ``configure_optimizers``.
        try:
            # above v1.6
            from pytorch_lightning.core.optimizer import (  # pylint: disable=import-error
                _configure_optimizers,  # type: ignore
                _configure_schedulers_automatic_opt,  # type: ignore
                _configure_schedulers_manual_opt  # type: ignore
            )
            w_optimizers, lr_schedulers, self.frequencies, monitor = \
                _configure_optimizers(self.model.configure_optimizers())  # type: ignore
            lr_schedulers = (
                _configure_schedulers_automatic_opt(lr_schedulers, monitor)
                if self.automatic_optimization
                else _configure_schedulers_manual_opt(lr_schedulers)
            )
        except ImportError:
            # under v1.5
            w_optimizers, lr_schedulers, self.frequencies, monitor = \
                self.trainer._configure_optimizers(self.model.configure_optimizers())  # type: ignore
            lr_schedulers = self.trainer._configure_schedulers(lr_schedulers, monitor, not self.automatic_optimization)  # type: ignore

        if any(sch["scheduler"].optimizer not in w_optimizers for sch in lr_schedulers):  # type: ignore
            raise Exception(
                "Some schedulers are attached with an optimizer that wasn't returned from `configure_optimizers`."
            )

        # variables used to handle optimizer frequency
        self.cur_optimizer_step = 0
        self.cur_optimizer_index = 0

        return arc_optimizers + w_optimizers, lr_schedulers

    def on_train_start(self):
        return self.model.on_train_start()

    def on_train_end(self):
        return self.model.on_train_end()

    def on_fit_start(self):
        # redirect the access to trainer/log to this module
        # but note that we might be missing other attributes,
        # which could potentially be a problem
        self.model.trainer = self.trainer  # type: ignore
        self.model.log = self.log
        return self.model.on_fit_start()

    def on_fit_end(self):
        return self.model.on_fit_end()

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        return self.model.on_train_batch_start(batch, batch_idx, unused)

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        return self.model.on_train_batch_end(outputs, batch, batch_idx, unused)

    # Deprecated hooks in pytorch-lightning
    def on_epoch_start(self):
        return self.model.on_epoch_start()

    def on_epoch_end(self):
        return self.model.on_epoch_end()

    def on_train_epoch_start(self):
        return self.model.on_train_epoch_start()

    def on_train_epoch_end(self):
        return self.model.on_train_epoch_end()

    def on_before_backward(self, loss):
        return self.model.on_before_backward(loss)

    def on_after_backward(self):
        return self.model.on_after_backward()

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val=None, gradient_clip_algorithm=None):
        return self.model.configure_gradient_clipping(optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm)

    def configure_architecture_optimizers(self):
        """
        Hook kept for subclasses. A specific NAS method inheriting this base class should return its architecture optimizers here
        if architecture parameters are needed. Note that lr schedulers are not supported now for architecture_optimizers.

        Returns
        ----------
        arc_optimizers : list[Optimizer], Optimizer
            Optimizers used by a specific NAS algorithm. Return None if no architecture optimizers are needed.
        """
        return None

    def call_lr_schedulers(self, batch_index):
        """
        Function that imitates lightning trainer's behaviour of calling user's lr schedulers. Since auto_optimization is turned off
        by this class, you can use this function to make schedulers behave as they were automatically handled by the lightning trainer.

        Parameters
        ----------
        batch_idx : int
            batch index
        """
        def apply(lr_scheduler):
            # single scheduler is called every epoch
            if isinstance(lr_scheduler, _LRScheduler):
                if self.trainer.is_last_batch:
                    lr_scheduler.step()
            # lr_scheduler_config is called as configured
            elif isinstance(lr_scheduler, dict):
                interval = lr_scheduler['interval']
                frequency = lr_scheduler['frequency']
                if (
                        interval == 'step' and
                        batch_index % frequency == 0
                    ) or \
                    (
                        interval == 'epoch' and
                        self.trainer.is_last_batch and
                        (self.trainer.current_epoch + 1) % frequency == 0
                ):
                    lr_scheduler['scheduler'].step()

        lr_schedulers = self.lr_schedulers()

        if isinstance(lr_schedulers, list):
            for lr_scheduler in lr_schedulers:
                apply(lr_scheduler)
        else:
            apply(lr_schedulers)

    def call_weight_optimizers(self, method: Literal['step', 'zero_grad']):
        """
        Function that imitates lightning trainer's behavior of calling user's optimizers. Since auto_optimization is turned off by this
        class, you can use this function to make user optimizers behave as they were automatically handled by the lightning trainer.

        Parameters
        ----------
        method : str
            Method to call. Only ``step`` and ``zero_grad`` are supported now.
        """
        def apply_method(optimizer, method):
            if method == 'step':
                optimizer.step()
            elif method == 'zero_grad':
                optimizer.zero_grad()

        optimizers = self.weight_optimizers()
        if optimizers is None:
            return

        assert isinstance(optimizers, list), 'Did you forget to set use_pl_optimizers to true?'

        if len(self.frequencies) > 0:
            self.cur_optimizer_step += 1
            if self.frequencies[self.cur_optimizer_index] == self.cur_optimizer_step:
                self.cur_optimizer_step = 0
                self.cur_optimizer_index = self.cur_optimizer_index + 1 \
                    if self.cur_optimizer_index + 1 < len(optimizers) \
                    else 0
            apply_method(optimizers[self.cur_optimizer_index], method)
        else:
            for optimizer in optimizers:
                apply_method(optimizer, method)

    def architecture_optimizers(self) -> list[Optimizer] | Optimizer | None:
        """
        Get architecture optimizers from all optimizers. Use this to get your architecture optimizers in ``training_step``.

        Returns
        ----------
        opts : list[Optimizer], Optimizer, None
            Architecture optimizers defined in ``configure_architecture_optimizers``. This will be None if there is no
            architecture optimizers.
        """
        opts = self.optimizers()
        if isinstance(opts, list):
            # pylint: disable=unsubscriptable-object
            arc_opts = opts[:self.arc_optim_count]
            if len(arc_opts) == 1:
                return cast(Optimizer, arc_opts[0])
            return cast(List[Optimizer], arc_opts)
        # If there is only 1 optimizer and it is the architecture optimizer
        if self.arc_optim_count == 1:
            return cast(Union[List[Optimizer], Optimizer], opts)
        return None

    def weight_optimizers(self) -> list[Optimizer] | Optimizer | None:
        """
        Get user optimizers from all optimizers. Use this to get user optimizers in ``training_step``.

        Returns
        ----------
        opts : list[Optimizer], Optimizer, None
            Optimizers defined by user's model. This will be None if there is no user optimizers.
        """
        # Since use_pl_optimizer is set true (by default) here.
        # opts always return a list
        opts = self.optimizers()
        if isinstance(opts, list):
            # pylint: disable=unsubscriptable-object
            return cast(List[Optimizer], opts[self.arc_optim_count:])
        # FIXME: this case is actually not correctly handled
        # If there is only 1 optimizer and no architecture optimizer
        if self.arc_optim_count == 0:
            return cast(Union[List[Optimizer], Optimizer], opts)
        return None
