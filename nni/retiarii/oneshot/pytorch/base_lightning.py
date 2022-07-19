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
        We call :meth:`~BaseSuperNetModule.advance_optimizers` and :meth:`~BaseSuperNetModule.advance_lr_schedulers`
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

    def resample(self) -> dict[str, Any]:
        """Trigger the resample for each :attr:`nas_modules`.
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

    def training_step(self, batch, batch_idx):
        """This is the implementation of what happens in training loops of one-shot algos.
        It usually calls ``self.model.training_step`` which implements the real training recipe of the users' model.
        """
        return self.model.training_step(batch, batch_idx)

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
            optimizer.is_arch_optimizer = True

        optim_conf = self.model.configure_optimizers()

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
            return optim_conf + [{'optimizer': optimizer} for optimizer in arch_optimizers]
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

    def clip_weight_gradients(self, gradient_clip_val: int | float | None = None, gradient_clip_algorithm: str | None = None):
        """Call ``self.clip_gradients()`` for every weight optimizer."""
        optimizers = self.weight_optimizers()
        if isinstance(optimizers, Optimizer):
            self.configure_gradient_clipping(optimizers, 0, gradient_clip_val, gradient_clip_algorithm)
        elif isinstance(optimizers, list):
            for optimizer_idx, optimizer in enumerate(optimizers):
                self.configure_gradient_clipping(optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm)

    def call_weight_optimizers(self, method: Literal['step', 'zero_grad']):
        """
        Function that imitates lightning trainer's behavior of calling user's optimizers.
        Since auto_optimization is turned off in most one-shot algorithm,
        this function is used to make user optimizers behave as they were automatically handled by the lightning trainer.

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
        Get architecture optimizers from all optimizers. Use this to get your architecture optimizers in :meth:`training_step`.

        Returns
        ----------
        opts : list[Optimizer], Optimizer, None
            Architecture optimizers defined in :meth:`configure_architecture_optimizers`. This will be None if there is no
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
        Get user optimizers from all optimizers. Use this to get user optimizers in :meth:`training_step`.

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

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        return self.model.on_train_batch_start(batch, batch_idx, unused)

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        return self.model.on_train_batch_end(outputs, batch, batch_idx, unused)

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
