# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Strategy integration of one-shot.

This file is put here simply because it relies on "pytorch".
For consistency, please consider importing strategies from ``nni.nas.strategy``.
For example, ``nni.nas.strategy.DartsStrategy`` (this requires pytorch to be installed of course).

When adding/modifying a new strategy in this file, don't forget to link it in strategy/oneshot.py.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable, Dict, Union, Tuple, TypeVar, Iterator, TYPE_CHECKING, cast

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from nni.mutable import Mutable, Categorical, CategoricalMultiple, Sample, frozen_context
from nni.nas.evaluator.pytorch.lightning import Lightning, LightningModule
from nni.nas.execution import SequentialExecutionEngine
from nni.nas.nn.pytorch import ModelSpace, MutableModule
from nni.nas.space import RawFormatModelSpace, ModelStatus
from nni.nas.strategy.base import Strategy

from .base_lightning import BaseOneShotLightningModule
from .differentiable import DartsLightningModule, LinearTemperatureScheduler, GumbelDartsLightningModule
from .profiler import ExpectationProfilerPenalty, SampleProfilerPenalty, ProfilerFilter, RangeProfilerFilter
from .sampling import EnasLightningModule, RandomSamplingLightningModule
from .supermodule.base import BaseSuperNetModule
from .supermodule.operation import NATIVE_SUPPORTED_OP_NAMES, NATIVE_MIXED_OPERATIONS

if TYPE_CHECKING:
    from nni.nas.strategy._rl_impl import PolicyFactory

_logger = logging.getLogger(__name__)

MutationHookReturnType = Union[nn.Module, bool, Tuple[nn.Module, bool]]
MutationHook = Callable[[nn.Module, str, Dict[str, Any], Dict[str, Any]], MutationHookReturnType]

ModuleType = TypeVar('ModuleType', bound=nn.Module)
ModelSpaceType = TypeVar('ModelSpaceType', bound=ModelSpace)


def _submodule_tree_map(name: str, module: ModuleType, map_fn: Callable[[str, nn.Module], nn.Module | None],
                        topdown: bool = True) -> ModuleType:
    """Transform every submodule with ``map_fn``.

    ``map_fn`` is expected to return a new module, or ``None`` to indicate that the module should not be changed.

    The root module is not transformed.
    """

    for subname, submodule in list(module.named_children()):  # list here because there're changes in-place
        subname_ = f'{name}.{subname}' if name else subname

        if not topdown:  # post-order DFS
            _submodule_tree_map(subname_, submodule, map_fn, topdown)

        new_submodule = map_fn(subname_, submodule)
        if new_submodule is not None:
            setattr(module, subname, new_submodule)
            submodule = new_submodule  # For DFS

        if topdown:  # pre-order DFS
            _submodule_tree_map(subname_, submodule, map_fn, topdown)

    return module


def no_default_hook(module: nn.Module, name: str, memo: dict[str, Any], mutate_kwargs: dict[str, Any]) -> bool:
    """Add this hook at the end of your hook list to raise error for unsupported mutation primitives.

    If error is not raised, it's possible that users assume it works but the model is actually wrong.
    """

    from nni.nas.nn.pytorch import Cell
    if isinstance(module, Cell) and module.merge_op != 'all':
        # need output_node_indices, which depends on super-net
        raise TypeError(f'Cell with merge_op `{module.merge_op}` is not supported')

    if isinstance(module, MutableModule) and module.mutables:
        raise TypeError(f'Module `{name}` of type `{type(module).__name__}` has dangling mutables and is not supported. '
                        'Please implement its one-shot version and register it into `mutation_hooks`.')

    return True  # suppress all other hooks


def is_supernet(module: nn.Module) -> bool:
    """Utility function to check whether the module (or its nested modules) have been mutated by the supernet."""
    return isinstance(module, nn.Module) and any(isinstance(module, BaseSuperNetModule) for module in module.modules())


class OneShotStrategy(Strategy):
    """Wrap an one-shot lightning module as a one-shot strategy.

    A one-shot strategy has the following workflow:

    1. Mutate the model to a supernet. (The current implementation will do this inplace.)
    2. Mutate the evaluator (must be written in Lightning).
       Core steps include: injecting the search logics into lightning module and process the dataloaders.
    3. Submit the model and evaluator for training.

    Notes
    -----
    In NNI, we try to separate the "search" part and "training" part in one-shot NAS.
    The "training" part is defined with evaluator interface (has to be lightning evaluator interface to work with oneshot).
    Since the lightning evaluator has already broken down the training into minimal building blocks,
    we can re-assemble them after combining them with the "search" part of a particular algorithm.

    After the re-assembling, this module has defined all the search + training. The experiment can use a lightning trainer
    (which is another part in the evaluator) to train this module, so as to complete the search process.

    Parameters
    ----------
    mutation_hooks
        Extra mutation hooks to support customized mutation on primitives other than built-ins.

        Mutation hooks are callable that inputs an Module and returns a
        :class:`~nni.nas.oneshot.pytorch.supermodule.base.BaseSuperNetModule`.
        They are invoked in :func:`~nni.nas.oneshot.pytorch.base_lightning.traverse_and_mutate_submodules`, on each submodules.
        For each submodule, the hook list are invoked subsequently,
        the later hooks can see the result from previous hooks.
        The modules that are processed by ``mutation_hooks`` will be replaced by the returned module,
        stored in :attr:`nas_modules`, and be the focus of the NAS algorithm.

        The hook list will be appended by ``default_mutation_hooks`` in each one-shot module.

        To be more specific, the input arguments of a hook are four arguments:

        1. a module that might be processed,
        2. name of the module in its parent module,
        3. a memo dict whose usage depends on the particular algorithm.
        4. keyword arguments (configurations).

        Note that the memo should be read/written by hooks.
        There won't be any hooks called on root module.

        The returned arguments can be also one of the three kinds:

        1. tuple of: :class:`~nni.nas.oneshot.pytorch.supermodule.base.BaseSuperNetModule` or None, and boolean,
        2. boolean,
        3. :class:`~nni.nas.oneshot.pytorch.supermodule.base.BaseSuperNetModule` or None.

        The boolean value is ``suppress`` indicates whether the following hooks should be called.
        When it's true, it suppresses the subsequent hooks, and they will never be invoked.
        Without boolean value specified, it's assumed to be false.
        If a none value appears on the place of
        :class:`~nni.nas.oneshot.pytorch.supermodule.base.BaseSuperNetModule`,
        it means the hook suggests to
        keep the module unchanged, and nothing will happen.

        An example of mutation hook is given in :func:`~nni.nas.oneshot.pytorch.base_lightning.no_default_hook`.
        However it's recommended to implement mutation hooks by deriving
        :class:`~nni.nas.oneshot.pytorch.supermodule.base.BaseSuperNetModule`,
        and add its classmethod ``mutate`` to this list.

    **kwargs
        Extra keyword arguments passed to :class:`~nni.nas.strategy.Strategy`.
    """

    def __init__(self, mutation_hooks: list[MutationHook] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.extra_mutation_hooks = mutation_hooks or []

        self._mutated_model_space: RawFormatModelSpace | None = None
        self._mutation_hooks_memo: dict[str, Any] = {}

    def configure_oneshot_module(self, training_module: LightningModule) -> BaseOneShotLightningModule:
        """Create the oneshot module, i.e., the "search" part of the algorithm.

        Subclass should override this.
        """
        raise NotImplementedError('configure_oneshot_module is not implemented.')

    def default_mutation_hooks(self) -> list[MutationHook]:
        """Override this to define class-default mutation hooks."""
        return [no_default_hook]

    def run_hook(self, hook: MutationHook, name: str, module: nn.Module, memo: dict[str, Any]) -> MutationHookReturnType:
        """Run a single mutation hook.

        For internal use only: subclass can override this to intercept the hooks for customization.
        For example, provide extra keyword arguments or tamper the memo.
        """
        kwargs = {}  # Subclass can rewrite this for extra kwargs.
        return hook(module, name, memo, kwargs)

    def train_dataloader(self, train_dataloader_fn: Callable[[], Any], val_dataloader_fn: Callable[[], Any]) -> Any:
        """
        One-shot strategy typically requires fusing train and validation dataloader in an ad-hoc way.
        As one-shot strategy doesn't try to open the blackbox of a batch,
        theoretically, these dataloader can be any dataloader types supported by Lightning.

        Parameters
        ----------
        train_dataloader_fn
            Factory that takes no argument, returning a train dataloader.
        val_dataloader_fn
            Similar to ``train_dataloader_fn``.

        Returns
        -------
        Preprocessed train dataloaders.
        """
        return train_dataloader_fn()

    def val_dataloader(self, train_dataloader_fn: Callable[[], Any], val_dataloader_fn: Callable[[], Any]) -> Any:
        """
        See :meth:`train_dataloader`.

        Returns
        -------
        Preprocessed **validation** dataloaders.
        """
        return val_dataloader_fn()

    def mutate_model(self, model: ModelSpaceType) -> ModelSpaceType:
        """Convert the model space to a supernet **inplace**.

        The core of a one-shot strategy is usually a carefully-designed supernet,
        which encodes the sharing pattern and mechanism.
        :meth:`create_supernet` transforms a model space into a one-shot supernet.

        Mostly useful for debugging and supernet inspection.

        Parameters
        ----------
        model
            The model space to be transformed. The raw model space written in PyTorch.

        Returns
        -------
        The one-shot supernet.
        Note that the changes will take inplace.
        Therefore the returned model is the same as the input ``model``.

        The mutated model is still a :class:`ModelSpace` instance.
        In most cases, ``simplify()`` and ``freeze(sample)`` would still return the same result,
        which is convenient for follow-up search on the supernet.
        """
        if not isinstance(model, ModelSpace):
            raise TypeError('The transformed model must be a ModelSpace.')

        model_defined_hooks = []
        if hasattr(model, 'extra_oneshot_hooks'):
            model_defined_hooks: list[MutationHook] = model.extra_oneshot_hooks(self)  # type: ignore

        # Find all hooks. User-defined ones are upfront.
        hooks = self.extra_mutation_hooks + model_defined_hooks + self.default_mutation_hooks()

        self._mutation_hooks_memo = {}

        # NOTE:
        # Some mutables (e.g., LayerChoice) require a frozen context when creating.
        # So we mock a frozen context here.
        with frozen_context():
            # traverse the model, calling hooks on every submodule
            _submodule_tree_map('', model, partial(self._execute_mutation_hooks, hooks=hooks))

        # Clear it.
        self._mutation_hooks_memo = {}

        return model

    def mutate_evaluator(self, evaluator: Lightning) -> Lightning:
        """Mutate the evaluator to the one used in one-shot.

        Specifically, it:

        - uses :attr:`oneshot_module` to wrap the ``module`` in evaluator.
        - calls :meth:`preprocess_dataloader` to refuse the dataloaders.

        Returns
        -------
        The mutated evaluator.
        """
        if not isinstance(evaluator, Lightning):
            raise TypeError('Evaluator needs to be a lightning evaluator to make one-shot strategy work.')
        if isinstance(evaluator, Mutable) and evaluator.simplify():
            raise ValueError('Evaluator cannot contain any mutable parameters for one-shot strategy.')

        evaluator_module: LightningModule = evaluator.module
        oneshot_module = self.configure_oneshot_module(evaluator_module)

        if evaluator.datamodule is not None:
            # Monkey-patch the datamodule.
            datamodule = evaluator.datamodule
            train_dataloader_fn = evaluator.datamodule.train_dataloader
            val_dataloader_fn = evaluator.datamodule.val_dataloader
            datamodule.train_dataloader = partial(self.train_dataloader, train_dataloader_fn, val_dataloader_fn)
            datamodule.val_dataloader = partial(self.val_dataloader, train_dataloader_fn, val_dataloader_fn)
            data_kwargs = {'datamodule': datamodule}
        else:
            if evaluator.train_dataloaders is None or evaluator.val_dataloaders is None:
                raise ValueError('Training and validation dataloader are both required to set in evaluator for one-shot strategy.')
            train_loader = self.train_dataloader(lambda: evaluator.train_dataloaders, lambda: evaluator.val_dataloaders)
            val_loader = self.val_dataloader(lambda: evaluator.train_dataloaders, lambda: evaluator.val_dataloaders)
            data_kwargs = {'train_dataloaders': train_loader, 'val_dataloaders': val_loader}

        return Lightning(
            oneshot_module,
            evaluator.trainer,
            fit_kwargs=evaluator.fit_kwargs,
            **data_kwargs,
        )

    def _initialize(self, model_space: RawFormatModelSpace, engine: SequentialExecutionEngine) -> RawFormatModelSpace:
        """One-shot strategy only mutates the model space once and generate one model."""
        if not isinstance(model_space, RawFormatModelSpace):
            raise TypeError(f'One-shot strategy only supports RawFormatModelSpace.')

        if not isinstance(engine, SequentialExecutionEngine):
            raise TypeError(f'One-shot strategy only supports SequentialExecutionEngine.')

        # Type validation inside.
        model = self.mutate_model(cast(ModelSpace, model_space.model_space))
        evaluator = self.mutate_evaluator(cast(Lightning, model_space.evaluator))

        self._mutated_model_space = RawFormatModelSpace(model, evaluator)
        self._mutated_model_space.status = ModelStatus.Frozen

        _logger.debug('%s strategy is initialized.', self.__class__.__name__)

        return self._mutated_model_space

    def _run(self) -> None:
        """Submit the only model generated in :meth:`_initialize` for training."""
        if self._mutated_model_space is None:
            raise RuntimeError('One-shot strategy is not initialized yet.')
        self.engine.submit_models(self._mutated_model_space)

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state dict of one-shot strategy."""
        if self._mutated_model_space is None:
            raise RuntimeError('One-shot strategy is not initialized yet.')
        if 'ckpt_path' in state_dict:
            evaluator = cast(Lightning, self._mutated_model_space.evaluator)
            if evaluator.fit_kwargs.get('ckpt_path') is not None:
                _logger.warning('ckpt_path is already set in evaluator. The ckpt_path in state_dict of strategy will be ignored.')
            else:
                _logger.debug('Loading ckpt_path from state_dict of strategy: %s', state_dict['ckpt_path'])
                evaluator.fit_kwargs['ckpt_path'] = state_dict['ckpt_path']

    def state_dict(self) -> dict:
        """Get the state dict of one-shot strategy.

        The state dict of one-shot strategy leverages the checkpoint callback in Lightning evaluator.
        It will look for ``last_model_path`` attribute (or ``best_model_path``) in ``trainer.checkpoint_callback``,
        save it, and put it back into ``fit_kwargs`` when :meth:`load_state_dict` is called.
        """
        if self._mutated_model_space is None:
            return {}
        evaluator = cast(Lightning, self._mutated_model_space.evaluator)
        checkpoint_callback = evaluator.trainer.checkpoint_callback
        if checkpoint_callback is not None:
            if getattr(checkpoint_callback, 'last_model_path', None):
                return {'ckpt_path': checkpoint_callback.last_model_path}  # type: ignore
            elif getattr(checkpoint_callback, 'best_model_path', None):
                _logger.debug('Checkpoint callback does not have last_model_path attribute, using best_model_path.')
                return {'ckpt_path': checkpoint_callback.best_model_path}  # type: ignore
            else:
                _logger.warning('Checkpoint callback does not have last_model_path or best_model_path attribute. '
                                'Either the strategy has not started, or it did not save any checkpoint: %s',
                                checkpoint_callback)
        else:
            _logger.warning('Checkpoint callback is found to be None in evaluator trainer.')
        # Return empty state dict
        return {}

    def list_models(self, sort: bool = True, limit: int | None = 1) -> Iterator[RawFormatModelSpace]:
        """Getting the best models searched by the one-shot strategy.

        The behavior of which models will be chosen depends on the implementation of inner one-shot module.

        Parameters
        ----------
        sort
            Must be true.
        limit
            The number of models to be returned. Only supports 1 for now.
        """
        if not sort:
            _logger.warning('One-shot strategy currently only supports returning the best models, '
                            'got sort = %s. Will be reset to true.', sort)
            sort = True
        if limit != 1:
            _logger.warning('One-shot strategy currently only supports exporting top-1 model, got %d. It will be reset to 1.', limit)
            limit = 1

        self.oneshot_module.set_model(self.supernet)
        sample = self.oneshot_module.export()
        yield self.model_space.freeze(sample)

    @property
    def supernet(self) -> ModelSpace:
        """The supernet created by one-shot strategy.

        Only available after :meth:`run` is called.
        """
        if self._mutated_model_space is None:
            raise RuntimeError('One-shot strategy needs to be run before accessing the supernet.')
        return cast(ModelSpace, self._mutated_model_space.model_space)

    @property
    def oneshot_module(self) -> BaseOneShotLightningModule:
        """The one-shot module created by one-shot strategy.

        Only available after :meth:`run` is called.
        """
        if self._mutated_model_space is None:
            raise RuntimeError('One-shot strategy needs to be run before accessing the one-shot module.')
        evaluator = cast(Lightning, self._mutated_model_space.evaluator)
        return cast(BaseOneShotLightningModule, evaluator.module)

    def _execute_mutation_hooks(self, name: str, module: nn.Module, hooks: list[MutationHook]) -> nn.Module | None:
        """Execute the mutation hooks on the module.

        See the note for ``mutation_hooks`` in :class:`OneShotStrategy` for more details.
        """
        is_replaced: bool = False

        for hook in hooks:
            hook_suggest = self.run_hook(hook, name, module, self._mutation_hooks_memo)

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
                    _logger.warning("Mutation hook on %s didn't return a BaseSuperNetModule. "
                                    "The replacement will still be effective but it will be probably ignored by the algorithm.",
                                    name)

                module = hook_suggest
                is_replaced = True

            # if suppress, no further mutation hooks are called
            if suppress:
                break

        _logger.debug('Mutation hook on %s returns type %s.', name, type(module).__name__)

        return module if is_replaced else None


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


class DARTS(OneShotStrategy):
    __doc__ = """
    Continuous relaxation of the architecture representation, allowing efficient search of the architecture using gradient descent.
    `Reference <https://arxiv.org/abs/1806.09055>`__.

    DARTS algorithm is one of the most fundamental one-shot algorithm.
    DARTS repeats iterations, where each iteration consists of 2 training phases.
    The phase 1 is architecture step, in which model parameters are frozen and the architecture parameters are trained.
    The phase 2 is model step, in which architecture parameters are frozen and model parameters are trained.
    In both phases, ``training_step`` of the Lightning evaluator will be used.

    The current implementation corresponds to DARTS (1st order) in paper.
    Second order (unrolled 2nd-order derivatives) is not supported yet.

    .. note::

       DARTS is running a weighted sum of possible architectures under the hood.
       Please bear in mind that it will be slower and consume more memory that training a single architecture.
       The common practice is to down-scale the network (e.g., smaller depth / width) for speedup.

    .. versionadded:: 2.8

       Supports searching for ValueChoices on operations, with the technique described in
       `FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions <https://arxiv.org/abs/2004.05565>`__.
       One difference is that, in DARTS, we are using Softmax instead of GumbelSoftmax.

    The supported mutation primitives of DARTS are:

    * :class:`nni.nas.nn.pytorch.LayerChoice`.
    * :class:`nni.nas.nn.pytorch.InputChoice`.
    * :class:`nni.nas.nn.pytorch.ParametrizedModule` (only when parameters are choices and type is in {supported_ops}).
    * :class:`nni.nas.nn.pytorch.Repeat`.
    * :class:`nni.nas.nn.pytorch.Cell`.

    {optimization_note}

    Parameters
    ----------
    arc_learning_rate
        Learning rate for architecture optimizer.
    gradient_clip_val 
        Clip gradients before optimizing models at each step.
        Disable gradient clipping by setting it to ``None``.
    log_prob_every_n_step
        Log current architecture parameters every ``log_prob_every_n_step`` steps.
    warmup_epochs
        The first ``warmup_epochs`` do not update architecture weights.
    penalty
        If a dict, it should contain the keys: ``profiler``, ``baseline``, and optionally ``scale``, ``nonlinear``, ``aggregate``.
        We will create a :class:`~nni.nas.oneshot.pytorch.profiler.ExpectationProfilerPenalty` with the given parameters.
    **kwargs
        Other parameters for :class:`~nni.nas.oneshot.pytorch.strategy.BaseOneShotStrategy`.
    """.format(
        supported_ops=', '.join(NATIVE_SUPPORTED_OP_NAMES),
        optimization_note=MANUAL_OPTIMIZATION_NOTE
    )

    def __init__(self, *, arc_learning_rate: float = 3.0e-4, gradient_clip_val: float | None = None,
                 log_prob_every_n_step: int = 10, warmup_epochs: int = 0,
                 penalty: dict | ExpectationProfilerPenalty | None = None, **kwargs):
        super().__init__(**kwargs)

        self.arc_learning_rate = arc_learning_rate
        self.gradient_clip_val = gradient_clip_val
        self.log_prob_every_n_step = log_prob_every_n_step
        self.warmup_epochs = warmup_epochs
        if isinstance(penalty, dict):
            self.penalty = ExpectationProfilerPenalty(**penalty)
        else:
            self.penalty = penalty

        self._arch_parameters: dict[str, nn.Parameter] = {}

    def configure_oneshot_module(self, training_module: LightningModule) -> BaseOneShotLightningModule:
        return DartsLightningModule(
            training_module=training_module,
            arc_learning_rate=self.arc_learning_rate,
            gradient_clip_val=self.gradient_clip_val,
            log_prob_every_n_step=self.log_prob_every_n_step,
            warmup_epochs=self.warmup_epochs,
            penalty=self.penalty
        )

    def configure_softmax(self) -> nn.Module:
        return nn.Softmax(dim=-1)

    def default_mutation_hooks(self) -> list[MutationHook]:
        """Replace modules with differentiable versions."""
        from .supermodule.differentiable import (
            DifferentiableMixedLayer, DifferentiableMixedInput,
            DifferentiableMixedCell, DifferentiableMixedRepeat
        )

        hooks = [
            DifferentiableMixedLayer.mutate,
            DifferentiableMixedInput.mutate,
            DifferentiableMixedCell.mutate,
            DifferentiableMixedRepeat.mutate,
        ]
        hooks += [operation.mutate for operation in NATIVE_MIXED_OPERATIONS]
        hooks.append(no_default_hook)
        return hooks

    def mutate_model(self, model: ModelSpaceType) -> ModelSpaceType:
        # Create architecture parameters beforehand here, in order to save the trouble of creating them inside.
        # It should only be done once because everything else.
        # But sometimes we need to create them inside, e.g., in the cell an extra connection is needed.
        # In these circumstances, they can still go ahead and ignoring the memo.
        self._arch_parameters = {}
        for label, mutable in model.simplify().items():
            if not isinstance(mutable, (Categorical, CategoricalMultiple)):
                raise TypeError(f'Differentiable strategies only support categorical variables, but got {type(mutable)}')
            alpha = nn.Parameter(torch.randn(len(mutable.values)) * 1E-3)
            self._arch_parameters[label] = alpha

        return super().mutate_model(model)

    def run_hook(self, hook: MutationHook, name: str, module: nn.Module, memo: dict[str, Any]) -> MutationHookReturnType:
        if not memo.get('_validated'):
            memo.update(self._arch_parameters)
            memo['_validated'] = True

        # Use differentiable strategy for mixed operations.
        from .supermodule.differentiable import MixedOpDifferentiablePolicy
        kwargs = {'mixed_op_sampling': MixedOpDifferentiablePolicy, 'softmax': self.configure_softmax()}
        return hook(module, name, memo, kwargs)

    def train_dataloader(self, train_dataloader_fn, val_dataloader_fn):
        # By returning a dict, we make a CombinedLoader (in Lightning)
        return {
            'train': train_dataloader_fn(),
            'val': val_dataloader_fn()
        }

    def val_dataloader(self, train_dataloader_fn, val_dataloader_fn):
        return None


class Proxyless(DARTS):
    __doc__ = """
    A low-memory-consuming optimized version of differentiable architecture search. See `reference <https://arxiv.org/abs/1812.00332>`__.

    This is a :class:`~nni.nas.strategy.DARTS`-based method that resamples the architecture to reduce memory consumption.
    Essentially, it samples one path on forward,
    and implements its own backward to update the architecture parameters based on only one path.

    The supported mutation primitives of :class:`Proxyless` are:

    * :class:`nni.nas.nn.pytorch.LayerChoice` (candidate layers must NOT have keyword arguments).
    * :class:`nni.nas.nn.pytorch.InputChoice`.
    * :class:`nni.nas.nn.pytorch.Repeat` (with categorical choice of no transformation).

    {optimization_note}

    Parameters
    ----------
    **kwargs
        Supported parameters are the same as :class:`~nni.nas.strategy.DARTS`.
    """.format(
        optimization_note=MANUAL_OPTIMIZATION_NOTE
    )

    def default_mutation_hooks(self) -> list[MutationHook]:
        """Replace modules with proxyless-differentiable versions."""
        from .supermodule.proxyless import ProxylessMixedLayer, ProxylessMixedInput, ProxylessMixedRepeat, suppress_already_mutated

        hooks = [
            suppress_already_mutated,
            ProxylessMixedLayer.mutate,
            ProxylessMixedInput.mutate,
            ProxylessMixedRepeat.mutate,
            no_default_hook,
        ]
        # FIXME: no support for mixed operation currently
        return hooks


class GumbelDARTS(DARTS):
    __doc__ = """
    Choose the best block by using Gumbel Softmax random sampling and differentiable training.
    See `FBNet <https://arxiv.org/abs/1812.03443>`__ and `SNAS <https://arxiv.org/abs/1812.09926>`__.

    This is a :class:`~nni.nas.strategy.DARTS`-based method that uses gumbel-softmax to simulate one-hot distribution.
    Essentially, it tries to mimick the behavior of sampling one path on forward by gradually
    cool down the temperature, aiming to bridge the gap between differentiable architecture weights and
    discretization of architectures.

    .. versionadded:: 2.8
    
       Supports searching for ValueChoices on operations, with the technique described in
       `FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions <https://arxiv.org/abs/2004.05565>`__.

    The supported mutation primitives of GumbelDARTS are:

    * :class:`nni.nas.nn.pytorch.LayerChoice`.
    * :class:`nni.nas.nn.pytorch.InputChoice`.
    * :class:`nni.nas.nn.pytorch.ParametrizedModule` (only when parameters are choices and type is in {supported_ops}).
    * :class:`nni.nas.nn.pytorch.Repeat`.
    * :class:`nni.nas.nn.pytorch.Cell`.

    .. note::

       GumbelDARTS is running a weighted sum of possible architectures under the hood.
       Please bear in mind that it will be slower and consume more memory that training a single architecture.
       The common practice is to down-scale the network (e.g., smaller depth / width) for speedup.

    {optimization_note}

    Parameters
    ----------
    temperature
        The temperature used in gumbel-softmax. It can be:

        * A float, which will be used as the fixed temperature throughout the training.
        * A tuple of two floats, which will be used as the initial and final temperature for annealing.
        * A dict with keys ``init`` and ``min``, which will be used as the initial and final temperature for annealing.
        * A :class:`~nni.nas.oneshot.pytorch.differentiable.LinearTemperatureScheduler` instance.
    **kwargs
        Other supported parameters can be found in :class:`~nni.nas.strategy.DARTS`.
    """.format(
        supported_ops=', '.join(NATIVE_SUPPORTED_OP_NAMES),
        optimization_note=MANUAL_OPTIMIZATION_NOTE,
    )

    def configure_softmax(self) -> nn.Module:
        from .supermodule.differentiable import GumbelSoftmax
        return GumbelSoftmax()

    def __init__(self, *, temperature: dict | tuple[float, float] | LinearTemperatureScheduler | float = (1.0, 0.33), **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def configure_oneshot_module(self, training_module: LightningModule) -> BaseOneShotLightningModule:
        if isinstance(self.temperature, float):
            temperature = LinearTemperatureScheduler(self.temperature, self.temperature)
        elif isinstance(self.temperature, tuple) and len(self.temperature) == 2:
            temperature = LinearTemperatureScheduler(*self.temperature)
        elif isinstance(self.temperature, dict) and 'init' in self.temperature and 'min' in self.temperature:
            temperature = LinearTemperatureScheduler(self.temperature['init'], self.temperature['min'])
        elif isinstance(self.temperature, LinearTemperatureScheduler):
            temperature = self.temperature
        else:
            raise ValueError(f'Invalid temperature: {self.temperature}')

        return GumbelDartsLightningModule(
            training_module=training_module,
            temperature_scheduler=temperature,
            arc_learning_rate=self.arc_learning_rate,
            gradient_clip_val=self.gradient_clip_val,
            log_prob_every_n_step=self.log_prob_every_n_step,
            warmup_epochs=self.warmup_epochs,
            penalty=self.penalty
        )


class RandomOneShot(OneShotStrategy):
    __doc__ = """
    Train a super-net with uniform path sampling. See `reference <https://arxiv.org/abs/1904.00420>`__.

    In each step, model parameters are trained after a uniformly random sampling of each choice.
    Notably, the exporting result is **also a random sample** of the search space.

    The supported mutation primitives of RandomOneShot are:

    * :class:`nni.nas.nn.pytorch.LayerChoice`.
    * :class:`nni.nas.nn.pytorch.InputChoice`.
    * :class:`nni.nas.nn.pytorch.ParametrizedModule` (only when parameters' type is in {supported_ops}).
    * :class:`nni.nas.nn.pytorch.Repeat`.
    * :class:`nni.nas.nn.pytorch.Cell`.

    This strategy assumes inner evaluator has set
    `automatic optimization <https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html>`__ to true.

    Parameters
    ----------
    filter
        A function that takes a sample and returns a boolean.
        We recommend using :class:`~nni.nas.oneshot.pytorch.profiler.ProfilerFilter` to filter samples.
        If it's a dict of keys of ``profiler``, and either (or both) of ``min`` and ``max``,
        it will be used to construct a :class:`~nni.nas.oneshot.pytorch.profiler.RangeProfilerFilter`.
    **kwargs
        Parameters for :class:`~nni.nas.oneshot.pytorch.strategy.BaseOneShotStrategy`.

    Examples
    --------
    This strategy is mostly used as a "pre"-strategy to speedup another multi-trial strategy.
    The multi-trial strategy can leverage the trained weights from :class:`RandomOneShot`
    such that each sampled model won't need to be trained from scratch.
    See `SPOS <https://arxiv.org/abs/1904.00420>`__,
    `OFA <https://arxiv.org/abs/1908.09791>`__ and
    `AutoFormer <https://arxiv.org/abs/2106.13008>`__ for how this is done in the arts.

    A typical workflow looks like follows::

        model_space = MyModelSpace()
        evaluator = Classification(max_epochs=100)  # usually trained longer
        strategy = RandomOneShot()
        NasExperiment(model_space, evaluator, strategy).run()  # pretrain the supernet

        # Now the model space is mutated and trained inplace
        evaluator = Classification(max_epochs=0)  # no training
        strategy = RegularizedEvolution()
        NasExperiment(model_space, evaluator, strategy).run()  # search a subnet

    .. warning::

        The second experiment must use ``keep`` model format and ``sequential`` execution engine
        (which is by default inferred in this setup).
        Otherwise, the weights will be lost during serialization.

    For debugging purposes, it's also possible to save and restore the pretrained supernet::

        # After run RandomOneShot strategy
        torch.save(model_space.state_dict(), '/path/to/somewhere')

        # Then load the pretrained supernet in a separate run
        model_space = MyModelSpace()
        pre_strategy = RandomOneShot()
        pre_strategy.mutate_model(model_space)
        model_space.load_state_dict(torch.load('/path/to/somewhere'))

    You can also manually use all the methods from :class:`~nni.nas.nn.pytorch.ModelSpace` for the supernet.
    Notably, the :meth:`~nni.nas.nn.pytorch.ModelSpace.freeze` method will be weight-preserving, i.e.,
    the weights of the subnet will inherit those on the supernet::

        model_space.freeze({{'layer1': 0, 'layer2': 1}})
    """.format(
        supported_ops=', '.join(NATIVE_SUPPORTED_OP_NAMES)
    )

    def __init__(self, filter: ProfilerFilter | dict | Callable[[Sample], bool] | None = None, **kwargs) -> None:  # pylint: disable=redefined-builtin
        super().__init__(**kwargs)
        if isinstance(filter, dict):
            self.filter = RangeProfilerFilter(**filter)
        else:
            self.filter = filter

    def default_mutation_hooks(self) -> list[MutationHook]:
        from .supermodule.sampling import (
            PathSamplingLayer, PathSamplingInput,
            PathSamplingRepeat, PathSamplingCell
        )

        hooks = [
            PathSamplingLayer.mutate,
            PathSamplingInput.mutate,
            PathSamplingRepeat.mutate,
            PathSamplingCell.mutate,
        ]
        hooks += [operation.mutate for operation in NATIVE_MIXED_OPERATIONS]
        hooks.append(no_default_hook)
        return hooks

    def configure_oneshot_module(self, training_module: LightningModule) -> RandomSamplingLightningModule:
        return RandomSamplingLightningModule(training_module, self.filter)

    def run_hook(self, hook: MutationHook, name: str, module: nn.Module, memo: dict[str, Any]) -> MutationHookReturnType:
        # Use path sampling strategy for mixed-operations.
        from .supermodule.sampling import MixedOpPathSamplingPolicy
        kwargs = {
            'mixed_op_sampling': MixedOpPathSamplingPolicy
        }
        return hook(module, name, memo, kwargs)


class ENAS(RandomOneShot):
    __doc__ = """
    RL controller learns to generate the best network on a super-net. See `ENAS paper <https://arxiv.org/abs/1802.03268>`__.

    In every epoch, training dataset and validation dataset are given sequentially in batches.
    For the training dataset, the agent sample subnet from the super-net and train the subnet.
    For the validation dataset, the agent sample subnet from the super-net and evaluate the subnet;
    the agent uses the metric evaluated as rewards, put into replay buffer and updates itself.

    As the process is similar to the multi-trial version :class:`~nni.nas.strategy.PolicyBasedRL`,
    this strategy shares some implementations and parameters with it.

    .. attention::

       ENAS requires the evaluator to report metrics via ``self.log`` in its ``validation_step``.
       See explanation of ``reward_metric_name`` for details.

    The supported mutation primitives of ENAS are:

    * :class:`nni.nas.nn.pytorch.LayerChoice`.
    * :class:`nni.nas.nn.pytorch.InputChoice` (only when ``n_chosen == 1`` or ``n_chosen is None``).
    * :class:`nni.nas.nn.pytorch.ParametrizedModule` (only when parameters are choices and type is in {supported_ops}).
    * :class:`nni.nas.nn.pytorch.Repeat`.
    * :class:`nni.nas.nn.pytorch.Cell`.

    {optimization_note}

    Parameters
    ----------
    batches_per_update
        Number of steps for which the gradients will be accumulated,
        before updating the weights of RL controller.
    log_prob_every_n_step
        Log the probability of choices every N steps. Useful for visualization and debugging.
    replay_buffer_size
        Size of replay buffer.
        If it's none, the size will be the expected trajectory length times ``batches_per_update``.
    reward_metric_name
        The name of the metric which is treated as reward.
        This will be not effective when there's only one metric returned from evaluator.
        If there are multiple, by default, it will find the metric with key name ``default``.
        If reward_metric_name is specified, it will find reward_metric_name.
        Otherwise it raises an exception indicating multiple metrics are found.
    policy_fn
        See :class:`~nni.nas.strategy.PolicyBasedRL`.
    update_kwargs
        See :class:`~nni.nas.strategy.PolicyBasedRL`.
    warmup_epochs
        The first ``warmup_epochs`` do not update architecture weights.
    penalty
        If a dict, it should contain the keys: ``profiler``, ``baseline``, and optionally ``scale``, ``nonlinear``, ``aggregate``.
        We will create a :class:`~nni.nas.oneshot.pytorch.profiler.SampleProfilerPenalty` with the given parameters.
        Note that the penalty is operated on the reward, not the loss.
        Thus in most cases, the ``scale`` should be set to a negative value.
    """.format(
        supported_ops=', '.join(NATIVE_SUPPORTED_OP_NAMES),
        optimization_note=MANUAL_OPTIMIZATION_NOTE
    )

    def __init__(self, *,
                 batches_per_update: float = 20,
                 log_prob_every_n_step: int = 10,
                 replay_buffer_size: int | None = None,
                 reward_metric_name: str | None = None,
                 policy_fn: PolicyFactory | None = None,
                 update_kwargs: dict | None = None,
                 warmup_epochs: int = 0,
                 penalty: dict | ExpectationProfilerPenalty | SampleProfilerPenalty | None = None,
                 **kwargs):
        super().__init__(**kwargs)

        if self.filter is not None:
            raise ValueError('ENAS does not support sampling filter.')

        self.batches_per_update = batches_per_update
        self.log_prob_every_n_step = log_prob_every_n_step
        self.replay_buffer_size = replay_buffer_size
        self.reward_metric_name = reward_metric_name
        self.policy_fn = policy_fn
        self.update_kwargs = {'batch_size': 32, 'repeat': 5, 'update_times': 5} if update_kwargs is None else update_kwargs
        self.warmup_epochs = warmup_epochs

        if isinstance(penalty, dict):
            self.penalty = SampleProfilerPenalty(**penalty)
        else:
            self.penalty = penalty
        if self.penalty is not None and self.penalty.scale > 0:
            _logger.warning('The penalty in ENAS is combined with the reward. So in most cases, scale should be less than 0. '
                            'If you want to use a positive scale, your reward should be minimizing.')

    def configure_oneshot_module(self, training_module: LightningModule) -> EnasLightningModule:
        return EnasLightningModule(
            training_module,
            batches_per_update=self.batches_per_update,
            log_prob_every_n_step=self.log_prob_every_n_step,
            replay_buffer_size=self.replay_buffer_size,
            reward_metric_name=self.reward_metric_name,
            policy_fn=self.policy_fn,
            update_kwargs=self.update_kwargs,
            warmup_epochs=self.warmup_epochs,
            penalty=self.penalty
        )

    def train_dataloader(self, train_dataloader_fn, val_dataloader_fn):
        import pytorch_lightning
        if pytorch_lightning.__version__.startswith('1.'):
            from ._dataloader_legacy import ConcatLoader
        else:
            from ._dataloader import ConcatLoader
        return ConcatLoader({
            'train': train_dataloader_fn(),
            'val': val_dataloader_fn()
        })

    def val_dataloader(self, train_dataloader_fn, val_dataloader_fn):
        return None

    def mutate_model(self, model: ModelSpaceType) -> ModelSpaceType:
        for mutable in model.simplify().values():
            if not (isinstance(mutable, Categorical) or (
                isinstance(mutable, CategoricalMultiple) and mutable.n_chosen in (1, None)
            )):
                raise TypeError(f'ENAS strategy only supports categorical variables, but got {type(mutable)}')
        return super().mutate_model(model)
