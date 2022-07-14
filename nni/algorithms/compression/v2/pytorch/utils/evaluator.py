# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from copy import deepcopy
import logging
import types
from typing import Dict, List, Tuple, Union, Any, Callable, Optional

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.hooks import RemovableHandle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from nni.common import is_traceable
from .constructor_helper import OptimizerConstructHelper, LRSchedulerConstructHelper

_logger = logging.getLogger(__name__)


class Hook:
    """
    The base class used to generate, register and remove torch hook.

    Parameters
    ----------
    target
        The hook target, a torch.Tensor or a torch.nn.Module.
    target_name
        The name of the target, use periods to separate, e.g., 'model.layers.0.conv1.weight'.
    hook_factory
        A factory fucntion, input is an empty list, output is a hook function.
        The empty list is used to store some useful information in hook.
    """

    def __init__(self, target: Module | Tensor, target_name: str, hook_factory: Callable[[List], Callable]):
        self.target = target
        self.target_name = target_name
        self.hook_factory = hook_factory
        self.buffer: List = []
        self.handle: RemovableHandle | None = None

    def _register(self, hook_func: Callable) -> RemovableHandle:
        raise NotImplementedError

    def register(self):
        if self.handle is not None:
            _logger.warning('%s for %s already has been registered.', self.__class__.__name__, self.target_name)
            return
        self.handle = self._register(self.hook_factory(self.buffer))

    def remove(self):
        if self.handle is None:
            _logger.warning('%s for %s has not been registered yet.', self.__class__.__name__, self.target_name)
            return
        self.handle.remove()
        self.handle = None
        self.buffer = []


class TensorHook(Hook):
    """
    Here is an example for hook_factory, in this example, the gradient on this tensor will be saved in the buffer::

        def hook_factory(buffer):
            def hook(grad):
                buffer.append(grad.clone())
            return hook
    """

    def __init__(self, target: Tensor, target_name: str, hook_factory: Callable[[List], Callable[[Tensor], Any]]):
        assert isinstance(target, Tensor)
        super().__init__(target, target_name, hook_factory)

    def _register(self, hook_func: Callable[[Tensor], Any]) -> RemovableHandle:
        return self.target.register_hook(hook_func)  # type: ignore


class ModuleHook(Hook):
    def __init__(self, target: Module, target_name: str, hook_factory: Callable[[List], Callable[[Module, Tensor, Tensor], Any]]):
        assert isinstance(target, Module)
        super().__init__(target, target_name, hook_factory)


class ForwardHook(ModuleHook):
    """
    Here is an example for hook_factory, in this example, the output of this module will be saved in the buffer::

        def hook_factory(buffer):
            def hook(module, input, output):
                buffer.append(output.clone())
            return hook
    """

    def _register(self, hook_func: Callable[[Module, Tensor, Tensor], Any]):
        return self.target.register_forward_hook(hook_func)  # type: ignore


class BackwardHook(ModuleHook):
    """
    Here is an example for hook_factory, in this example, the gradient of this module input will be saved in the buffer::

        def hook_factory(buffer):
            def hook(module, grad_input, grad_output):
                buffer.append(grad_input.clone())
            return hook
    """

    def _register(self, hook_func: Callable[[Module, Tensor, Tensor], Any]):
        return self.target.register_backward_hook(hook_func)  # type: ignore


class Evaluator:
    """
    Evaluator is a package for the training & evaluation process. In model compression,
    NNI have the need to intervene in the training process to collect intermediate information,
    and even modify part of the training loop. Evaluator provides a series of member functions that are convenient to modify these,
    and the compressor can easily intervene in training by calling these functions.

    Notes
    -----
    Users are not recommended to use any member functions of this class.
    """

    # A flag to indicate whether the evaluator is initialized complete.
    _initialization_complete: bool
    _hook: List[Hook]

    def _init_optimizer_helpers(self, pure_model: Module | pl.LightningModule):
        """
        This is an internal API, ``pure_model`` means the model is the original model passed in by the user,
        it should not be the modified model (wrapped, hooked, or patched by NNI).
        That is, the optimizers & lr_schedulers obtained by ``Evaluator`` match the ``pure_model``.

        This function is used to record the status of the optimizers & lr_schedulers,
        and ensure NNI can reinitialize the optimizers & lr_schedulers with a similar but modified model.

        Notes
        -----
        This is a part of Evaluator initialization, please make sure this function has been called before using other evaluator functions.
        """
        raise NotImplementedError

    def bind_model(self, model: Module | pl.LightningModule, param_names_map: Dict[str, str] | None = None):
        """
        Bind the model suitable for this ``Evaluator`` to use the evaluator's abilities of model modification, model training, and model evaluation.

        Parameter
        ---------
        model
            The model bind to this ``Evaluator``, usually a wrapped model.
        param_names_map
            ``param_names_map`` maps the names of the parameters in the pure_model to the names of the parameters in the bound model.
            The format of param_names_map is {pure_model_param_name: bound_model_param_name}.
            It is for initializing the optimizers for the bound model.
        """
        raise NotImplementedError

    def unbind_model(self):
        """
        Unbind the model bound by ``bind_model``. Then ``Evaluator`` can be reused by binding a new model by `bind_model`.
        """
        raise NotImplementedError

    def patch_loss(self, patch: Callable[[Tensor], Tensor]):
        """
        The patch may add additional loss or replace the original loss. Here is an example::

            def loss_patch(original_loss):
                params_norm = 0
                for param in model.parameters():
                    params_norm += torch.norm(param)
                return original_loss + params_norm

        Something like ``loss = patch(criterion(result, target))`` will happen during each time loss computation.
        """
        raise NotImplementedError

    def revert_loss(self):
        """
        Revert the loss to the original one.
        """
        raise NotImplementedError

    def patch_optimizer_step(self, before_step_tasks: List[Callable], after_step_tasks: List[Callable]):
        """
        Run tasks in `before_step_tasks` before `optimizer.step()` each time.
        Run tasks in `after_step_tasks` after `optimizer.step()` each time.

        Notes
        -----
        If the model has multiple optimizers, this function only patches tasks to the first optimizer right now.
        """
        raise NotImplementedError

    def revert_optimizer_step(self):
        """
        Revert the optimizer step to the original one.
        """
        raise NotImplementedError

    def register_hooks(self, hooks: List[Hook]):
        """
        The input is a list of ``TensorHook``, ``ForwardHook``, ``BackwardHook``,
        please view how to use ``TensorHook``, ``ForwardHook``, ``BackwardHook``.
        This function will call ``Hook.register()`` of hook in ``hooks``, and record the hook in ``self._hooks``.
        """
        if not hasattr(self, '_hooks'):
            self._hooks: List[Hook] = []
        for hook in hooks:
            hook.register()
            self._hooks.append(hook)

    def get_all_hooks(self) -> List[Hook]:
        """
        Get all registered ``Hook``.
        """
        return getattr(self, '_hooks', [])

    def remove_all_hooks(self):
        """
        Call ``Hook.remove()`` of all ``Hook`` instances in ``self._hooks``, then clear ``self._hooks``.
        """
        if hasattr(self, '_hooks'):
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()

    def train(self, max_steps: int | None = None, max_epochs: int | None = None):
        """
        Train the bound model with default optimization loop defined by user and only change the training duration.
        """
        raise NotImplementedError

    def finetune(self):
        """
        Finetune the bound model with default optimization loop defined by user.
        """
        raise NotImplementedError

    def evaluate(self) -> float | None | Tuple[float, Any] | Tuple[None, Any]:
        """
        NNI assume the evaluation function user passed in should return a float number or a dict as metric.
        If the evaluation function returned a dict, take the value with dict key ``default`` as the first element of ``evaluate`` returned value,
        and put the dict as the second element of the returned value.
        For any other type of the metric returned by evaluation function, ``evaluate`` will directly returned
        (it should be a float, but NNI does not prevent other types from being returned, this will handle by the object calling ``evaluate``).
        """
        # Note that the first item of the returned value will be used as the default metric used by NNI.
        raise NotImplementedError

    def get_dummy_input(self) -> Any:
        """
        The returned value is a dummy input for the model, always used by ``torch.jit.trace``.
        """
        raise NotImplementedError


class LightningEvaluator(Evaluator):
    """
    LightningEvaluator is the Evaluator based on PytorchLightning.
    It is very friendly to the users who are familiar to PytorchLightning
    or already have training/validation/testing code written in PytorchLightning.
    The only need is to use ``nni.trace`` to trace the Trainer & LightningDataModule.

    Additionally, please make sure the ``Optimizer`` class and ``LR_Scheduler`` class used in ``LightningModule.configure_optimizers()``
    are also be traced by ``nni.trace``.

    Parameters
    ----------
    trainer
        Pytorch-Lightning Trainer. It should be traced by nni, e.g., ``trainer = nni.trace(pl.Trainer)(...)``.
    data_module
        Pytorch-Lightning LightningDataModule. It should be traced by nni, e.g., ``data_module = nni.trace(pl.LightningDataModule)(...)``.
    dummy_input
        The dummy_input is used to trace the graph. If dummy_input is not given, will use the data in data_module.train_dataloader().

    Notes
    -----
    If the the test metric is needed by nni, please make sure log metric with key ``default`` in ``LightningModule.test_step()``.
    """

    def __init__(self, trainer: pl.Trainer, data_module: pl.LightningDataModule,
                 dummy_input: Any | None = None):
        err_msg = 'Only support traced {}, please use nni.trace({}) to initialize the trainer.'
        assert isinstance(trainer, pl.Trainer) and is_traceable(trainer), err_msg.format('pytorch_lightning.Trainer', 'pytorch_lightning.Trainer')
        assert isinstance(data_module, pl.LightningDataModule) and is_traceable(data_module), err_msg.format('pytorch_lightning.LightningDataModule', 'pytorch_lightning.LightningDataModule')
        self.trainer = trainer
        self.data_module = data_module
        self._dummy_input = dummy_input

        self.model: pl.LightningModule | None = None
        self._ori_model_attr = {}
        self._param_names_map: Dict[str, str] | None = None

        self._initialization_complete = False

    def _init_optimizer_helpers(self, pure_model: pl.LightningModule):
        assert self._initialization_complete is False, 'Evaluator initialization is already complete.'

        self._optimizer_helpers = []
        self._lr_scheduler_helpers = []
        # record i-th lr_scheduler scheduling j-th optimizer lr
        self._lrs_opt_map = {}
        # record `LightningModule.configure_optimizers` 6-th option returned dict information
        self._opt_returned_dicts = []

        # The return value of `configure_optimizers` may one of the following six options:
        optimizers_lr_schedulers: Any = pure_model.configure_optimizers()
        # 1. None - Fit will run without any optimizer.
        if optimizers_lr_schedulers is None:
            err_msg = 'NNI does not support `LightningModule.configure_optimizers` returned None, '
            err_msg += 'if you have a reason why you must, please file an issue at https://github.com/microsoft/nni/issues'
            raise ValueError(err_msg)
        # 2. Single optimizer.
        # 3. Dictionary, with an "optimizer" key, and (optionally) a "lr_scheduler" key whose value is a single LR scheduler or lr_scheduler_config.
        elif isinstance(optimizers_lr_schedulers, (Optimizer, dict)):
            optimizers_lr_schedulers = [optimizers_lr_schedulers]

        err_msg = f'Got an wrong returned value type of `LightningModule.configure_optimizers`: {type(optimizers_lr_schedulers).__name__}'
        assert isinstance(optimizers_lr_schedulers, (list, tuple)), err_msg

        # 4. Two lists - the first list has multiple optimizers, and the second has multiple LR schedulers (or multiple lr_scheduler_config).
        if isinstance(optimizers_lr_schedulers[0], (list, tuple)):
            optimizers, lr_schedulers = optimizers_lr_schedulers
            self._optimizer_helpers = [OptimizerConstructHelper.from_trace(pure_model, optimizer) for optimizer in optimizers]
            self._lr_scheduler_helpers = [LRSchedulerConstructHelper.from_trace(lr_scheduler) for lr_scheduler in lr_schedulers]
            optimizer_ids_map = {id(optimizer): i for i, optimizer in enumerate(optimizers)}
            self._lrs_opt_map = {i: optimizer_ids_map[id(lr_scheduler.optimizer)] for i, lr_scheduler in enumerate(lr_schedulers)}
        # 5. List or Tuple of optimizers.
        elif isinstance(optimizers_lr_schedulers[0], Optimizer):
            self._optimizer_helpers = [OptimizerConstructHelper.from_trace(pure_model, optimizer) for optimizer in optimizers_lr_schedulers]
        # 6. Tuple of dictionaries as described above, with an optional "frequency" key.
        elif isinstance(optimizers_lr_schedulers[0], dict):
            optimizer_ids_map = {}
            lr_scheduler_opt_ids_map = {}
            optimizer_count = 0
            scheduler_count = 0
            for opt_dict in optimizers_lr_schedulers:
                opt_dict: Dict
                self._optimizer_helpers.append(OptimizerConstructHelper.from_trace(pure_model, opt_dict['optimizer']))
                optimizer_ids_map[id(opt_dict['optimizer'])] = optimizer_count
                opt_dict['optimizer'] = optimizer_count
                optimizer_count += 1

                lr_scheduler = opt_dict.get('lr_scheduler', {}).get('scheduler', None)
                if lr_scheduler is not None:
                    self._lr_scheduler_helpers.append(LRSchedulerConstructHelper.from_trace(lr_scheduler))
                    lr_scheduler_opt_ids_map[scheduler_count] = id(lr_scheduler.optimizer)
                    opt_dict['lr_scheduler']['scheduler'] = scheduler_count
                    scheduler_count += 1
                self._opt_returned_dicts.append(opt_dict)
            self._lrs_opt_map = {scheduler_count: optimizer_ids_map[opt_id] for scheduler_count, opt_id in lr_scheduler_opt_ids_map.items()}
        else:
            err_msg = 'Got an wrong returned value type of `LightningModule.configure_optimizers`: '
            err_msg += f'list or tuple of {type(optimizers_lr_schedulers[0]).__name__}'
            raise TypeError(err_msg)

        self._initialization_complete = True

    def bind_model(self, model: pl.LightningModule, param_names_map: Dict[str, str] | None = None):
        assert self._initialization_complete is True, 'Evaluator initialization is not complete, please call `_init_optimizer_helpers` before bind model.'
        assert isinstance(model, pl.LightningModule)
        if self.model is not None:
            _logger.warning('Already bound a model, will unbind it before bind a new model.')
            self.unbind_model()

        self.model = model
        self._ori_model_attr.update({
            'training_step': model.training_step,
            'configure_optimizers': model.configure_optimizers,
            'configure_callbacks': model.configure_callbacks
        })
        self._param_names_map = param_names_map
        self._patch_configure_optimizers()

    def unbind_model(self):
        if self.model:
            self.revert_loss()
            self.revert_optimizer_step()
            self.remove_all_hooks()
            self._revert_configure_optimizers()
            self._param_names_map = None
            self._ori_model_attr.clear()
            self.model = None
        else:
            _logger.warning('Did not bind any model, no need to unbind model.')

    def _patch_configure_optimizers(self):
        assert isinstance(self.model, pl.LightningModule)

        if self._opt_returned_dicts:
            def new_configure_optimizers(_):  # type: ignore
                optimizers = [opt_helper.call(self.model, self._param_names_map) for opt_helper in self._optimizer_helpers]  # type: ignore
                lr_schedulers = [lrs_helper.call(optimizers[self._lrs_opt_map[i]]) for i, lrs_helper in enumerate(self._lr_scheduler_helpers)]
                opt_lrs_dicts = deepcopy(self._opt_returned_dicts)
                for opt_lrs_dict in opt_lrs_dicts:
                    opt_lrs_dict['optimizer'] = optimizers[opt_lrs_dict['optimizer']]
                    if 'lr_scheduler' in opt_lrs_dict:
                        opt_lrs_dict['lr_scheduler']['scheduler'] = lr_schedulers[opt_lrs_dict['lr_scheduler']['scheduler']]
                return opt_lrs_dicts
        elif self._lr_scheduler_helpers:
            def new_configure_optimizers(_):  # type: ignore
                optimizers = [opt_helper.call(self.model, self._param_names_map) for opt_helper in self._optimizer_helpers]  # type: ignore
                lr_schedulers = [lrs_helper.call(optimizers[self._lrs_opt_map[i]]) for i, lrs_helper in enumerate(self._lr_scheduler_helpers)]
                return optimizers, lr_schedulers
        else:
            def new_configure_optimizers(_):
                optimizers = [opt_helper.call(self.model, self._param_names_map) for opt_helper in self._optimizer_helpers]  # type: ignore
                return optimizers

        self.model.configure_optimizers = types.MethodType(new_configure_optimizers, self.model)

    def _revert_configure_optimizers(self):
        assert isinstance(self.model, pl.LightningModule)
        self.model.configure_optimizers = self._ori_model_attr['configure_optimizers']

    def patch_loss(self, patch: Callable[[Tensor], Tensor]):
        assert isinstance(self.model, pl.LightningModule)
        old_training_step = self.model.training_step

        def patched_training_step(_, *args, **kwargs):
            output = old_training_step(*args, **kwargs)
            if isinstance(output, Tensor):
                output = patch(output)
            else:
                output['loss'] = patch(output['loss'])
            return output

        self.model.training_step = types.MethodType(patched_training_step, self.model)

    def revert_loss(self):
        assert isinstance(self.model, pl.LightningModule)
        self.model.training_step = self._ori_model_attr['training_step']

    def patch_optimizer_step(self, before_step_tasks: List[Callable], after_step_tasks: List[Callable]):
        assert isinstance(self.model, pl.LightningModule)

        class OptimizerCallback(Callback):
            def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Optimizer, opt_idx: int) -> None:
                for task in before_step_tasks:
                    task()

            def on_before_zero_grad(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Optimizer) -> None:
                for task in after_step_tasks:
                    task()

        old_configure_callbacks = self.model.configure_callbacks

        def patched_configure_callbacks(_):
            callbacks = old_configure_callbacks()
            callbacks.append(OptimizerCallback())  # type: ignore
            return callbacks

        self.model.configure_callbacks = types.MethodType(patched_configure_callbacks, self.model)

    def revert_optimizer_step(self):
        assert isinstance(self.model, pl.LightningModule)
        self.model.configure_callbacks = self._ori_model_attr['configure_callbacks']

    def train(self, max_steps: int | None = None, max_epochs: int | None = None):
        assert isinstance(self.model, pl.LightningModule)
        # reset trainer
        trainer: pl.Trainer = self.trainer.trace_copy().get()  # type: ignore
        # NOTE: lightning may dry run some steps at first for sanity check in Trainer.fit() by default,
        # If we want to record some information in the forward hook, we may get some additional information,
        # so using Trainer.num_sanity_val_steps = 0 disable sanity check.
        trainer.num_sanity_val_steps = 0

        if max_steps:
            trainer.fit_loop.max_steps = max_steps
        if max_epochs:
            trainer.fit_loop.max_epochs = max_epochs
        trainer.fit(self.model, self.data_module)

        # del trainer reference, we don't want to dump trainer when we dump the entire model.
        self.model.trainer = None

    def finetune(self):
        self.train()

    def evaluate(self) -> Tuple[float | None, List[Dict[str, float]]]:
        """
        NNI will use metric with key ``default`` for evaluating model, please make sure you have this key in your ``Trainer.test()`` returned metric dicts.
        If ``Trainer.test()`` returned list contains multiple dicts with key ``default``, NNI will take their average as the final metric.
        E.g., if ``Trainer.test()`` returned ``[{'default': 0.8, 'loss': 2.3}, {'default': 0.6, 'loss': 2.4}, {'default': 0.7, 'loss': 2.3}]``,
        NNI will take the final metric ``(0.8 + 0.6 + 0.7) / 3 = 0.7``.
        """
        assert isinstance(self.model, pl.LightningModule)
        # reset trainer
        trainer: pl.Trainer = self.trainer.trace_copy().get()  # type: ignore
        original_results = trainer.test(self.model, self.data_module)
        # del trainer reference, we don't want to dump trainer when we dump the entire model.
        self.model.trainer = None
        nni_metrics_list = [metrics['default'] for metrics in original_results if 'default' in metrics]
        if nni_metrics_list:
            nni_metric = sum(nni_metrics_list) / len(nni_metrics_list)
        else:
            nni_metric = None
        return nni_metric, original_results

    def get_dummy_input(self) -> Any:
        if self._dummy_input is not None:
            return self._dummy_input
        try:
            return next(iter(self.data_module.train_dataloader()))
        except Exception as e:
            _logger.error('Get default dummy input failed, please manually set dummy_input.')
            raise e


_CRITERION = Callable[[Any, Any], Any]
_EVALUATING_FUNC = Callable[[Module], Union[float, Dict]]
_TRAINING_FUNC = Callable[[Module, Union[Optimizer, List[Optimizer]], _CRITERION, Union[None, _LRScheduler, List[_LRScheduler]], Optional[int], Optional[int]], None]


class TorchEvaluator(Evaluator):
    """
    TorchEvaluator is the Evaluator for native Pytorch users.
    It has some requirements for the writing of the training loop, please refer to the documentation for details.

    Parameters
    ----------
    training_func
        The training function is used to train the model, note that this a entire optimization training loop.
        It should have three required parameters [model, optimizers, criterion] and three optional parameters [schedulers, max_steps, max_epochs].
        ``optimizers`` can be an instance of ``torch.optim.Optimizer`` or a list of ``torch.optim.Optimizer``, it belongs to the ``optimizers`` pass to ``TorchEvaluator``.
        ``criterion`` and ``schedulers`` are also belonging to the ``criterion`` and ``schedulers`` pass to ``TorchEvaluator``.
        ``max_steps`` and ``max_epochs`` are used to control the training duration.

        Example::

            def training_func(model: Module, optimizer: Optimizer, criterion: Callable, scheduler: _LRScheduler,
                              max_steps: int | None = None, max_epochs: int | None = None, *args, **kwargs):
                model.train()

                # prepare data
                data_dir = Path(__file__).parent / 'data'
                MNIST(data_dir, train=True, download=True)
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                mnist_train = MNIST(data_dir, train=True, transform=transform)
                train_dataloader = DataLoader(mnist_train, batch_size=32)

                max_epochs = max_epochs if max_epochs else 3
                max_steps = max_steps if max_steps else 6000
                current_steps = 0

                # training
                for _ in range(max_epochs):
                    for x, y in train_dataloader:
                        optimizer.zero_grad()
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        loss: torch.Tensor = criterion(logits, y)
                        loss.backward()
                        optimizer.step()
                        current_steps += 1
                        if max_steps and current_steps == max_steps:
                            return
                    scheduler.step()

    optimziers
        The traced optimizer instance which the optimizer class is wrapped by nni.trace.
        E.g. ``traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters())``.
    criterion
        The criterion function used in trainer. Take model output and target as input, and return the loss.
        E.g. ``criterion = torch.nn.functional.nll_loss``.
    lr_schedulers
        Optional. The traced _LRScheduler instance which the lr scheduler class is wrapped by nni.trace.
        E.g. ``traced_lr_scheduler = nni.trace(ExponentialLR)(optimizer, 0.1)``.
    dummy_input
        Optional. The dummy_input is used to trace the graph, the same with ``example_inputs`` in ``torch.jit.trace(func, example_inputs, ...)``.
    evaluating_func
        Optional. A function that input is model and return the evaluation metric.
        The return value can be a single float or a tuple (float, Any).

        Example::

            def evaluating_func(model: Module):
                model.eval()

                # prepare data
                data_dir = Path(__file__).parent / 'data'
                MNIST(data_dir, train=False, download=True)
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                mnist_test = MNIST(data_dir, train=False, transform=transform)
                test_dataloader = DataLoader(mnist_test, batch_size=32)

                # testing
                correct = 0
                with torch.no_grad():
                    for x, y in test_dataloader:
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        preds = torch.argmax(logits, dim=1)
                        correct += preds.eq(y.view_as(preds)).sum().item()
                return correct / len(mnist_test)
    """

    def __init__(self, training_func: _TRAINING_FUNC, optimizers: Optimizer | List[Optimizer], criterion: _CRITERION,
                 lr_schedulers: _LRScheduler | List[_LRScheduler] | None = None, dummy_input: Any | None = None,
                 evaluating_func: _EVALUATING_FUNC | None = None):
        self.training_func = training_func
        self._ori_criterion = criterion
        self._criterion = self._ori_criterion
        self.dummy_input = dummy_input
        self.evaluating_func = evaluating_func

        self._train_with_single_optimizer = isinstance(optimizers, Optimizer)
        self._train_with_single_scheduler = isinstance(lr_schedulers, _LRScheduler)

        self.model: Module | None = None
        self._optimizers: List[Optimizer] | None = None
        self._lr_schedulers: List[_LRScheduler] | None = None
        self._first_optimizer_step: Callable | None = None
        self._param_names_map: Dict[str, str] | None = None

        # will del self._tmp_optimizers and self._tmp_lr_schedulers in `_init_optimizer_helpers`
        self._tmp_optimizers = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
        assert all(isinstance(optimizer, Optimizer) and is_traceable(optimizer) for optimizer in self._tmp_optimizers)
        self._tmp_lr_schedulers = lr_schedulers if isinstance(lr_schedulers, (list, tuple)) else [lr_schedulers] if lr_schedulers else []
        assert all(isinstance(lr_scheduler, _LRScheduler) and is_traceable(lr_scheduler) for lr_scheduler in self._tmp_lr_schedulers)
        self._initialization_complete = False

    def _init_optimizer_helpers(self, pure_model: Module):
        assert self._initialization_complete is False, 'Evaluator initialization is already complete.'

        self._optimizer_helpers = [OptimizerConstructHelper.from_trace(pure_model, optimizer) for optimizer in self._tmp_optimizers]
        self._lr_scheduler_helpers = [LRSchedulerConstructHelper.from_trace(lr_scheduler) for lr_scheduler in self._tmp_lr_schedulers]
        optimizer_ids_map = {id(optimizer): i for i, optimizer in enumerate(self._tmp_optimizers)}
        # record i-th lr_scheduler scheduling j-th optimizer lr
        self._lrs_opt_map = {i: optimizer_ids_map[id(lr_scheduler.optimizer)] for i, lr_scheduler in enumerate(self._tmp_lr_schedulers)}  # type: ignore

        delattr(self, '_tmp_optimizers')
        delattr(self, '_tmp_lr_schedulers')
        self._initialization_complete = True

    def bind_model(self, model: Module, param_names_map: Dict[str, str] | None = None):
        assert self._initialization_complete is True, 'Evaluator initialization is not complete, please call `_init_optimizer_helpers` before bind model.'
        assert isinstance(model, Module)
        if self.model is not None:
            _logger.warning('Already bound a model, will unbind it before bind a new model.')
            self.unbind_model()

        self.model = model
        self._param_names_map = param_names_map
        # initialize optimizers & lr_schedulers for the bound model here
        self._optimizers = [helper.call(model, param_names_map) for helper in self._optimizer_helpers]
        self._lr_schedulers = [lrs_helper.call(self._optimizers[self._lrs_opt_map[i]]) for i, lrs_helper in enumerate(self._lr_scheduler_helpers)]
        self._first_optimizer_step = self._optimizers[0].step

    def unbind_model(self):
        if self.model:
            self.revert_loss()
            self.revert_optimizer_step()
            self.remove_all_hooks()
            self._first_optimizer_step = None
            self._lr_schedulers = None
            self._optimizers = None
            self._param_names_map = None
            self.model = None
        else:
            _logger.warning('Did not bind any model, no need to unbind model.')

    def patch_loss(self, patch: Callable[[Tensor], Tensor]):
        old_criterion = self._criterion

        def patched_criterion(*args, **kwargs):
            loss = old_criterion(*args, **kwargs)
            return patch(loss)

        self._criterion = patched_criterion

    def revert_loss(self):
        self._criterion = self._ori_criterion

    def patch_optimizer_step(self, before_step_tasks: List[Callable], after_step_tasks: List[Callable]):
        assert self._optimizers is not None
        old_step = self._optimizers[0].step

        def patched_step(_, *args, **kwargs):
            for task in before_step_tasks:
                task()
            # call origin optimizer step method
            output = old_step(*args, **kwargs)
            for task in after_step_tasks:
                task()
            return output

        self._optimizers[0].step = types.MethodType(patched_step, self._optimizers[0])

    def revert_optimizer_step(self):
        assert self._optimizers is not None
        if self._first_optimizer_step:
            self._optimizers[0].step = self._first_optimizer_step

    def train(self, max_steps: int | None = None, max_epochs: int | None = None):
        assert self.model is not None
        assert self._optimizers is not None
        assert self._criterion is not None
        optimizers = self._optimizers[0] if self._train_with_single_optimizer else self._optimizers
        lr_schedulers = self._lr_schedulers[0] if self._train_with_single_scheduler else self._lr_schedulers  # type: ignore
        self.training_func(self.model, optimizers, self._criterion, lr_schedulers, max_steps, max_epochs)

    def finetune(self):
        self.train()

    def evaluate(self) -> float | None | Tuple[float, Dict[str, Any]] | Tuple[None, Dict[str, Any]]:
        assert self.model is not None
        assert self.evaluating_func is not None
        metric = self.evaluating_func(self.model)
        if isinstance(metric, dict):
            nni_used_metric = metric.get('default', None)
            if nni_used_metric is None:
                warn_msg = f'Evaluation function returns a dict metric without key `default`, will return None as the model evaluation metric value.'
                _logger.warning(warn_msg)
            return nni_used_metric, metric
        else:
            return metric

    def get_dummy_input(self) -> Any:
        return self.dummy_input
