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
    def _register(self, hook_func: Callable[[Module, Tensor, Tensor], Any]):
        return self.target.register_forward_hook(hook_func)  # type: ignore


class BackwardHook(ModuleHook):
    def _register(self, hook_func: Callable[[Module, Tensor, Tensor], Any]):
        return self.target.register_backward_hook(hook_func)  # type: ignore


class Evaluator:

    def bind_model(self, model: Module | pl.LightningModule, param_names_map: Dict[str, str] | None = None):
        # param_names_map maps the names of the parameters in the pure_model to the names of the parameters in the binded model.
        # The format of param_names_map is {pure_model_param_name: binded_model_param_name}.
        # param_names_map is for initializing the optimizers for the binded model.
        raise NotImplementedError

    def unbind_model(self):
        # Evaluator can be reused by `unbind_model` then bind a new model by `bind_model`.
        raise NotImplementedError

    def patch_loss(self, patch: Callable[[Tensor], Tensor]):
        raise NotImplementedError

    def revert_loss(self):
        raise NotImplementedError

    def patch_optimizer_step(self, before_step_tasks: List[Callable], after_step_tasks: List[Callable]):
        # Run tasks in `before_step_tasks` before `optimizer.step()` each time.
        # Run tasks in `after_step_tasks` after `optimizer.step()` each time.
        # NOTE: we only patch these tasks to the first optimizer right now.
        raise NotImplementedError

    def revert_optimizer_step(self):
        raise NotImplementedError

    def register_hooks(self, hooks: List[Hook]):
        raise NotImplementedError

    def remove_all_hooks(self):
        raise NotImplementedError

    def train(self, max_steps: int | None = None, max_epochs: int | None = None):
        raise NotImplementedError

    def finetune(self):
        raise NotImplementedError

    def evaluate(self) -> float | Tuple[float, Any]:
        # Note that the first item of the returned value will be used as the default metric used by NNI.
        raise NotImplementedError

    def get_dummy_input(self) -> Any:
        raise NotImplementedError


class LightningEvaluator(Evaluator):
    def __init__(self, pure_model: pl.LightningModule, trainer: pl.Trainer, data_module: pl.LightningDataModule,
                 dummy_input: Any | None = None):
        assert isinstance(trainer, pl.Trainer)
        assert isinstance(data_module, pl.LightningDataModule)
        self.trainer = trainer
        self.data_module = data_module
        self._dummy_input = dummy_input

        self._optimizer_helpers, self._lr_scheduler_helpers, self._lrs_opt_map, self._opt_returned_dicts = \
            self._init_optimizer_helpers(pure_model)

        self.model: pl.LightningModule | None = None
        self._hooks: List[Hook] = []
        self._ori_model_attr = {}
        self._param_names_map: Dict[str, str] | None = None

    def _init_optimizer_helpers(self, pure_model: pl.LightningModule):
        optimizer_helpers = []
        lr_scheduler_helpers = []
        # record i-th lr_scheduler scheduling j-th optimizer lr
        lrs_opt_map = {}
        # record `LightningModule.configure_optimizers` 6-th option returned dict information
        opt_returned_dicts = []

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
            optimizer_helpers = [OptimizerConstructHelper.from_trace(pure_model, optimizer) for optimizer in optimizers]
            lr_scheduler_helpers = [LRSchedulerConstructHelper.from_trace(lr_scheduler) for lr_scheduler in lr_schedulers]
            optimizer_ids_map = {id(optimizer): i for i, optimizer in enumerate(optimizers)}
            lrs_opt_map = {i: optimizer_ids_map[id(lr_scheduler.optimizer)] for i, lr_scheduler in enumerate(lr_schedulers)}
        # 5. List or Tuple of optimizers.
        elif isinstance(optimizers_lr_schedulers[0], Optimizer):
            optimizer_helpers = [OptimizerConstructHelper.from_trace(pure_model, optimizer) for optimizer in optimizers_lr_schedulers]
        # 6. Tuple of dictionaries as described above, with an optional "frequency" key.
        elif isinstance(optimizers_lr_schedulers[0], dict):
            optimizer_ids_map = {}
            lr_scheduler_opt_ids_map = {}
            optimizer_count = 0
            scheduler_count = 0
            for opt_dict in optimizers_lr_schedulers:
                opt_dict: Dict
                optimizer_helpers.append(OptimizerConstructHelper.from_trace(pure_model, opt_dict['optimizer']))
                optimizer_ids_map[id(opt_dict['optimizer'])] = optimizer_count
                opt_dict['optimizer'] = optimizer_count
                optimizer_count += 1

                lr_scheduler = opt_dict.get('lr_scheduler', {}).get('scheduler', None)
                if lr_scheduler is not None:
                    lr_scheduler_helpers.append(LRSchedulerConstructHelper.from_trace(lr_scheduler))
                    lr_scheduler_opt_ids_map[scheduler_count] = id(lr_scheduler.optimizer)
                    opt_dict['lr_scheduler']['scheduler'] = scheduler_count
                    scheduler_count += 1
                opt_returned_dicts.append(opt_dict)
            lrs_opt_map = {scheduler_count: optimizer_ids_map[opt_id] for scheduler_count, opt_id in lr_scheduler_opt_ids_map.items()}
        else:
            err_msg = 'Got an wrong returned value type of `LightningModule.configure_optimizers`: '
            err_msg += f'list or tuple of {type(optimizers_lr_schedulers[0]).__name__}'
            raise TypeError(err_msg)

        return optimizer_helpers, lr_scheduler_helpers, lrs_opt_map, opt_returned_dicts

    def bind_model(self, model: pl.LightningModule, param_names_map: Dict[str, str] | None = None):
        assert isinstance(model, pl.LightningModule)
        self.model = model
        self._ori_model_attr.update({
            'training_step': model.training_step,
            'configure_optimizers': model.configure_optimizers,
            'configure_callbacks': model.configure_callbacks
        })
        self._param_names_map = param_names_map
        self._patch_configure_optimizers()

    def unbind_model(self):
        self.revert_loss()
        self.revert_optimizer_step()
        self.remove_all_hooks()
        self._revert_configure_optimizers()
        self._param_names_map = None
        self._ori_model_attr.clear()
        self.model = None

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

    def register_hooks(self, hooks: List[Hook]):
        for hook in hooks:
            hook.register()
            self._hooks.append(hook)

    def remove_all_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def train(self, max_steps: int | None = None, max_epochs: int | None = None):
        assert isinstance(self.model, pl.LightningModule)
        ori_max_steps, ori_max_epochs = self.trainer.fit_loop.max_steps, self.trainer.fit_loop.max_epochs
        if max_steps:
            self.trainer.fit_loop.max_steps = max_steps
        if max_epochs:
            self.trainer.fit_loop.max_epochs = max_epochs
        self.trainer.fit(self.model, self.data_module)
        self.trainer.fit_loop.max_steps, self.trainer.fit_loop.max_epochs = ori_max_steps, ori_max_epochs

    def finetune(self):
        self.train()

    def evaluate(self) -> Tuple[float | None, List[Dict[str, float]]]:
        """
        NNI will use metric with key `NNI_METRIC` for evaluating model, please make sure you have this key in your `Trainer.test()` returned metric dicts.
        If `Trainer.test()` returned list contains multiple dicts with key `NNI_METRIC`, NNI will take their average as the final metric.
        E.g., if `Trainer.test()` returned `[{'NNI_METRIC': 0.8, 'loss': 2.3}, {'NNI_METRIC': 0.6, 'loss': 2.4}, {'NNI_METRIC': 0.7, 'loss': 2.3}]`,
        NNI will take the final metric `(0.8 + 0.6 + 0.7) / 3 = 0.7`.
        """
        assert isinstance(self.model, pl.LightningModule)
        original_results = self.trainer.test(self.model, self.data_module)
        nni_metrics_list = [metrics['NNI_METRIC'] for metrics in original_results if 'NNI_METRIC' in metrics]
        if nni_metrics_list:
            nni_metric = sum(nni_metrics_list) / len(nni_metrics_list)
        else:
            nni_metric = None
        return nni_metric, original_results

    def get_dummy_input(self) -> Any:
        if self._dummy_input:
            return self._dummy_input
        return next(iter(self.data_module.train_dataloader()))


_CRITERION = Callable[[Any, Any], Any]
_EVALUATOR = Union[Callable[[Module], float], Callable[[Module], Tuple[float, Any]]]
_TRAINER = Callable[[Module, Union[Optimizer, List[Optimizer]], _CRITERION, Union[None, _LRScheduler, List[_LRScheduler]], Optional[int], Optional[int]], None]


class LegacyEvaluator(Evaluator):
    def __init__(self, pure_model: Module, trainer: _TRAINER, optimizers: Optimizer | List[Optimizer], criterion: _CRITERION,
                 lr_schedulers: _LRScheduler | List[_LRScheduler] | None = None, dummy_input: Any | None = None, evaluator: _EVALUATOR | None = None):
        self.trainer = trainer
        self._ori_criterion = criterion
        self._criterion = self._ori_criterion
        self.dummy_input = dummy_input
        self.evaluator = evaluator

        self._optimizer_helpers, self._lr_scheduler_helpers, self._lrs_opt_map = \
            self._init_optimizer_helpers(pure_model, optimizers, lr_schedulers)

        self.model: Module | None = None
        self._optimizers: List[Optimizer] | None = None
        self._lr_schedulers: List[_LRScheduler] | None = None
        self._first_optimizer_step: Callable | None = None
        self._param_names_map: Dict[str, str] | None = None
        self._hooks: List[Hook] = []

    def _init_optimizer_helpers(self, pure_model: Module, optimizers: Optimizer | List[Optimizer], lr_schedulers: _LRScheduler | List[_LRScheduler] | None):
        traced_optimizers = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
        assert all(isinstance(optimizer, Optimizer) and is_traceable(optimizer) for optimizer in traced_optimizers)
        traced_lr_schedulers = lr_schedulers if isinstance(lr_schedulers, (list, tuple)) else [lr_schedulers] if lr_schedulers else []
        assert all(isinstance(lr_scheduler, _LRScheduler) and is_traceable(lr_scheduler) for lr_scheduler in traced_lr_schedulers)

        optimizer_helpers = [OptimizerConstructHelper.from_trace(pure_model, optimizer) for optimizer in traced_optimizers]
        lr_scheduler_helpers = [LRSchedulerConstructHelper.from_trace(lr_scheduler) for lr_scheduler in traced_lr_schedulers]
        optimizer_ids_map = {id(optimizer): i for i, optimizer in enumerate(traced_optimizers)}
        # record i-th lr_scheduler scheduling j-th optimizer lr
        lrs_opt_map = {i: optimizer_ids_map[id(lr_scheduler.optimizer)] for i, lr_scheduler in enumerate(traced_lr_schedulers)}  # type: ignore

        return optimizer_helpers, lr_scheduler_helpers, lrs_opt_map

    def bind_model(self, model: Module, param_names_map: Dict[str, str] | None = None):
        assert isinstance(model, Module)
        self.model = model
        self._param_names_map = param_names_map
        self._optimizers = [helper.call(model, param_names_map) for helper in self._optimizer_helpers]
        self._lr_schedulers = [lrs_helper.call(self._optimizers[self._lrs_opt_map[i]]) for i, lrs_helper in enumerate(self._lr_scheduler_helpers)]
        self._first_optimizer_step = self._optimizers[0].step

    def unbind_model(self):
        self.revert_loss()
        self.revert_optimizer_step()
        self.remove_all_hooks()
        self._first_optimizer_step = None
        self._lr_schedulers = None
        self._optimizers = None
        self._param_names_map = None
        self.model = None

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

    def register_hooks(self, hooks: List[Hook]):
        for hook in hooks:
            hook.register()
            self._hooks.append(hook)

    def remove_all_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def train(self, max_steps: int | None = None, max_epochs: int | None = None):
        assert self.model is not None
        assert self._optimizers is not None
        assert self._criterion is not None
        self.trainer(self.model, self._optimizers, self._criterion, self._lr_schedulers, max_steps, max_epochs)

    def finetune(self):
        self.train()

    def evaluate(self) -> float | Tuple[float, Any]:
        """
        Note that the first item of the returned value will be used as the default metric used by NNI.
        """
        assert self.model is not None
        assert self.evaluator is not None
        return self.evaluator(self.model)

    def get_dummy_input(self) -> Any:
        return self.dummy_input
