# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import types
from typing import Dict, List, Tuple, Union, Any, Callable, Optional

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
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
        A factory fucntion, input is a list buffer, output is a hook function.
        The buffer is used to store some useful information in hook.
    """

    def __init__(self, target: Module | Tensor, target_name: str, hook_factory: Callable[[List], Callable]):
        self.target = target
        self.target_name = target_name
        self.hook_factory = hook_factory
        self.buffer: Optional[List] = None
        self.handle: Optional[RemovableHandle] = None

    def _register(self, hook_func: Callable) -> RemovableHandle:
        raise NotImplementedError

    def register(self, buffer: List):
        if self.handle is not None:
            _logger.warning('%s for %s already has been registered.', self.__class__.__name__, self.target_name)
            return
        self.buffer = buffer
        self.handle = self._register(self.hook_factory(buffer))

    def remove(self):
        if self.handle is None:
            print('%s for %s has not been registered yet.', self.__class__.__name__, self.target_name)
            return
        self.handle.remove()
        self.handle = None
        self.buffer = None


class TensorHook(Hook):
    def __init__(self, target: Tensor, target_name: str, hook_factory: Callable[[List], Callable[[Tensor], Any]]):
        assert isinstance(target, Tensor)
        super().__init__(target, target_name, hook_factory)

    def _register(self, hook_func: Callable[[Tensor], Any]) -> RemovableHandle:
        return self.target.register_hook(hook_func)


class ModuleHook(Hook):
    def __init__(self, target: Module, target_name: str, hook_factory: Callable[[List], Callable[[Module, Tensor, Tensor], Any]]):
        assert isinstance(target, Module)
        super().__init__(target, target_name, hook_factory)


class ForwardHook(ModuleHook):
    def _register(self, hook_func: Callable[[Module, Tensor, Tensor], Any]):
        return self.target.register_forward_hook(hook_func)


class BackwardHook(ModuleHook):
    def _register(self, hook_func: Callable[[Module, Tensor, Tensor], Any]):
        return self.target.register_backward_hook(hook_func)


class Evaluator:
    def init_optimizer_helpers(self, pure_model: Module | pl.LightningModule):
        # Note for developer, please make sure you can get pure_model (which means the unwrapped model) from other place.
        # This function should be called before bind_model().
        raise NotImplementedError

    def bind_model(self, model: Module | pl.LightningModule, param_names_map: Optional[Dict[str, str]] = None):
        # param_names_map maps the names of the parameters in the pure_model to the names of the parameters in the binded model.
        # The format of param_names_map is {pure_model_param_name: binded_model_param_name}.
        # param_names_map is for initializing the optimizers for the binded model.
        raise NotImplementedError

    def unbind_model(self):
        # Evaluator can be reused by `unbind_model` then bind a new model by `bind_model`.
        raise NotImplementedError

    def patch_criterion(self, patch: Callable[[Tensor], Tensor]):
        raise NotImplementedError

    def revert_criterion(self):
        raise NotImplementedError

    def patch_optimizer(self, before_step_tasks: List[Callable], after_step_tasks: List[Callable]):
        # Run tasks in `before_step_tasks` before `optimizer.step()` each time.
        # Run tasks in `after_step_tasks` after `optimizer.step()` each time.
        # NOTE: we only patch these tasks to the first optimizer right now.
        raise NotImplementedError

    def revert_optimizer(self):
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
    def __init__(self, trainer: pl.Trainer, data_module: pl.LightningDataModule, dummy_input: Optional[Any] = None):
        assert isinstance(trainer, pl.Trainer)
        assert isinstance(data_module, pl.LightningDataModule)
        self.trainer = trainer
        self.data_module = data_module
        self.model: Optional[pl.LightningModule] = None

        self._dummy_input = dummy_input
        self._hooks: List[Hook] = []
        self._ori_model_attr = {}
        self._optimizer_helpers: Optional[List[OptimizerConstructHelper]] = None
        self._lr_scheduler_helpers: Optional[List[LRSchedulerConstructHelper]] = None
        self._param_names_map: Optional[Dict[str, str]] = None

    def init_optimizer_helpers(self, pure_model: pl.LightningModule):
        if self._optimizer_helpers is None:
            # FIXME: support more return type
            optimizer: Optimizer = pure_model.configure_optimizers()
            self._optimizer_helpers = [OptimizerConstructHelper.from_trace(pure_model, optimizer)]
        else:
            _logger.warning('%s already have initialized optimizer helpers.', self.__class__.__name__)

    def bind_model(self, model: pl.LightningModule, param_names_map: Optional[Dict[str, str]] = None):
        assert isinstance(model, pl.LightningModule)
        assert self._optimizer_helpers is not None
        self.model = model
        self._ori_model_attr.update({
            'training_step': model.training_step,
            'configure_optimizers': model.configure_optimizers,
            'configure_callbacks': model.configure_callbacks
        })
        self._param_names_map = param_names_map
        self._patch_configure_optimizers()

    def unbind_model(self):
        self.revert_criterion()
        self.revert_optimizer()
        self.remove_all_hooks()
        self._revert_configure_optimizers()
        self._param_names_map = None
        self._ori_model_attr.clear()
        self.model = None

    def _patch_configure_optimizers(self):
        assert self._optimizer_helpers is not None
        assert self._param_names_map is not None

        # FIXME: add scheduler
        def new_configure_optimizers(_):
            optimizers = []
            for optimizer_helper in self._optimizer_helpers:
                optimizers.append(optimizer_helper.call(self.model, self._param_names_map))
            return optimizers

        self.model.configure_optimizers = types.MethodType(new_configure_optimizers, self.model)

    def _revert_configure_optimizers(self):
        self.model.configure_optimizers = self._ori_model_attr['configure_optimizers']

    def patch_criterion(self, patch: Callable[[Tensor], Tensor]):
        old_training_step = self.model.training_step

        def patched_training_step(_, *args, **kwargs):
            output = old_training_step(*args, **kwargs)
            if isinstance(output, Tensor):
                output = patch(output)
            else:
                output['loss'] = patch(output['loss'])
            return output

        self.model.training_step = types.MethodType(patched_training_step, self.model)

    def revert_criterion(self):
        self.model.training_step = self._ori_model_attr['training_step']

    def patch_optimizer(self, before_step_tasks: List[Callable], after_step_tasks: List[Callable]):
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
            callbacks.append(OptimizerCallback())
            return callbacks

        self.model.configure_callbacks = types.MethodType(patched_configure_callbacks, self.model)

    def revert_optimizer(self):
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
        ori_max_steps, ori_max_epochs = self.trainer.fit_loop.max_steps, self.trainer.fit_loop.max_epochs
        if max_steps:
            self.trainer.fit_loop.max_steps = max_steps
        if max_epochs:
            self.trainer.fit_loop.max_epochs = max_epochs
        self.trainer.fit(self.model, self.data_module)
        self.trainer.fit_loop.max_steps, self.trainer.fit_loop.max_epochs = ori_max_steps, ori_max_epochs

    def finetune(self):
        self.train()

    def evaluate(self) -> Tuple[float, List[Dict[str, float]]]:
        """
        NNI will use metric with key `NNI_METRIC` for evaluating model, please make sure you have this key in your `Trainer.test()` returned metric dicts.
        If `Trainer.test()` returned list contains multiple dicts with key `NNI_METRIC`, NNI will take their average as the final metric.
        E.g., if `Trainer.test()` returned `[{'NNI_METRIC': 0.8, 'loss': 2.3}, {'NNI_METRIC': 0.6, 'loss': 2.4}, {'NNI_METRIC': 0.7, 'loss': 2.3}]`,
        NNI will take the final metric `(0.8 + 0.6 + 0.7) / 3 = 0.7`.
        """
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
        return next(iter(self.data_module.train_dataloader))


_CRITERION = Callable[[Any, Any], Any]
_EVALUATOR = Union[Callable[[Module], float], Callable[[Module], Tuple[float, Any]]]
_TRAINER = Callable[[Module, Union[Optimizer, List[Optimizer]], _CRITERION, Optional[int], Optional[int]], None]


class LegacyEvaluator(Evaluator):
    def __init__(self, trainer: _TRAINER, optimizer: Optimizer, criterion: _CRITERION, dummy_input: Any,
                 evaluator: _EVALUATOR):
        self.trainer = trainer
        assert isinstance(optimizer, Optimizer) and is_traceable(optimizer)
        self._traced_optimizers = [optimizer]
        self._ori_criterion = criterion
        self.dummy_input = dummy_input
        self.evaluator = evaluator
        self.model: Optional[Module] = None

        self._hooks: List[Hook] = []
        self._criterion = self._ori_criterion
        self._optimizer_helpers: Optional[List[OptimizerConstructHelper]] = None
        self._optimizers: Optional[List[Optimizer]] = None
        self._first_optimizer_step: Optional[Callable] = None
        self._param_names_map: Optional[Dict[str, str]] = None

    def init_optimizer_helpers(self, pure_model: Module):
        if self._optimizer_helpers:
            self._optimizer_helpers = [OptimizerConstructHelper.from_trace(pure_model, optimizer) for optimizer in self._traced_optimizers]
            self._traced_optimizers.clear()
        else:
            _logger.warning('%s already have initialized optimizer helper.', self.__class__.__name__)

    def bind_model(self, model: Module, param_names_map: Optional[Dict[str, str]] = None):
        assert isinstance(model, Module)
        assert self._optimizer_helpers is not None
        self.model = model
        self._param_names_map = param_names_map
        self._optimizers = [helper.call(model, param_names_map) for helper in self._optimizer_helpers]
        self._first_optimizer_step = self._optimizers[0].step

    def unbind_model(self):
        self.revert_criterion()
        self.revert_optimizer()
        self.remove_all_hooks()
        self._first_optimizer_step = None
        self._optimizers = None
        self._param_names_map = None
        self.model = None

    def patch_criterion(self, patch: Callable[[Tensor], Tensor]):
        old_criterion = self._criterion

        def patched_criterion(*args, **kwargs):
            loss = old_criterion(*args, **kwargs)
            return patch(loss)

        self._criterion = patched_criterion

    def revert_criterion(self):
        self._criterion = self._ori_criterion

    def patch_optimizer(self, before_step_tasks: List[Callable], after_step_tasks: List[Callable]):
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

    def revert_optimizer(self):
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
        optimizer = self._optimizers[0]
        self.trainer(self.model, optimizer, self._criterion, max_steps, max_epochs)

    def finetune(self):
        optimizer = self._optimizers[0]
        self.trainer(self.model, optimizer, self._criterion)

    def evaluate(self) -> float | Tuple[float, Any]:
        """
        Note that the first item of the returned value will be used as the default metric used by NNI.
        """
        return self.evaluator(self.model)

    def get_dummy_input(self) -> Any:
        return self.dummy_input
