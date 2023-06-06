# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from datetime import datetime
import logging
from pathlib import Path
import types
from typing import List, Dict, Tuple, Optional, Callable, Union
from typing_extensions import Literal

import json_tricks
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from ...base import Pruner, LayerInfo, Task, TaskResult
from ...utils import Evaluator, Hook, OptimizerConstructHelper, Scaling

_logger = logging.getLogger(__name__)


def _get_scaler(scalers: Dict[str, Dict[str, Scaling]] | None, module_name: str, target_name: str) -> Scaling | None:
    # Get scaler for the specific target in the specific module. Return None if don't find it.
    # `module_name` is not used in current nni version, will support different modules using different scalers in the future.
    if scalers:
        default_module_scalers = scalers.get('_default', {})
        default_target_scaler = default_module_scalers.get(target_name, default_module_scalers.get('_default', None))
        module_scalers = scalers.get(module_name, {})
        return module_scalers.get(target_name, module_scalers.get('_default', default_target_scaler))
    else:
        return None


class DataCollector:
    """
    An abstract class for collect the data needed by the compressor.

    Parameters
    ----------
    compressor
        The compressor binded with this DataCollector.
    """

    def __init__(self, compressor: Pruner):
        self.compressor = compressor

    def reset(self, *args, **kwargs):
        """
        Reset the `DataCollector`.
        """
        raise NotImplementedError()

    def collect(self) -> Dict:
        """
        Collect the compressor needed data, i.e., module weight, the output of activation function.

        Returns
        -------
        Dict
            Usually has format like {module_name: tensor_type_data}.
        """
        raise NotImplementedError()


# TODO: remove in nni v3.0.
COLLECTOR_TYPE = Union[Callable[[List, Tensor], Callable[[Tensor], None]], Callable[[List], Callable[[Module, Tensor, Tensor], None]]]

class HookCollectorInfo:
    def __init__(self, targets: Union[Dict[str, Tensor], List[LayerInfo]], hook_type: str,
                 collector: COLLECTOR_TYPE):
        """
        This class used to aggregate the information of what kind of hook is placed on which layers.

        Parameters
        ----------
        targets
            List of LayerInfo or Dict of {layer_name: weight_tensor}, the hook targets.
        hook_type
            'forward' or 'backward'.
        collector
            A hook function generator, the input is a buffer (empty list) or a buffer (empty list) and tensor,
            the output is a hook function. The buffer is used to store the data wanted to hook.
        """
        self.targets = targets
        self.hook_type = hook_type
        self.collector = collector


# TODO: remove in nni v3.0.
class TrainerBasedDataCollector(DataCollector):
    """
    This class includes some trainer based util functions, i.e., patch optimizer or criterion, add hooks.
    """

    def __init__(self, compressor: Pruner, trainer: Callable[[Module, Optimizer, Callable], None],
                 optimizer_helper: OptimizerConstructHelper, criterion: Callable[[Tensor, Tensor], Tensor], training_epochs: int,
                 opt_before_tasks: List = [], opt_after_tasks: List = [], collector_infos: List[HookCollectorInfo] = [],
                 criterion_patch: Optional[Callable[[Callable], Callable]] = None):
        """
        Parameters
        ----------
        compressor
            The compressor binded with this DataCollector.
        trainer
            A callable function used to train model or just inference. Take model, optimizer, criterion as input.
            The model will be trained or inferenced `training_epochs` epochs.

            Example::

                def trainer(model: Module, optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor]):
                    training = model.training
                    model.train(mode=True)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        # If you don't want to update the model, you can skip `optimizer.step()`, and set train mode False.
                        optimizer.step()
                    model.train(mode=training)
        optimizer
            The optimizer instance used in trainer. Note that this optimizer might be patched during collect data,
            so do not use this optimizer in other places.
        criterion
            The criterion function used in trainer. Take model output and target value as input, and return the loss.
        training_epochs
            The total number of calling trainer.
        opt_before_tasks
            A list of function that will be called one by one before origin `optimizer.step()`.
            Note that these functions will be patched into `optimizer.step()`.
        opt_after_tasks
            A list of function that will be called one by one after origin `optimizer.step()`.
            Note that these functions will be patched into `optimizer.step()`.
        collector_infos
            A list of `HookCollectorInfo` instance. And the hooks will be registered in `__init__`.
        criterion_patch
            A callable function used to patch the criterion. Take a criterion function as input and return a new one.

            Example::

                def criterion_patch(criterion: Callable[[Tensor, Tensor], Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
                    weight = ...
                    def patched_criterion(output, target):
                        return criterion(output, target) + torch.norm(weight)
                    return patched_criterion
        """
        super().__init__(compressor)
        self.trainer = trainer
        self.training_epochs = training_epochs
        self.optimizer_helper = optimizer_helper
        self._origin_criterion = criterion
        self._opt_before_tasks = opt_before_tasks
        self._opt_after_tasks = opt_after_tasks

        self._criterion_patch = criterion_patch

        self.reset(collector_infos)

    def reset(self, collector_infos: List[HookCollectorInfo] = []):
        # refresh optimizer and criterion
        self._reset_optimizer()

        if self._criterion_patch is not None:
            self.criterion = self._criterion_patch(self._origin_criterion)
        else:
            self.criterion = self._origin_criterion

        # patch optimizer
        self._patch_optimizer()

        # hook
        self._remove_all_hook()
        self._hook_id = 0
        self._hook_handles = {}
        self._hook_buffer = {}

        self._collector_infos = collector_infos
        self._add_all_hook()

    def _reset_optimizer(self):
        parameter_name_map = self.compressor.get_origin2wrapped_parameter_name_map()
        assert self.compressor.bound_model is not None
        self.optimizer = self.optimizer_helper.call(self.compressor.bound_model, parameter_name_map)

    def _patch_optimizer(self):
        def patch_step(old_step):
            def new_step(_, *args, **kwargs):
                for task in self._opt_before_tasks:
                    task()
                # call origin optimizer step method
                output = old_step(*args, **kwargs)
                for task in self._opt_after_tasks:
                    task()
                return output
            return new_step
        if self.optimizer is not None:
            self.optimizer.step = types.MethodType(patch_step(self.optimizer.step), self.optimizer)

    def _add_hook(self, collector_info: HookCollectorInfo) -> int:
        self._hook_id += 1
        self._hook_handles[self._hook_id] = {}
        self._hook_buffer[self._hook_id] = {}

        if collector_info.hook_type == 'forward':
            self._add_forward_hook(self._hook_id, collector_info.targets, collector_info.collector)  # type: ignore
        elif collector_info.hook_type == 'backward':
            self._add_backward_hook(self._hook_id, collector_info.targets, collector_info.collector)  # type: ignore
        elif collector_info.hook_type == 'tensor':
            self._add_tensor_hook(self._hook_id, collector_info.targets, collector_info.collector)  # type: ignore
        else:
            _logger.warning('Skip unsupported hook type: %s', collector_info.hook_type)

        return self._hook_id

    def _add_forward_hook(self, hook_id: int, layers: List[LayerInfo],
                          collector: Callable[[List], Callable[[Module, Tensor, Tensor], None]]):
        assert all(isinstance(layer_info, LayerInfo) for layer_info in layers)
        for layer in layers:
            self._hook_buffer[hook_id][layer.name] = []
            handle = layer.module.register_forward_hook(collector(self._hook_buffer[hook_id][layer.name]))
            self._hook_handles[hook_id][layer.name] = handle

    def _add_backward_hook(self, hook_id: int, layers: List[LayerInfo],
                           collector: Callable[[List], Callable[[Module, Tensor, Tensor], None]]):
        assert all(isinstance(layer_info, LayerInfo) for layer_info in layers)
        for layer in layers:
            self._hook_buffer[hook_id][layer.name] = []
            handle = layer.module.register_backward_hook(collector(self._hook_buffer[hook_id][layer.name]))  # type: ignore
            self._hook_handles[hook_id][layer.name] = handle

    def _add_tensor_hook(self, hook_id: int, tensors: Dict[str, Tensor],
                         collector: Callable[[List, Tensor], Callable[[Tensor], None]]):
        assert all(isinstance(tensor, Tensor) for _, tensor in tensors.items())
        for layer_name, tensor in tensors.items():
            self._hook_buffer[hook_id][layer_name] = []
            handle = tensor.register_hook(collector(self._hook_buffer[hook_id][layer_name], tensor))
            self._hook_handles[hook_id][layer_name] = handle

    def _remove_hook(self, hook_id: int):
        if hook_id not in self._hook_handles:
            raise ValueError("%s is not a valid collector id" % str(hook_id))
        for handle in self._hook_handles[hook_id].values():
            handle.remove()
        del self._hook_handles[hook_id]

    def _add_all_hook(self):
        for collector_info in self._collector_infos:
            self._add_hook(collector_info)

    def _remove_all_hook(self):
        if hasattr(self, '_hook_handles'):
            for hook_id in list(self._hook_handles.keys()):
                self._remove_hook(hook_id)


class EvaluatorBasedDataCollector(DataCollector):
    """
    This data collector is the base class for the data collectors that want to use ``Evaluator`` to train or inference.
    Three main usages are supported in this data collector:

    1. Doing something before ``optimzer.step()`` and after ``optimzer.step()``. ``before_opt_step_tasks`` is a list of task functions
       that will execute before ``optimzer.step()``. ``after_opt_step_tasks`` is a list of task functions that will execute after
       ``optimzer.step()``. All the task functions in the list should not have input arguments, function return value is allowed,
       but ``Evaluator`` will not catch it.
    2. Patch or modify the training loss. ``loss_patch`` is a function with input is the original loss and the output is the modified loss.
    3. Add hooks on ``torch.nn.Module`` or ``Parameter`` or ``Buffer``. Three kinds of hook are supported, ``TensorHook``, ``ForwardHook``
       and ``BackwardHook``. For initializing a ``Hook``, a hook function factory is needed, the factory function's input is an empty list,
       and the output is a hook function defined by Pytorch.
       Please refer `register_hook <https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html>`_,
       `register_forward_hook <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook>`_,
       `register_backward_hook <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_backward_hook>`_.
    """

    def __init__(self, compressor: Pruner, evaluator: Evaluator, before_opt_step_tasks: List[Callable] | None = None,
                 after_opt_step_tasks: List[Callable] | None = None, loss_patch: Callable[[Tensor], Tensor] | None = None,
                 hooks: Dict[str, Dict[str, Hook]] | None = None, max_steps: int | None = None, max_epochs: int | None = None):
        super().__init__(compressor)
        self.evaluator = evaluator
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.reset(before_opt_step_tasks, after_opt_step_tasks, loss_patch, hooks)

    def reset(self, before_opt_step_tasks: List[Callable] | None = None, after_opt_step_tasks: List[Callable] | None = None,
              loss_patch: Callable[[Tensor], Tensor] | None = None, hooks: Dict[str, Dict[str, Hook]] | None = None):
        if before_opt_step_tasks or after_opt_step_tasks:
            before_opt_step_tasks = before_opt_step_tasks if before_opt_step_tasks else []
            after_opt_step_tasks = after_opt_step_tasks if after_opt_step_tasks else []
            self.evaluator.patch_optimizer_step(before_opt_step_tasks, after_opt_step_tasks)
        if loss_patch:
            self.evaluator.patch_loss(loss_patch)
        if hooks:
            self._hooks = hooks
            hook_list = [hook for _ in hooks.values() for hook in _.values()]
            self.evaluator.register_hooks(hook_list)


class MetricsCalculator:
    """
    An abstract class for calculate a kind of metrics of the given data.

    Parameters
    ----------
    scalers
        Scaler is used to scale the metrics' size. It scaling metric to the same size as the shrinked mask in the sparsity allocator.
        If you want to use different scalers for different pruning targets in different modules,
        please use a dict `{module_name: {target_name: scaler}}`.
        If allocator meets an unspecified module name, it will try to use `scalers['_default'][target_name]` to scale its mask.
        If allocator meets an unspecified target name, it will try to use `scalers[module_name]['_default']` to scale its mask.
        Passing in a scaler instead of a `dict` of scalers will be treated as passed in `{'_default': {'_default': scalers}}`.
        Passing in `None` means no need to scale.
    """

    def __init__(self, scalers: Dict[str, Dict[str, Scaling]] | Scaling | None = None):
        self.scalers: Dict[str, Dict[str, Scaling]] | None = scalers \
            if isinstance(scalers, (dict, type(None))) else {'_default': {'_default': scalers}}  # type: ignore

    def _get_scaler(self, module_name: str, target_name: str) -> Scaling:
        scaler = _get_scaler(self.scalers, module_name, target_name)
        return scaler if scaler else Scaling([1])

    def calculate_metrics(self, data: Dict) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        data
            A dict handle the data used to calculate metrics. Usually has format like {module_name: tensor_type_data}.

        Returns
        -------
        Dict[str, Tensor]
            The key is the layer_name, value is the metric.
            Note that the metric has the same size with the data size on `dim`.
        """
        raise NotImplementedError()


class SparsityAllocator:
    """
    A base class for allocating mask based on metrics.

    Parameters
    ----------
    pruner
        The pruner that binded with this `SparsityAllocator`.
    scalers
        Scaler is used to scale the masks' size. It shrinks the mask of the same size as the pruning target to the same size as the metric,
        or expands the mask of the same size as the metric to the same size as the pruning target.
        If you want to use different scalers for different pruning targets in different modules,
        please use a dict `{module_name: {target_name: scaler}}`.
        If allocator meets an unspecified module name, it will try to use `scalers['_default'][target_name]` to scale its mask.
        If allocator meets an unspecified target name, it will try to use `scalers[module_name]['_default']` to scale its mask.
        Passing in a scaler instead of a `dict` of scalers will be treated as passed in `{'_default': {'_default': scalers}}`.
        Passing in `None` means no need to scale.
    continuous_mask
        If set True, the part that has been masked will be masked first.
        If set False, the part that has been masked may be unmasked due to the increase of its corresponding metric.
    """

    def __init__(self, pruner: Pruner, scalers: Dict[str, Dict[str, Scaling]] | Scaling | None = None, continuous_mask: bool = True):
        self.pruner = pruner
        self.scalers: Dict[str, Dict[str, Scaling]] | None = scalers \
            if isinstance(scalers, (dict, type(None))) else {'_default': {'_default': scalers}}  # type: ignore
        self.continuous_mask = continuous_mask

    def _get_scaler(self, module_name: str, target_name: str) -> Scaling | None:
        return _get_scaler(self.scalers, module_name, target_name)

    def _expand_mask(self, module_name: str, target_name: str, mask: Tensor) -> Tensor:
        # Expand the shrinked mask to the pruning target size.
        scaler = self._get_scaler(module_name=module_name, target_name=target_name)
        if scaler:
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            return scaler.expand(mask, getattr(wrapper, f'{target_name}_mask').shape)
        else:
            return mask.clone()

    def _shrink_mask(self, module_name: str, target_name: str, mask: Tensor) -> Tensor:
        # Shrink the mask by scaler, shrinked mask usually has the same size with metric.
        scaler = self._get_scaler(module_name=module_name, target_name=target_name)
        if scaler:
            mask = (scaler.shrink(mask) != 0).type_as(mask)
        return mask

    def _mask_metric(self, metrics: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        # Set the already masked part in the metric to the minimum value.
        target_name = 'weight'
        for module_name, targets_metric in metrics.items():
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            old_target_mask: Tensor = getattr(wrapper, f'{target_name}_mask')
            shrinked_target_mask = self._shrink_mask(module_name, target_name, old_target_mask)
            # make sure the masked position has the minimum metric
            targets_metric[target_name] = targets_metric[target_name].to(shrinked_target_mask.device)
            min_value = targets_metric[target_name].min() - 1
            targets_metric[target_name] = torch.where(shrinked_target_mask != 0, targets_metric[target_name], min_value)
        return metrics

    def _continuous_mask(self, new_masks: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        # Set the already masked part to zero in the new_masks.
        target_name = 'weight'
        for module_name, target_mask in new_masks.items():
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            old_target_mask: Tensor | None = getattr(wrapper, f'{target_name}_mask', None)
            if old_target_mask is not None:
                new_masks[module_name][target_name] = torch.min(target_mask[target_name],
                                                                old_target_mask.to(target_mask[target_name].device))
        return new_masks

    def common_target_masks_generation(self, metrics: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        """
        Generate masks for metrics-dependent targets.

        Parameters
        ----------
        metrics
            The format is {module_name: {target_name: target_metric}}.
            The metric of usually has the same size with shrinked mask.

        Return
        ------
        Dict[str, Dict[str, Tensor]]
            The format is {module_name: {target_name: mask}}.
            Return the masks of the same size as its target.
        """
        raise NotImplementedError()

    def special_target_masks_generation(self, masks: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        """
        Some pruning targets' mask generation depends on other targets, i.e., bias mask depends on weight mask.
        This function is used to generate these masks, and it be called at the end of `generate_sparsity`.

        Parameters
        ----------
        masks
            The format is {module_name: {target_name: mask}}.
            It is usually the return value of `common_target_masks_generation`.
        """
        for module_name, module_masks in masks.items():
            # generate bias mask, this may move to wrapper in the future
            weight_mask = module_masks.get('weight', None)
            wrapper = self.pruner.get_modules_wrapper()[module_name]
            old_bias_mask = getattr(wrapper, 'bias_mask', None)
            if weight_mask is not None and old_bias_mask is not None and weight_mask.shape[0] == old_bias_mask.shape[0]:
                # keep dim 0 and reduce all other dims by sum
                reduce_dims = [reduce_dim for reduce_dim in range(1, len(weight_mask.shape))]
                # count unmasked number of values on dim 0 (output channel) of weight
                unmasked_num_on_dim0 = weight_mask.sum(reduce_dims) if reduce_dims else weight_mask
                module_masks['bias'] = (unmasked_num_on_dim0 != 0).type_as(weight_mask)
        return masks

    def generate_sparsity(self, metrics: Dict) -> Dict[str, Dict[str, Tensor]]:
        """
        The main function of `SparsityAllocator`, generate a set of masks based on the given metrics.

        Parameters
        ----------
        metrics
            A metric dict with format {module_name: weight_metric}

        Returns
        -------
        Dict[str, Dict[str, Tensor]]
            The masks format is {module_name: {target_name: mask}}.
        """
        if self.continuous_mask:
            metrics = self._mask_metric(metrics)
        masks = self.common_target_masks_generation(metrics)
        masks = self.special_target_masks_generation(masks)
        if self.continuous_mask:
            masks = self._continuous_mask(masks)
        return masks


class TaskGenerator:
    """
    This class used to generate config list for pruner in each iteration.

    Parameters
    ----------
    origin_model
        The origin unwrapped pytorch model to be pruned.
    origin_masks
        The pre masks on the origin model. This mask maybe user-defined or maybe generate by previous pruning.
    origin_config_list
        The origin config list provided by the user. Note that this config_list is directly config the origin model.
        This means the sparsity provided by the origin_masks should also be recorded in the origin_config_list.
    log_dir
        The log directory use to saving the task generator log.
    keep_intermediate_result
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    best_result_mode
        The way to decide which one is the best result. Three modes are supported.
        If the task results don't contain scores (task_result.score is None), it will fall back to ``latest``.

        1. latest: The newest received result is the best result.
        2. maximize: The one with largest task result score is the best result.
        3. minimize: The one with smallest task result score is the best result.
    """

    def __init__(self, origin_model: Optional[Module], origin_masks: Optional[Dict[str, Dict[str, Tensor]]] = {},
                 origin_config_list: Optional[List[Dict]] = [], log_dir: Union[str, Path] = '.', keep_intermediate_result: bool = False,
                 best_result_mode: Literal['latest', 'maximize', 'minimize'] = 'maximize'):
        self._log_dir = log_dir
        self._keep_intermediate_result = keep_intermediate_result
        assert best_result_mode in ['latest', 'maximize', 'minimize'], f'Unsupported best_result_mode value: {best_result_mode}'
        self._best_result_mode = best_result_mode

        if origin_model is not None and origin_config_list is not None and origin_masks is not None:
            self.reset(origin_model, origin_config_list, origin_masks)

    def reset(self, model: Module, config_list: List[Dict] = [], masks: Dict[str, Dict[str, Tensor]] = {}):
        assert isinstance(model, Module), 'Only support pytorch module.'

        self._log_dir_root = Path(self._log_dir, datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')).absolute()
        self._log_dir_root.mkdir(parents=True, exist_ok=True)

        self._intermediate_result_dir = Path(self._log_dir_root, 'intermediate_result')
        self._intermediate_result_dir.mkdir(parents=True, exist_ok=True)

        # save origin data in {log_dir}/origin
        self._origin_model_path = Path(self._log_dir_root, 'origin', 'model.pth')
        self._origin_masks_path = Path(self._log_dir_root, 'origin', 'masks.pth')
        self._origin_config_list_path = Path(self._log_dir_root, 'origin', 'config_list.json')
        self._save_data('origin', model, masks, config_list)

        self._task_id_candidate = 0
        self._tasks: Dict[Union[int, str], Task] = {}
        self._pending_tasks: List[Task] = self.init_pending_tasks()

        self._best_score = None
        self._best_task_id = None

        # dump self._tasks into {log_dir}/.tasks
        self._dump_tasks_info()

    def _dump_tasks_info(self):
        tasks = {task_id: task.to_dict() for task_id, task in self._tasks.items()}
        with Path(self._log_dir_root, '.tasks').open('w') as f:
            json_tricks.dump(tasks, f, indent=4)

    def _save_data(self, folder_name: str, model: Module, masks: Dict[str, Dict[str, Tensor]], config_list: List[Dict]):
        Path(self._log_dir_root, folder_name).mkdir(parents=True, exist_ok=True)
        torch.save(model, Path(self._log_dir_root, folder_name, 'model.pth'))
        torch.save(masks, Path(self._log_dir_root, folder_name, 'masks.pth'))
        with Path(self._log_dir_root, folder_name, 'config_list.json').open('w') as f:
            json_tricks.dump(config_list, f, indent=4)

    def update_best_result(self, task_result: TaskResult):
        save_as_best_result = False
        task = self._tasks[task_result.task_id]
        task.score = task_result.score

        if self._best_result_mode == 'latest':
            self._best_task_id, save_as_best_result = task_result.task_id, True

        if self._best_result_mode == 'maximize':
            if self._best_score is None or (task.score is not None and task.score > self._best_score):
                self._best_score = task.score
                self._best_task_id, save_as_best_result = task_result.task_id, True

        if self._best_result_mode == 'minimize':
            if self._best_score is None or (task.score is not None and task.score < self._best_score):
                self._best_score = task.score
                self._best_task_id, save_as_best_result = task_result.task_id, True

        if save_as_best_result:
            with Path(task.config_list_path).open('r') as fr:
                best_config_list = json_tricks.load(fr)
            self._save_data('best_result', task_result.compact_model, task_result.compact_model_masks, best_config_list)

    def init_pending_tasks(self) -> List[Task]:
        raise NotImplementedError()

    def generate_tasks(self, task_result: TaskResult) -> List[Task]:
        raise NotImplementedError()

    def receive_task_result(self, task_result: TaskResult):
        """
        Parameters
        ----------
        task_result
            The result of the task.
        """
        task_id = task_result.task_id
        assert task_id in self._tasks, 'Task {} does not exist.'.format(task_id)
        self.update_best_result(task_result)

        self._tasks[task_id].status = 'Finished'
        self._dump_tasks_info()

        self._pending_tasks.extend(self.generate_tasks(task_result))
        self._dump_tasks_info()

        if not self._keep_intermediate_result:
            self._tasks[task_id].clean_up()

    def next(self) -> Optional[Task]:
        """
        Returns
        -------
        Optional[Task]
            Return the next task from pending tasks.
        """
        if len(self._pending_tasks) == 0:
            return None
        else:
            task = self._pending_tasks.pop(0)
            task.status = 'Running'
            self._dump_tasks_info()
            return task

    def get_best_result(self) -> Optional[Tuple[Union[int, str], Module, Dict[str, Dict[str, Tensor]], Optional[float], List[Dict]]]:
        """
        Returns
        -------
        Optional[Tuple[int, Module, Dict[str, Dict[str, Tensor]], float, List[Dict]]]
            If self._best_task_id is not None,
            return best task id, best compact model, masks on the compact model, score, config list used in this task.
        """
        if self._best_task_id is not None:
            compact_model = torch.load(Path(self._log_dir_root, 'best_result', 'model.pth'))
            compact_model_masks = torch.load(Path(self._log_dir_root, 'best_result', 'masks.pth'))
            with Path(self._log_dir_root, 'best_result', 'config_list.json').open('r') as f:
                config_list = json_tricks.load(f)
            return self._best_task_id, compact_model, compact_model_masks, self._best_score, config_list
        return None
