# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import types
from typing import List, Dict, Tuple, Optional, Callable, Union

import json_tricks
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from nni.algorithms.compression.v2.pytorch.base import Compressor, LayerInfo

_logger = logging.getLogger(__name__)

__all__ = ['DataCollector', 'TrainerBasedDataCollector', 'HookCollectorInfo', 'MetricsCalculator', 'SparsityAllocator']


class DataCollector:
    """
    An abstract class for collect the data needed by the compressor.
    """

    def __init__(self, compressor: Compressor):
        """
        Parameters
        ----------
        compressor
            The compressor binded with this DataCollector.
        """
        self.compressor = compressor

    def reset(self):
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


class HookCollectorInfo:
    def __init__(self, targets: Union[Dict[str, Tensor], List[LayerInfo]], hook_type: str,
                 collector: Union[Callable[[List, Tensor], Callable[[Tensor], None]], Callable[[List], Callable[[Module, Tensor, Tensor], None]]]):
        """
        This class used to aggregate the information of what kind of hook is placed on which layers.

        Parameters
        ----------
        targets
            List of LayerInfo or Dict of {layer_name: weight_tensor}, the hook targets.
        hook_type
            'forward' or 'backward'.
        collector
            A hook function generator, the input is a buffer (empty list) or a buffer (empty list) and tensor, the output is a hook function.
            The buffer is used to store the data wanted to hook.
        """
        self.targets = targets
        self.hook_type = hook_type
        self.collector = collector


class TrainerBasedDataCollector(DataCollector):
    """
    This class includes some trainer based util functions, i.e., patch optimizer or criterion, add hooks.
    """

    def __init__(self, compressor: Compressor, trainer: Callable[[Module, Optimizer, Callable], None], optimizer: Optimizer,
                 criterion: Callable[[Tensor, Tensor], Tensor], training_epochs: int,
                 opt_before_tasks: List = [], opt_after_tasks: List = [],
                 collector_infos: List[HookCollectorInfo] = [], criterion_patch: Callable[[Callable], Callable] = None):
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
        self._origin_optimizer = optimizer
        self._origin_criterion = criterion
        self._opt_before_tasks = opt_before_tasks
        self._opt_after_tasks = opt_after_tasks

        self._collector_infos = collector_infos

        self._criterion_patch = criterion_patch

        self.reset()

    def reset(self):
        # refresh optimizer and criterion
        self.compressor._unwrap_model()
        if self._origin_optimizer is not None:
            optimizer_cls = self._origin_optimizer.__class__
            if optimizer_cls.__name__ == 'SGD':
                self.optimizer = optimizer_cls(self.compressor.bound_model.parameters(), lr=0.001)
            else:
                self.optimizer = optimizer_cls(self.compressor.bound_model.parameters())
            self.optimizer.load_state_dict(self._origin_optimizer.state_dict())
        else:
            self.optimizer = None

        if self._criterion_patch is not None:
            self.criterion = self._criterion_patch(self._origin_criterion)
        else:
            self.criterion = self._origin_criterion
        self.compressor._wrap_model()

        # patch optimizer
        self._patch_optimizer()

        # hook
        self._remove_all_hook()
        self._hook_id = 0
        self._hook_handles = {}
        self._hook_buffer = {}
        self._add_all_hook()

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
            self._add_forward_hook(self._hook_id, collector_info.targets, collector_info.collector)
        elif collector_info.hook_type == 'backward':
            self._add_backward_hook(self._hook_id, collector_info.targets, collector_info.collector)
        elif collector_info.hook_type == 'tensor':
            self._add_tensor_hook(self._hook_id, collector_info.targets, collector_info.collector)
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
            handle = layer.module.register_backward_hook(collector(self._hook_buffer[hook_id][layer.name]))
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
        for handle in self._hook_handles[hook_id]:
            handle.remove()
        del self._hook_handles[hook_id]

    def _add_all_hook(self):
        for collector_info in self._collector_infos:
            self._add_hook(collector_info)

    def _remove_all_hook(self):
        if hasattr(self, '_hook_handles'):
            for hook_id in list(self._hook_handles.keys()):
                self._remove_hook(hook_id)


class MetricsCalculator:
    """
    An abstract class for calculate a kind of metrics of the given data.
    """
    def __init__(self, dim: Optional[Union[int, List[int]]] = None,
                 block_sparse_size: Optional[Union[int, List[int]]] = None):
        """
        Parameters
        ----------
        dim
            The dimensions that corresponding to the under pruning weight dimensions in collected data.
            None means one-to-one correspondence between pruned dimensions and data, which equal to set `dim` as all data dimensions.
            Only these `dim` will be kept and other dimensions of the data will be reduced.

            Example:

            If you want to prune the Conv2d weight in filter level, and the weight size is (32, 16, 3, 3) [out-channel, in-channel, kernal-size-1, kernal-size-2].
            Then the under pruning dimensions is [0], which means you want to prune the filter or out-channel.

                Case 1: Directly collect the conv module weight as data to calculate the metric.
                Then the data has size (32, 16, 3, 3).
                Mention that the dimension 0 of the data is corresponding to the under pruning weight dimension 0.
                So in this case, `dim=0` will set in `__init__`.

                Case 2: Use the output of the conv module as data to calculate the metric.
                Then the data has size (batch_num, 32, feature_map_size_1, feature_map_size_2).
                Mention that the dimension 1 of the data is corresponding to the under pruning weight dimension 0.
                So in this case, `dim=1` will set in `__init__`.

            In both of these two case, the metric of this module has size (32,).
        block_sparse_size
            This used to describe the block size a metric value represented. By default, None means the block size is ones(len(dim)).
            Make sure len(dim) == len(block_sparse_size), and the block_sparse_size dimension position is corresponding to dim.

            Example:

            The under pruning weight size is (768, 768), and you want to apply a block sparse on dim=[0] with block size [64, 768],
            then you can set block_sparse_size=[64]. The final metric size is (12,).
        """
        self.dim = dim if not isinstance(dim, int) else [dim]
        self.block_sparse_size = block_sparse_size if not isinstance(block_sparse_size, int) else [block_sparse_size]
        if self.block_sparse_size is not None:
            assert all(i >= 1 for i in self.block_sparse_size)
        elif self.dim is not None:
            self.block_sparse_size = [1] * len(self.dim)
        if self.dim is not None:
            assert all(i >= 0 for i in self.dim)
            self.dim, self.block_sparse_size = (list(t) for t in zip(*sorted(zip(self.dim, self.block_sparse_size))))

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
    An abstract class for allocate mask based on metrics.
    """

    def __init__(self, pruner: Compressor, dim: Optional[Union[int, List[int]]] = None,
                 block_sparse_size: Optional[Union[int, List[int]]] = None):
        """
        Parameters
        ----------
        pruner
            The pruner that binded with this `SparsityAllocator`.
        dim
            The under pruning weight dimensions, which metric size should equal to the under pruning weight size on these dimensions.
            None means one-to-one correspondence between pruned dimensions and metric, which equal to set `dim` as all under pruning weight dimensions.
            The mask will expand to the weight size depend on `dim`.

            Example:

            The under pruning weight has size (2, 3, 4), and `dim=1` means the under pruning weight dimension is 1.
            Then the metric should have a size (3,), i.e., `metric=[0.9, 0.1, 0.8]`.
            Assuming by some kind of `SparsityAllocator` get the mask on weight dimension 1 `mask=[1, 0, 1]`,
            then the dimension mask will expand to the final mask `[[[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]]`.
        block_sparse_size
            This used to describe the block size a metric value represented. By default, None means the block size is ones(len(dim)).
            Make sure len(dim) == len(block_sparse_size), and the block_sparse_size dimension position is corresponding to dim.

            Example:

            The metric size is (12,), and block_sparse_size=[64], then the mask will expand to (768,) at first before expand with `dim`.
        """
        self.pruner = pruner
        self.dim = dim if not isinstance(dim, int) else [dim]
        self.block_sparse_size = block_sparse_size if not isinstance(block_sparse_size, int) else [block_sparse_size]
        if self.block_sparse_size is not None:
            assert all(i >= 1 for i in self.block_sparse_size)
        elif self.dim is not None:
            self.block_sparse_size = [1] * len(self.dim)
        if self.dim is not None:
            assert all(i >= 0 for i in self.dim)
            self.dim, self.block_sparse_size = (list(t) for t in zip(*sorted(zip(self.dim, self.block_sparse_size))))

    def generate_sparsity(self, metrics: Dict) -> Dict[str, Dict[str, Tensor]]:
        """
        Parameters
        ----------
        metrics
            A metric dict. The key is the name of layer, the value is its metric.
        """
        raise NotImplementedError()

    def _expand_mask(self, name: str, mask: Tensor) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        name
            The masked module name.
        mask
            The reduced mask with `self.dim` and `self.block_sparse_size`.

        Returns
        -------
        Dict[str, Tensor]
            The key is `weight_mask` or `bias_mask`, value is the final mask.
        """
        weight_mask = mask.clone()

        if self.block_sparse_size is not None:
            # expend mask with block_sparse_size
            expand_size = list(weight_mask.size())
            reshape_size = list(weight_mask.size())
            for i, block_width in reversed(list(enumerate(self.block_sparse_size))):
                weight_mask = weight_mask.unsqueeze(i + 1)
                expand_size.insert(i + 1, block_width)
                reshape_size[i] *= block_width
            weight_mask = weight_mask.expand(expand_size).reshape(reshape_size)

        wrapper = self.pruner.get_modules_wrapper()[name]
        weight_size = wrapper.module.weight.data.size()

        if self.dim is None:
            assert weight_mask.size() == weight_size
            expand_mask = {'weight_mask': weight_mask}
        else:
            # expand mask to weight size with dim
            assert len(weight_mask.size()) == len(self.dim)
            assert all(weight_size[j] == weight_mask.size(i) for i, j in enumerate(self.dim))

            idxs = list(range(len(weight_size)))
            [idxs.pop(i) for i in reversed(self.dim)]
            for i in idxs:
                weight_mask = weight_mask.unsqueeze(i)
            expand_mask = {'weight_mask': weight_mask.expand(weight_size).clone()}
            # NOTE: assume we only mask output, so the mask and bias have a one-to-one correspondence.
            # If we support more kind of masks, this place need refactor.
            if wrapper.bias_mask is not None and weight_mask.size() == wrapper.bias_mask.size():
                expand_mask['bias_mask'] = weight_mask.clone()
        return expand_mask

    def _compress_mask(self, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        name
            The masked module name.
        mask
            The entire mask has the same size with weight.

        Returns
        -------
        Tensor
            Reduce the mask with `self.dim` and `self.block_sparse_size`.
        """
        if self.dim is None or len(mask.size()) == 1:
            mask = mask.clone()
        else:
            mask_dim = list(range(len(mask.size())))
            for dim in self.dim:
                mask_dim.remove(dim)
            mask = torch.sum(mask, dim=mask_dim)

        if self.block_sparse_size is not None:
            # operation like pooling
            lower_case_letters = 'abcdefghijklmnopqrstuvwxyz'
            ein_expression = ''
            for i, step in enumerate(self.block_sparse_size):
                mask = mask.unfold(i, step, step)
                ein_expression += lower_case_letters[i]
            ein_expression = '...{},{}'.format(ein_expression, ein_expression)
            mask = torch.einsum(ein_expression, mask, torch.ones(self.block_sparse_size))

        return (mask != 0).type_as(mask)


CONFIG_LIST_NAME = 'config_list.json'
MODEL_NAME = 'pruned_model.pth'
MASKS_NAME = 'masks.pth'

PRE_TASK_ID = 'preTaskId'
SCORE = 'score'
LOG_DIR = 'logDir'
STATUS = 'status'


@dataclass
class Task:
    """
    Task saves the related information about the task.
    """
    task_id: int
    pre_task_id: Optional[int]
    config_list: List[Dict]
    log_dir: Path
    score: Optional[float] = None
    status: dict = field(default_factory=dict)

    def __init__(self, task_id, pre_task_id, config_list, log_dir, score=None, status=None):
        self.task_id = task_id
        self.pre_task_id = pre_task_id
        self.config_list = config_list
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.score = score
        self.status = status


class TaskGenerator:
    """
    This class used to generate config list for pruner in each iteration.
    """
    def __init__(self, origin_model: Module, origin_config_list: List[Dict] = [],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = '.', save_model: bool = True):
        assert isinstance(origin_model, Module), 'Only support pytorch module.'

        self.log_dir_root = Path(log_dir)
        self.log_dir_root.mkdir(parents=True, exist_ok=True)

        # init tasks info file, json format {TASK_ID: {PRE_TASK_ID: xxx, SCORE: xxx, LOG_DIR: xxx}}
        self.tasks_info_file = Path(self.log_dir_root, '.tasks')
        with self.tasks_info_file.open(mode='w') as f:
            json_tricks.dump({}, f)

        self.tasks_map: Dict[int, Task] = {}
        self.pending_tasks: List[Task] = []
        self.task_id_candidate = 0

        self.best_score = None
        self.best_task = None

        self.origin_task_id = None

        self._init_origin_task(origin_model, origin_config_list, origin_masks)

    def _init_origin_task(self, origin_model: Module, origin_config_list: Optional[List[Dict]] = None,
                          origin_masks: Optional[Dict[str, Dict[str, Tensor]]] = None):
        task_id = self.task_id_candidate
        task_log_dir = Path(self.log_dir_root, str(task_id))

        origin_task = Task(task_id, None, deepcopy(origin_config_list), task_log_dir)
        self.tasks_map[task_id] = origin_task
        self.origin_task_id = task_id

        self.task_id_candidate += 1

        self.receive_task_result(task_id, origin_model, origin_masks)

    def receive_task_result(self, task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]],
                            score: Optional[float] = None):
        """
        Receive the compressed model, masks and score then save the task result.
        Usually generate new task and put it into `self.pending_tasks` in this function.
        Parameters
        ----------
        task_id
            The id of the task registered in `self.tasks_map`.
        pruned_model
            The pruned model in the last iteration. It might be a sparsify model or a speed-up model.
        masks
            If masks is empty, the pruned model is a compact model after speed up.
            If masks is not None, the pruned model is a sparsify model without speed up.
        score
            The score of the model, higher score means better performance.
        """
        assert task_id in self.tasks_map, 'Task {} does not exist.'.format(task_id)
        task = self.tasks_map[task_id]

        # update the task that has the best score
        if score is not None:
            task.score = score
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_task = task_id

        self._save_task_result(task_id=task_id, pruned_model=pruned_model, masks=masks)

        self.pending_tasks.extend(self._generate_tasks(received_task_id=task_id))

    def _save_task_result(self, task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]]):
        """
        Save the task result.
        Parameters
        ----------
        task_id
            The id of the task registered in `self.tasks_map`.
        pruned_model
            The pruned model in the last iteration. It might be a sparsify model or a speed-up model.
        masks
            If masks is empty, the pruned model is a compact model after speed up.
            If masks is not None, the pruned model is a sparsify model without speed up.
        """
        task = self.tasks_map[task_id]

        # save tasks info
        with self.tasks_info_file.open(mode='r') as f:
            tasks_info = json_tricks.load(f)

        with self.tasks_info_file.open(mode='w') as f:
            tasks_info[task_id] = {PRE_TASK_ID: task.pre_task_id, SCORE: task.score, LOG_DIR: task.log_dir,
                                   STATUS: task.status}
            json_tricks.dump(tasks_info, f, indent=4)

        # save config list, pruned model and masks
        with Path(task.log_dir, CONFIG_LIST_NAME).open(mode='w') as f:
            json_tricks.dump(task.config_list, f, indent=4)
        torch.save(pruned_model, Path(task.log_dir, MODEL_NAME))
        torch.save(masks, Path(task.log_dir, MASKS_NAME))

    def load_task_result(self, task_id: int) -> Tuple[Module, Dict[str, Dict[str, Tensor]]]:
        """
        Return the pruned model and masks of the task.
        """
        task = self.tasks_map[task_id]
        model = torch.load(Path(task.log_dir, MODEL_NAME))
        masks = torch.load(Path(task.log_dir, MASKS_NAME))
        return model, masks

    def get_best_result(self) -> Optional[Tuple[int, Module, Dict[str, Dict[str, Tensor]], float]]:
        if self.best_task is None:
            _logger.warning('Do not record the score of each task, if you want to check which is the best, \
                            please pass score to `receive_task_result`')
            return None
        model, masks = self.load_task_result(self.best_task)
        return self.best_task, model, masks, self.best_score

    def _generate_tasks(self, received_task_id: int) -> List[Task]:
        """
        Subclass need implement this function to push new tasks into `self.pending_tasks`.
        """
        raise NotImplementedError()

    def next(self) -> Tuple[int, Module, List[Dict], Dict[str, Dict[str, Tensor]]]:
        """
        Get the next task.
        Returns
        -------
        Tuple[int, Module, List[Dict], Dict[str, Dict[str, Tensor]]]
            The task id, model, config_list and masks.
        """
        if len(self.pending_tasks) == 0:
            return None, None, None, None
        else:
            task = self.pending_tasks.pop(0)
            model = None
            config_list = deepcopy(task.config_list)
            masks = None
            if task.pre_task_id is not None:
                pre_task = self.tasks_map[task.pre_task_id]
                model = torch.load(Path(pre_task.log_dir, MODEL_NAME))
                masks = torch.load(Path(pre_task.log_dir, MASKS_NAME))
            return task.task_id, model, config_list, masks
