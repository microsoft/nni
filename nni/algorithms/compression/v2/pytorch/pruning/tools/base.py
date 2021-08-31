# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import types
from typing import List, Dict, Literal, Tuple, Optional, Callable, Union

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
            mask = torch.einsum(ein_expression, mask, torch.ones(self.block_sparse_size).to(mask.device))

        return (mask != 0).type_as(mask)


@dataclass
class Task:
    """
    Task saves the related information about the task.
    """
    task_id: int
    # the path of the model under pruning in this task
    model_path: Path
    # the config_list used in this task
    config_list: List[Dict]
    # the path of masks applying on the under pruning model in this task
    masks_path: Path
    status: Literal['Pending', 'Running', 'Finished'] = 'Pending'
    score: Optional[float] = None
    log_dir: Optional[Path] = None
    state: dict = field(default_factory=dict)


class TaskGenerator:
    """
    This class used to generate config list for pruner in each iteration.
    """
    def __init__(self, origin_model: Module, origin_config_list: List[Dict] = [],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = '.', save_result: bool = True):
        """
        Parameters
        ----------
        origin_model
            The origin model under pruning.
        origin_config_list
            The origin config list provided by the user. Note that this config_list is directly config the origin model.
            This means the sparsity provided by the origin_masks should also be recorded in the origin_config_list.
        origin_masks
            The pre masks apply on the origin model.
        log_dir
            The log directory use to saving the task generator log.
        save_result
            If saving the intermediate result, including intermediate model and masks during each iteration.
        """
        assert isinstance(origin_model, Module), 'Only support pytorch module.'

        self._log_dir_root = Path(log_dir).absolute()
        self._log_dir_root.mkdir(parents=True, exist_ok=True)
        self._save_result = save_result

        # save the origin data under {log_dir}/origin/
        self._save_origin_data(origin_model, origin_config_list, origin_masks)

        self._task_id_candidate = 0
        self._tasks: Dict[int, Task] = {}
        self._pending_tasks: List[Task] = self._init_pending_tasks()
        self._best_score = None
        self._best_task_id = None

        # dump self._tasks into {log_dir}/.tasks
        self._dump_tasks_info()

    def _save_data(self, name: str, model: Module, config_list: List[Dict],
                   masks: Dict[str, Dict[str, Tensor]]):
        save_dir = Path(self._log_dir_root, name)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model, Path(save_dir, '{}_model.pth'.format(name)))
        torch.save(masks, Path(save_dir, '{}_masks.pth'.format(name)))
        with Path(save_dir, 'config_list.json').open('w') as f:
            json_tricks.dump(config_list, f, indent=4)

    def _load_data(self, name: str) -> Tuple[Module, List[Dict], Dict[str, Dict[str, Tensor]]]:
        save_dir = Path(self._log_dir_root, name)
        model = torch.load(Path(save_dir, '{}_model.pth'.format(name)))
        masks = torch.load(Path(save_dir, '{}_masks.pth'.format(name)))
        with Path(save_dir, 'config_list.json').open('r') as f:
            config_list = json_tricks.load(f)
        return model, config_list, masks

    def _save_origin_data(self, origin_model: Module, origin_config_list: List[Dict],
                          origin_masks: Dict[str, Dict[str, Tensor]]):
        self._save_data('origin', origin_model, origin_config_list, origin_masks)

    def _load_origin_data(self) -> Tuple[Module, List[Dict], Dict[str, Dict[str, Tensor]]]:
        return self._load_data('origin')

    def _save_best_result(self, best_model: Module, best_config_list: List[Dict],
                          best_masks: Dict[str, Dict[str, Tensor]]):
        self._save_data('best', best_model, best_config_list, best_masks)

    def _load_best_result(self) -> Tuple[Module, List[Dict], Dict[str, Dict[str, Tensor]]]:
        return self._load_data('best')

    def get_best_result(self) -> Optional[Tuple[int, Module, List[Dict], Dict[str, Dict[str, Tensor]], float]]:
        """
        Returns
        -------
         Optional[Tuple[int, Module, List[Dict], Dict[str, Dict[str, Tensor]], float]]
            If self._best_task_id is not None,
            return best task id, best pruned model, config list used in this task, masks on the pruned model, score.
        """
        if self._best_task_id is not None:
            model, config_list, masks = self._load_best_result()
            return self._best_task_id, model, config_list, masks, self._best_score
        return None

    def _save_task_result(self, task_id: int, model: Module, masks: Dict[str, Dict[str, Tensor]],
                          score: Optional[float], origin_masks: Dict[str, Dict[str, Tensor]]):
        task = self._tasks[task_id]
        if score is not None:
            task.score = score
            if self._best_score is None or score > self._best_score:
                self._best_score = score
                self._best_task_id = task_id
                self._save_best_result(model, task.config_list, masks)
        if self._save_result:
            torch.save(model, Path(task.log_dir, 'pruned_model.pth'))
            torch.save(masks, Path(task.log_dir, 'masks_on_pruned_model.pth'))
            under_pruning_model, _, _ = self._load_task_data(task_id)
            torch.save(under_pruning_model, Path(task.log_dir, 'under_pruning_model.pth'))
            torch.save(origin_masks, Path(task.log_dir, 'masks_on_under_pruning_model.pth'))

    def _load_task_data(self, task_id: int) -> Tuple[Module, List[Dict], Dict[str, Dict[str, Tensor]]]:
        task = self._tasks[task_id]
        model = torch.load(task.model_path)
        config_list = deepcopy(task.config_list)
        masks = torch.load(task.masks_path)
        return model, config_list, masks

    def _dump_tasks_info(self):
        tasks = {task_id: asdict(task) for task_id, task in self._tasks.items()}
        with Path(self._log_dir_root, '.tasks').open('w') as f:
            json_tricks.dump(tasks, f, indent=4)

    def _init_pending_tasks(self) -> List[Task]:
        raise NotImplementedError()

    def _generate_tasks(self, received_task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]],
                        origin_masks: Dict[str, Dict[str, Tensor]]) -> List[Task]:
        raise NotImplementedError()

    def receive_task_result(self, task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]],
                            score: Optional[float], origin_masks: Dict[str, Dict[str, Tensor]]):
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
        origin_masks
            The masks should be apply on the under pruning model.
            If the pruned model did not speed up, origin_masks is same as masks.
        """
        assert task_id in self._tasks, 'Task {} does not exist.'.format(task_id)
        self._save_task_result(task_id, pruned_model, masks, score, origin_masks)
        self._tasks[task_id].status = 'Finished'
        self._dump_tasks_info()
        self._pending_tasks.extend(self._generate_tasks(task_id, pruned_model, masks, origin_masks))
        self._dump_tasks_info()

    def next(self) -> Tuple[int, Module, List[Dict], Dict[str, Dict[str, Tensor]]]:
        """
        Get the next task.

        Returns
        -------
        Tuple[int, Module, List[Dict], Dict[str, Dict[str, Tensor]]]
            The task id, model, config_list and masks.
        """
        if len(self._pending_tasks) == 0:
            return None, None, None, None
        else:
            task = self._pending_tasks.pop(0)
            model, config_list, masks = self._load_task_data(task.task_id)
            task.status = 'Running'
            self._dump_tasks_info()
            return task.task_id, model, config_list, masks
