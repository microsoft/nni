import logging
import types
from typing import List, Dict, Optional, Callable, Union

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from .compressor import Compressor, LayerInfo

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

            Example::

                def trainer(model: Module, optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor]):
                    model.train()
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
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
        optimizer_cls = self._origin_optimizer.__class__
        if optimizer_cls.__name__ == 'SGD':
            self.optimizer = optimizer_cls(self.compressor.bound_model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer_cls(self.compressor.bound_model.parameters())
        self.optimizer.load_state_dict(self._origin_optimizer.state_dict())

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
    def __init__(self, dim: Optional[Union[int, List[int]]] = None):
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
        """
        self.dim = dim if not isinstance(dim, int) else [dim]
        if self.dim is not None:
            assert all(i >= 0 for i in self.dim)
            self.dim = sorted(self.dim)

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

    def __init__(self, pruner: Compressor, dim: Optional[Union[int, List[int]]] = None):
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
        """
        self.pruner = pruner
        self.dim = dim if not isinstance(dim, int) else [dim]
        if self.dim is not None:
            assert all(i >= 0 for i in self.dim)
            self.dim = sorted(self.dim)

    def generate_sparsity(self, metrics: Dict) -> Dict[str, Dict[str, Tensor]]:
        """
        Parameters
        ----------
        metrics
            A metric dict. The key is the name of layer, the value is its metric.
        """
        raise NotImplementedError()

    def _expand_mask_with_dim(self, name: str, mask: Tensor) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        name
            The masked module name.
        mask
            The dimension mask on `dim` of the weight.

        Returns
        -------
        Dict[str, Tensor]
            The key is `weight_mask` or `bias_mask`, value is the final mask.
        """
        wrapper = self.pruner._get_modules_wrapper()[name]
        weight_size = wrapper.module.weight.data.size()
        if self.dim is None:
            assert len(mask.size()) == len(weight_size)
            expand_mask = {'weight_mask': mask}
        else:
            # expand mask to weight size
            assert len(mask.size()) == len(self.dim)
            assert all(weight_size[j] == mask.size()[i] for i, j in enumerate(self.dim))
            idxs = list(range(len(weight_size)))
            [idxs.pop(i) for i in reversed(self.dim)]
            weight_mask = mask.clone()
            for i in idxs:
                weight_mask = weight_mask.unsqueeze(i)
            expand_mask = {'weight_mask': weight_mask.expand(weight_size).clone()}
            # NOTE: assume we only mask output, so the mask and bias have a one-to-one correspondence.
            # If we support more kind of masks, this place need refactor.
            if wrapper.bias_mask is not None and mask.size() == wrapper.bias_mask.size():
                expand_mask['bias_mask'] = mask.clone()
        return expand_mask
