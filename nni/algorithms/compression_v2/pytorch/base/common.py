from copy import deepcopy
import logging
import types
from typing import List, Dict, Optional, Callable, Union

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from nni.algorithms.compression_v2.pytorch.base.compressor import Compressor, LayerInfo

_logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, compressor: Compressor):
        self.compressor = compressor

    def reset(self):
        raise NotImplementedError()

    def collect(self) -> Dict:
        raise NotImplementedError()


class HookCollectorInfo:
    def __init__(self, layers: List[LayerInfo], hook_type: str,
                 collector: Callable[[List], Callable[[Module, Tensor, Tensor], None]]):
        self.layers = layers
        self.hook_type = hook_type
        self.collector = collector


class TrainerBasedDataCollector(DataCollector):
    def __init__(self, compressor: Compressor, trainer: Callable[[Module, Optimizer, Callable], None], optimizer: Optimizer,
                 criterion: Callable[[Tensor, Tensor], Tensor], training_epochs: int,
                 opt_before_tasks: List = [], opt_after_tasks: List = [],
                 collector_infos: List[HookCollectorInfo] = [], criterion_patch: Callable[[Callable], Callable] = None):
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
            self._add_forward_hook(self._hook_id, collector_info.layers, collector_info.collector)
        elif collector_info.hook_type == 'backward':
            self._add_backward_hook(self._hook_id, collector_info.layers, collector_info.collector)
        else:
            _logger.warning('Skip unsupported hook type: %s', collector_info.hook_type)

        return self._hook_id

    def _add_forward_hook(self, hook_id: int, layers: List[LayerInfo],
                          collector: Callable[[List], Callable[[Module, Tensor, Tensor], None]]):
        for layer in layers:
            self._hook_buffer[hook_id][layer.name] = []
            handle = layer.module.register_forward_hook(collector(self._hook_buffer[hook_id][layer.name]))
            self._hook_handles[hook_id][layer.name] = handle

    def _add_backward_hook(self, hook_id: int, layers: List[LayerInfo],
                           collector: Callable[[List], Callable[[Module, Tensor, Tensor], None]]):
        for layer in layers:
            self._hook_buffer[hook_id][layer.name] = []
            handle = layer.module.register_backward_hook(collector(self._hook_buffer[hook_id][layer.name]))
            self._hook_handles[hook_id][layer.name] = handle

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
    def calculate_metrics(self, data: Dict) -> Dict[str, Tensor]:
        raise NotImplementedError()


class SparsityAllocator:
    def __init__(self, pruner: Compressor, dim: Optional[Union[int, List[int]]] = None):
        """
        Parameters
        ----------
        pruner
            The pruner that wrapped the module.
        dim
            The dimensions that corresponding to the metric, None means one-to-one correspondence.
        """
        self.pruner = pruner
        self.dim = dim if not isinstance(dim, int) else [dim]
        if self.dim is not None:
            assert all(i >= 0 for i in self.dim)
            self.dim = sorted(self.dim)

    def generate_sparsity(self, metrics: Dict) -> Dict[str, Dict[str, Tensor]]:
        raise NotImplementedError()

    def _expand_mask_with_dim(self, name: str, mask: Tensor) -> Dict[str, Tensor]:
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
            # NOTE: assume we only mask output
            if wrapper.bias_mask is not None and mask.size() == wrapper.bias_mask.size():
                expand_mask['bias_mask'] = mask.clone()
        return expand_mask
