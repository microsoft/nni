import logging
import types
from typing import List, Dict, Optional, Callable

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from compression_v2.base.compressor import Compressor, LayerInfo
from compression_v2.base.common import DataCollector

logger = logging.getLogger(__name__)


class WeightDataCollector(DataCollector):
    def reset(self):
        pass

    def collect(self) -> Dict[str, Tensor]:
        data = {}
        for _, wrapper in self.compressor._get_modules_wrapper().items():
            data[wrapper.name] = wrapper.module.weight.data.clone()
        return data


class HookCollectorInfo:
    def __init__(self, layers: List[LayerInfo], hook_type: str,
                 collector: Callable[[List], Callable[[Module, Tensor, Tensor], None]]):
        self.layers = layers
        self.hook_type = hook_type
        self.collector = collector


class TrainerBasedDataCollector(DataCollector):
    def __init__(self, compressor: Compressor, trainer: Callable[[Module, Optimizer, Callable], None], optimizer: Optimizer,
                 criterion: Callable[[Tensor, Tensor], Tensor], opt_before_tasks: Optional[List] = None,
                 opt_after_tasks: Optional[List] = None,
                 collector_infos: List[HookCollectorInfo] = None):
        super().__init__(compressor)
        self.trainer = trainer
        self._origin_optimizer = optimizer
        self._origin_criterion = criterion
        self._opt_before_tasks = opt_before_tasks
        self._opt_after_tasks = opt_after_tasks

        self._collector_infos = collector_infos

        self.reset()

    def reset(self):
        # refresh optimizer
        self.compressor._unwrap_model()
        optimizer_cls = self._origin_optimizer.__class__
        if optimizer_cls.__name__ == 'SGD':
            self.optimizer = optimizer_cls(self.compressor.bound_model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer_cls(self.compressor.bound_model.parameters())
        self.optimizer.load_state_dict(self._origin_optimizer.state_dict())
        self.compressor._wrap_model()

        # patch optimizer
        self._patch_optimizer()

        # hook
        self._remove_all_hook()
        self._hook_id = 0
        self._hook_handles = {}
        self._hook_buffer = {}

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

    def _add_hook(self, collector_info: HookCollectorInfo):
        self._hook_id += 1
        self._hook_handles[self._hook_id] = {}
        self._hook_buffer[self._hook_id] = {}

        if collector_info.hook_type == 'forward':
            self._add_forward_hook(self._hook_id, collector_info.layers, collector_info.collector)
        elif collector_info.hook_type == 'backward':
            self._add_backward_hook(self._hook_id, collector_info.layers, collector_info.collector)
        else:
            logger.warning('Skip unsupported hook type: %s', collector_info.hook_type)

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

    def _remove_all_hook(self):
        if hasattr(self, '_hook_handles'):
            for hook_id in list(self._hook_handles.keys()):
                self._remove_hook(hook_id)
