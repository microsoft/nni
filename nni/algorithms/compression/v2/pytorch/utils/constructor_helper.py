from copy import deepcopy
from typing import Callable, Dict, List

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import StepLR, _LRScheduler

from nni.common.serializer import _trace_cls, SerializableObject
from nni.algorithms.compression.v2.pytorch.base import Pruner

from examples.model_compress.models.cifar10.vgg import VGG


def trace_cls(base, kw_only: bool = True):
    return _trace_cls(base, kw_only=kw_only, _self_contained=False)


class ConstructHelper:
    def __init__(self, callable_obj: Callable, *args, **kwargs):
        assert callable(callable_obj), '`callable_obj` must be a callable object.'
        self.callable_obj = callable_obj
        self.args = deepcopy(args)
        self.kwargs = deepcopy(kwargs)

    def call(self):
        args = deepcopy(self.args)
        kwargs = deepcopy(self.kwargs)
        return self.callable_obj(*args, **kwargs)


class OptimizerConstructHelper(ConstructHelper):
    def __init__(self, model: Module, optimizer_class: Optimizer, *args, **kwargs):
        assert isinstance(model, Module), 'Only support pytorch module.'
        assert issubclass(optimizer_class, Optimizer), 'Only support pytorch optimizer'

        args = list(args)
        if 'params' in kwargs:
            kwargs['params'] = self.params2names(model, kwargs['params'])
        else:
            args[0] = self.params2names(model, args[0])
        super().__init__(optimizer_class, *args, **kwargs)

    def params2names(self, model: Module, params: List) -> List[Dict]:
        param_groups = list(params)
        assert len(param_groups) > 0
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            params = param_group['params']
            if isinstance(params, Tensor):
                params = [params]
            elif isinstance(params, set):
                raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                                'the ordering of tensors in sets will change between runs. Please use a list instead.')
            else:
                params = list(params)
            param_ids = [id(p) for p in params]
            param_group['params'] = [name for name, p in model.named_parameters() if id(p) in param_ids]

        return param_groups

    def names2params(self, wrapped_model: Module, origin2wrapped_name_map: Dict, params: List[Dict]) -> List[Dict]:
        param_groups = deepcopy(params)
        for param_group in param_groups:
            wrapped_names = [origin2wrapped_name_map.get(name, name) for name in param_group['params']]
            param_group['params'] = [p for name, p in wrapped_model.named_parameters() if name in wrapped_names]
        return param_groups

    def call(self, wrapped_model: Module, origin2wrapped_name_map: Dict) -> Optimizer:
        args = deepcopy(self.args)
        kwargs = deepcopy(self.kwargs)

        if 'params' in kwargs:
            kwargs['params'] = self.names2params(wrapped_model, origin2wrapped_name_map, kwargs['params'])
        else:
            args[0] = self.names2params(wrapped_model, origin2wrapped_name_map, args[0])

        return self.callable_obj(*args, **kwargs)


class LRSchedulerConstructHelper(ConstructHelper):
    def __init__(self, lr_scheduler_class: Callable, *args, **kwargs):
        args = list(args)
        if 'optimizer' in kwargs:
            kwargs['optimizer'] = None
        else:
            args[0] = None
        super().__init__(lr_scheduler_class, *args, **kwargs)

    def call(self, optimizer: Optimizer) -> _LRScheduler:
        args = deepcopy(self.args)
        kwargs = deepcopy(self.kwargs)

        if 'optimizer' in kwargs:
            kwargs['optimizer'] = optimizer
        else:
            args[0] = optimizer

        return self.callable_obj(*args, **kwargs)


class FakePruner(Pruner):
    def __init__(self, model, config_list, optimizer_trace, lr_scheduler_trace):
        self.optimizer_helper = self._optimizer_construct_helper(model, optimizer_trace)
        self.lr_scheduler_helper = self._lr_scheduler_construct_helper(lr_scheduler_trace)
        super().__init__(model, config_list)

    def _optimizer_construct_helper(self, model: Module, optimizer_trace: SerializableObject):
        assert isinstance(optimizer_trace, SerializableObject), \
            'Please use nni.trace to wrap the optimizer class before initialize the optimizer.'
        assert isinstance(optimizer_trace, Optimizer), \
            'It is not an instance of torch.nn.Optimizer.'
        return OptimizerConstructHelper(model,
                                        optimizer_trace._get_nni_attr('symbol'),
                                        *optimizer_trace._get_nni_attr('args'),
                                        **optimizer_trace._get_nni_attr('kwargs'))

    def _lr_scheduler_construct_helper(self, lr_scheduler_trace: SerializableObject):
        assert isinstance(lr_scheduler_trace, SerializableObject), \
            'Please use nni.trace to wrap the lr scheduler class before initialize the scheduler.'
        assert isinstance(lr_scheduler_trace, _LRScheduler), \
            'It is not an instance of torch.nn.lr_scheduler._LRScheduler.'
        return LRSchedulerConstructHelper(lr_scheduler_trace._get_nni_attr('symbol'),
                                          *lr_scheduler_trace._get_nni_attr('args'),
                                          **lr_scheduler_trace._get_nni_attr('kwargs'))

    @property
    def origin2wrapped_name_map(self):
        self._unwrap_model()
        origin_param_names = [name for name, _ in self.bound_model.named_parameters()]
        self._wrap_model()
        wrapped_param_names = [name for name, _ in self.bound_model.named_parameters()]
        origin2wrapped_name_map = {k: v for k, v in zip(origin_param_names, wrapped_param_names)}
        return origin2wrapped_name_map


if __name__ == '__main__':
    model = VGG()
    optimizer_trace = trace_cls(Adam)(model.parameters(), 0.001, (0.9, 0.999))
    scheduler_trace = trace_cls(StepLR)(optimizer_trace, step_size=2)

    pruner = FakePruner(model, [{'op_types': ['Conv2d'], 'sparsity': 0.5}], optimizer_trace, scheduler_trace)

    optimizer = pruner.optimizer_helper.call(model, pruner.origin2wrapped_name_map)
    lr_scheduler = pruner.lr_scheduler_helper.call(optimizer)
