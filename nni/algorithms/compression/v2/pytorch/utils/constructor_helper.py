# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Callable, Dict, List, Type

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from nni.common.serializer import _trace_cls
from nni.common.serializer import Traceable

__all__ = ['OptimizerConstructHelper', 'LRSchedulerConstructHelper']


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
    def __init__(self, model: Module, optimizer_class: Type[Optimizer], *args, **kwargs):
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

    @staticmethod
    def from_trace(model: Module, optimizer_trace: Traceable):
        assert isinstance(optimizer_trace, Traceable), \
            'Please use nni.trace to wrap the optimizer class before initialize the optimizer.'
        assert isinstance(optimizer_trace, Optimizer), \
            'It is not an instance of torch.nn.Optimizer.'
        return OptimizerConstructHelper(model,
                                        optimizer_trace._get_nni_attr('symbol'),
                                        *optimizer_trace._get_nni_attr('args'),
                                        **optimizer_trace._get_nni_attr('kwargs'))


class LRSchedulerConstructHelper(ConstructHelper):
    def __init__(self, lr_scheduler_class: Type[_LRScheduler], *args, **kwargs):
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

    @staticmethod
    def from_trace(lr_scheduler_trace: Traceable):
        assert isinstance(lr_scheduler_trace, Traceable), \
            'Please use nni.trace to wrap the lr scheduler class before initialize the scheduler.'
        assert isinstance(lr_scheduler_trace, _LRScheduler), \
            'It is not an instance of torch.nn.lr_scheduler._LRScheduler.'
        return LRSchedulerConstructHelper(lr_scheduler_trace.trace_symbol,
                                          *lr_scheduler_trace.trace_args,
                                          **lr_scheduler_trace.trace_kwargs)
