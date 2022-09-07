from __future__ import annotations

import dis
import torch
import inspect
import operator

from typing import Dict, Optional, Iterable, Any, Union, Mapping
from torch.fx._compatibility import compatibility
from torch.fx.graph import magic_methods, reflectable_magic_methods
from torch.fx.node import Node
from torch.fx.proxy import Proxy

from . import concrete_tracer as et


@compatibility(is_backward_compatible=True)
class ConcreteProxy(Proxy, Dict):
    """
    ``ConcreteProxy`` is a wrapped proxy carried the real intermediate value, so we can use it to trace a more compatibal model, and pass the branches.
    """

    jump_opnames = (
        'JUMP_IF_FALSE_OR_POP',
        'JUMP_IF_TRUE_OR_POP',
        'POP_JUMP_IF_FALSE',
        'POP_JUMP_IF_TRUE',
        'JUMP_IF_NOT_EXC_MATCH',
    )
    jump_opcodes = tuple(dis.opmap[name] for name in jump_opnames if name in dis.opmap)


    def __init__(self, node: Node, value: Any, tracer: Optional[et.ConcreteTracer] = None):
        if tracer is None:
            # This allows you to create a ConcreteProxy object around a raw Node
            tracer = et.GraphAppendingConcreteTracer(node.graph)
        self.tracer = tracer
        self.value = value
        self.node = node

    def __repr__(self) -> str:
        return f'ConcreteProxy({self.node.name})'

    def __getattr__(self, k) -> 'ConcreteProxy':
        return ConcreteAttrProxy(self, k)

    def __call__(self, *args, **kwargs) -> 'ConcreteProxy':
        return self.tracer.create_proxy('call_method', '__call__', (self,) + args, kwargs)

    def __iter__(self) -> Iterable['ConcreteProxy']:
        return self.tracer.create_proxy('call_method', '__iter__', (self,), {})

    def __next__(self) -> Iterable['ConcreteProxy']:
        return self.tracer.create_proxy('call_method', '__next__', (self,), {})

    def __len__(self):
        return self.tracer.create_proxy('call_method', '__len__', (self,), {})

    def __getitem__(self) -> 'ConcreteProxy':
        return self.tracer.create_proxy('call_method', '__getitem__', (self,), {})

    def __bool__(self) -> Union[bool, ConcreteProxy]:
        # detect if in executing branch condition
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        insts = list(dis.get_instructions(calling_frame.f_code))
        cur = calling_frame.f_lasti // 2
        while insts[cur].opcode == dis.opmap['EXTENDED_ARG']:
            cur += 1

        if insts[cur].opcode in self.jump_opcodes:
            return self.value.__bool__()
        elif insts[cur].opcode == dis.opmap['UNARY_NOT']:
            # log warning
            print('please use the function patcher, or use "from operator import not_; x = not_(y)" instead of "x = not y", or the traced graph may be wrong')
            return self.value.__bool__()
        else:
            return self.tracer.create_proxy('call_method', '__bool__', (self,), {})

    @compatibility(is_backward_compatible=True)
    def keys(self):
        """
        for calling '**proxy'
        """
        return self.tracer.create_proxy('call_method', 'keys', (self,), {})

    @compatibility(is_backward_compatible=True)
    def values(self):
        return self.tracer.create_proxy('call_method', 'values', (self,), {})

    @compatibility(is_backward_compatible=True)
    def items(self):
        return self.tracer.create_proxy('call_method', 'items', (self,), {})

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        tracers: Dict[Any, None] = {}

        def find_tracer(a):
            if isinstance(a, cls):
                tracers[a.tracer] = None
        torch.fx.node.map_aggregate(args, find_tracer)
        torch.fx.node.map_aggregate(kwargs, find_tracer)

        if len(tracers) > 1:
            raise RuntimeError(f'Found multiple different tracers {list(tracers.keys())} while '
                               f'trying to trace operations {orig_method}')
        tracer = next(iter(tracers.keys()))

        if isinstance(orig_method, torch._C.ScriptMethod):
            args = (orig_method.owner,) + args
            return tracer.create_proxy('call_method', orig_method.name, args, kwargs)
        if torch.overrides.is_tensor_method_or_property(orig_method):
            return tracer.create_proxy('call_method', orig_method.__name__, args, kwargs)
        else:
            return tracer.create_proxy('call_function', orig_method, args, kwargs,
                                       name=tracer.graph._target_to_str(orig_method.__name__))


@compatibility(is_backward_compatible=True)
class ConcreteAttrProxy(ConcreteProxy):
    @compatibility(is_backward_compatible=True)
    def __init__(self, root: ConcreteProxy, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Optional[Node] = None
        self.value = getattr(root.value, attr)

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy(
                'call_function', getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', self.attr, (self.root,) + args, kwargs)

@compatibility(is_backward_compatible=True)
def map_aggregate_not_proxy(a, fn):
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if isinstance(a, ConcreteProxy):
        return fn(a)
    elif isinstance(a, tuple):
        t = tuple(map_aggregate_not_proxy(elem, fn) for elem in a)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        return t if not hasattr(a, '_fields') else type(a)(*t)
    elif isinstance(a, list):
        return list(map_aggregate_not_proxy(elem, fn) for elem in a)
    elif isinstance(a, dict):
        return dict((k, map_aggregate_not_proxy(v, fn)) for k, v in a.items())
    elif isinstance(a, slice):
        return slice(map_aggregate_not_proxy(a.start, fn), map_aggregate_not_proxy(a.stop, fn), map_aggregate_not_proxy(a.step, fn))
    else:
        return fn(a)

# register or wrap common methods on 'ConcreteProxy'
for method in magic_methods:
    def _scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = getattr(operator, method)
            return tracer.create_proxy('call_function', target, args, kwargs)
        impl.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(ConcreteProxy, as_magic, impl)
    _scope(method)


def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name.strip("_")}__'

    def impl(self, rhs):
        target = getattr(operator, orig_method_name)
        return self.tracer.create_proxy('call_function', target, (rhs, self), {})
    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(ConcreteProxy, method_name, impl)


for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)
