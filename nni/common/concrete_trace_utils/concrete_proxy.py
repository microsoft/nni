# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import dis
import logging
import inspect
import operator

from typing import List, Optional, Iterable, Any, Set, Union

import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import magic_methods, reflectable_magic_methods
from torch.fx.node import Node
from torch.fx.proxy import Proxy
from torch.overrides import is_tensor_method_or_property

from . import concrete_tracer as et
from .utils import (
    _orig_tuple,
    _orig_list,
    _orig_type,
    _orig_isinstance,
    _orig_getattr,
    _orig_range,
    _orig_dict,
    _orig_len,
    _orig_index,
    _orig_bool,
    _orig_slice,
    _orig_set,
    map_recursive,
)

_logger = logging.getLogger(__name__)

@compatibility(is_backward_compatible=True)
class ConcreteProxy(Proxy):
    """
    `ConcreteProxy` is a wrapped proxy carried the real intermediate value.
    We can use it to trace a more compatible model, and pass the branches.
    """

    # TODO: python bytecode changes a lot in version 3.11. these ops should be updated.
    jump_opnames = (
        'JUMP_IF_FALSE_OR_POP',
        'JUMP_IF_TRUE_OR_POP',
        'POP_JUMP_IF_FALSE',
        'POP_JUMP_IF_TRUE',
        'JUMP_IF_NOT_EXC_MATCH', # occurred in new python vertion, not tested
    )
    jump_opcodes = _orig_tuple(dis.opmap[name] for name in jump_opnames if name in dis.opmap)
    op_compare = dis.opmap['COMPARE_OP']
    op_extended_arg = dis.opmap['EXTENDED_ARG']
    op_call_ex = dis.opmap['CALL_FUNCTION_EX']
    op_not = dis.opmap['UNARY_NOT']
    op_unpack_sequence = dis.opmap['UNPACK_SEQUENCE']
    op_dict_merge = dis.opmap.get('DICT_MERGE', None)  # DICT_MERGE is new in python 3.9
    jump_before_opcodes = (op_compare, op_not)

    # occurred in different python versions
    op_list_extend = dis.opmap['LIST_EXTEND'] if 'LIST_EXTEND' in dis.opmap else None
    op_tuple_unpack_call = dis.opmap['BUILD_TUPLE_UNPACK_WITH_CALL'] if 'BUILD_TUPLE_UNPACK_WITH_CALL' in dis.opmap else None

    def __init__(self, node: Node, value: Any, tracer: Optional[et.ConcreteTracer] = None):
        if tracer is None:
            # This allows you to create a ConcreteProxy object around a raw Node
            tracer = et.GraphAppendingConcreteTracer(node.graph)
        self.tracer = tracer
        self.value = value
        self.node = node

    def __repr__(self) -> str:
        return f'ConcreteProxy({self.node.name}, {self.value})'

    def __getattr__(self, k) -> ConcreteProxy:
        return ConcreteAttrProxy(self, k)

    def __call__(self, *args, **kwargs) -> ConcreteProxy:
        return self.tracer.create_proxy('call_method', '__call__', (self,) + args, kwargs)

    def __iter__(self) -> Union[Iterable, ConcreteProxy]:
        # to detect if in executing `*proxy`, or `a, b, c = atuple`
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        cur = calling_frame.f_lasti // 2
        insts: List[dis.Instruction] = _orig_list(dis.get_instructions(calling_frame.f_code))
        while insts[cur].opcode == self.op_extended_arg:
            cur += 1

        if insts[cur].opcode == self.op_call_ex:
            # in executing func(..., *proxy)
            # todo: don't know the func has type_guard or not
            return ConcreteUnpackIterProxy(self)
        elif insts[cur].opcode == self.op_tuple_unpack_call:
            # in executing func(*..., *proxy)
            # todo: don't know the func has type_guard or not
            # <= python 3.8
            return ConcreteUnpackIterProxy(self)
        elif insts[cur].opcode == self.op_list_extend:
            # in executing x.extend(proxy) or [x, *proxy]
            # >= python 3.9
            return ConcreteUnpackIterProxy(self)
        elif insts[cur].opcode == self.op_unpack_sequence:
            # in executing `a, b, c = atuple`
            return ConcreteUnpackIterProxy(self)
        elif insts[cur].opname == 'GET_ITER' and insts[cur + 1].opname == 'FOR_ITER' and _orig_isinstance(self.value, _orig_range):
            # in executing `for i in range(...)`
            return iter(self.value)
        # elif insts[cur].opname == 'CONTAINS_OP':
        #     # in executing `for i in range(...)`
        #     return iter(self.value)
        else:
            return self.tracer.create_proxy('call_function', iter, (self,), {})

    def __next__(self) -> ConcreteProxy:
        return self.tracer.create_proxy('call_function', next, (self,), {})

    def __len__(self) -> Union[int, ConcreteProxy]:
        # to detect if in executing `*proxy`
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        cur = calling_frame.f_lasti // 2
        insts: List[dis.Instruction] = _orig_list(dis.get_instructions(calling_frame.f_code))
        while insts[cur].opcode == self.op_extended_arg:
            cur += 1

        if insts[cur].opcode == self.op_call_ex:
            # in executing func(..., *proxy)
            return _orig_len(self.value)
        elif insts[cur].opcode == self.op_tuple_unpack_call:
            # in executing func(*..., *proxy)
            # <= python 3.8
            return _orig_len(self.value)
        elif insts[cur].opcode == self.op_list_extend:
            # in executing x.extend(*proxy) or [x, *proxy]
            # >= python 3.9
            return _orig_len(self.value)
        else:
            return self.tracer.create_proxy('call_function', _orig_len, (self,), {})

    def __getitem__(self, *args, **kwargs) -> ConcreteProxy:
        return self.tracer.create_proxy('call_function', operator.getitem, (self,) + args, kwargs)

    def __setitem__(self, *args, **kwargs) -> ConcreteProxy:
        return self.tracer.create_proxy('call_function', operator.setitem, (self,) + args, kwargs)

    def __bool__(self) -> Union[bool, ConcreteProxy]:
        # to detect if in executing branch condition
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        cur = calling_frame.f_lasti // 2
        insts: List[dis.Instruction] = _orig_list(dis.get_instructions(calling_frame.f_code))
        while insts[cur].opcode == self.op_extended_arg:
            cur += 1

        if insts[cur].opcode in self.jump_opcodes or (
            insts[cur].opcode in self.jump_before_opcodes and insts[cur + 1].opcode in self.jump_opcodes):
            # in executing branch condition
            return _orig_bool(self.value)
        elif insts[cur].opname == 'CONTAINS_OP':
            # in executing 'in'
            return _orig_bool(self.value)
        elif insts[cur].opcode == self.op_call_ex:
            # in executing func(..., *proxy)
            return _orig_bool(self.value)
        elif insts[cur].opcode == self.op_not:
            # We cannot return a proxy because 'UNARY_NOT' op will check the type.
            _logger.warning('please use the function patcher, or use "x = operator.not_(y)" instead of "x = not y",'
                            'otherwise the traced graph may be wrong')
            return _orig_bool(self.value)
        else:
            return self.tracer.create_proxy('call_function', _orig_bool, (self,), {})

    def __index__(self) -> Union[int, ConcreteProxy]:
        # should only be in list/tuple getitem
        return _orig_index(self.value)

    def __hash__(self) -> Union[int, ConcreteProxy]:
        # should only be in dict getitem
        return hash(self.value)

    @compatibility(is_backward_compatible=True)
    def keys(self):
        # to detect if in executing `**proxy`
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        cur = calling_frame.f_lasti // 2
        insts: List[dis.Instruction] = _orig_list(dis.get_instructions(calling_frame.f_code))
        while insts[cur].opcode == self.op_extended_arg:
            cur += 1

        if insts[cur].opcode == self.op_call_ex or insts[cur].opcode == self.op_dict_merge:
            # in executing `**proxy`
            return self.value.keys()
        else:
            return self.tracer.create_proxy('call_method', 'keys', (self,), {})

    @compatibility(is_backward_compatible=True)
    def values(self):
        return self.tracer.create_proxy('call_method', 'values', (self,), {})

    @compatibility(is_backward_compatible=True)
    def items(self):
        return self.tracer.create_proxy('call_method', 'items', (self,), {})

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        # to wrap all the functions/methods with tensor inputs in the namespace 'torch.*'.
        # actually a simple way to do wrap, but may get wrong in functions with no tensor inputs.
        # TODO: now for most functions in torch namespace, we do wrap directly and not use __torch_function__

        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        tracers: Set[Any] = _orig_set()

        def find_tracer(a):
            if _orig_isinstance(a, cls):
                tracers.add(a.tracer)
        map_recursive(find_tracer, args)
        map_recursive(find_tracer, kwargs)

        if _orig_len(tracers) > 1:
            raise RuntimeError(f'Found multiple different tracers {_orig_list(tracers)} while '
                               f'trying to trace operations {orig_method}')
        tracer, = tracers

        if isinstance(orig_method, torch._C.ScriptMethod):
            args = (orig_method.owner,) + args
            return tracer.create_proxy('call_method', orig_method.name, args, kwargs)
        if is_tensor_method_or_property(orig_method):
            return tracer.create_proxy('call_method', orig_method.__name__, args, kwargs)
        else:
            return tracer.create_proxy('call_function', orig_method, args, kwargs,
                                       name=tracer.graph._target_to_str(orig_method.__name__))


@compatibility(is_backward_compatible=True)
class ConcreteAttrProxy(ConcreteProxy):
    """
    A more understandable way to deal with sub-field like 'x.y'.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, root: ConcreteProxy, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Optional[Node] = None
        self.value = _orig_getattr(root.value, attr)

    def __repr__(self) -> str:
        calling_frame_name = inspect.stack()[1][1]
        if calling_frame_name.endswith('pydevd_exe2.py') or calling_frame_name.endswith('pydevd_safe_repr.py'):
            return f'ConcreteAttrProxy({self.node.name})'
        return repr(self.value)

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy(
                'call_function', _orig_getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', self.attr, (self.root,) + args, kwargs)


@compatibility(is_backward_compatible=True)
class ConcreteUnpackIterProxy(ConcreteProxy):
    """
    A more understandable way to deal with iterables.
    Only support 'tuple' and 'list'. Will transfer un-subscriptables such as 'set', to 'tuple'.
    todo: support for 'zip'

    examples:
        1. `a, b = c` =>
            ori:
                iter1 = c.__iter__()
                a = iter1.__next__()
                b = iter1.__next__()
            new:
                a = c[0]
                b = c[1]

        2. `y = [x, *proxy]` =>
            ori:
                iter1 = c.__iter__()
                a = iter1.__next__()
                b = iter1.__next__()
                y = [x, a, b]
            new:
                a = proxy[0]
                b = proxy[1]
                y = [x, a, b]
    """

    @staticmethod
    def try_create(root: Any):
        if isinstance(root, ConcreteProxy):
            return ConcreteUnpackIterProxy(root)
        else:
            return iter(root)

    @compatibility(is_backward_compatible=True)
    def __init__(self, root: ConcreteProxy):
        if not hasattr(root.value, '__getitem__'):
            # transfer 'set' to 'tuple'
            # it's tuple not _orig_tuple!
            # root = tuple(root)
            root = root.tracer.create_proxy('call_function', _orig_tuple, (root,), {})
        self.root = root
        self.tracer = root.tracer
        self._node: Optional[Node] = None
        self._value: List[Any] = []
        self.index = -1
        self.len = _orig_len(root.value)

    def __repr__(self) -> str:
        return f'ConcreteUnpackIterProxy({self.node.name})'

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy(
                'call_function', iter, (self.root,), {}).node
        return self._node

    @property
    def value(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if _orig_len(self._value) == 0:
            self._value.append(iter(self.root.value))
        return self._value[0]

    def __next__(self):
        self.index += 1
        if self.index == self.len:
            raise StopIteration()
        return self.tracer.create_proxy('call_function', operator.getitem, (self.root, self.index), {})

@compatibility(is_backward_compatible=True)
def map_aggregate_not_proxy(a, fn):
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if _orig_isinstance(a, ConcreteProxy):
        return fn(a)
    elif _orig_isinstance(a, _orig_tuple):
        t = _orig_tuple(map_aggregate_not_proxy(elem, fn) for elem in a)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        return t if not hasattr(a, '_fields') else _orig_type(a)(*t)
    elif _orig_type(a) == _orig_list:
        return _orig_list(map_aggregate_not_proxy(elem, fn) for elem in a)
    elif _orig_isinstance(a, _orig_dict):
        return _orig_dict((k, map_aggregate_not_proxy(v, fn)) for k, v in a.items())
    elif _orig_isinstance(a, _orig_slice):
        return _orig_slice(map_aggregate_not_proxy(a.start, fn), map_aggregate_not_proxy(a.stop, fn), map_aggregate_not_proxy(a.step, fn))
    else:
        return fn(a)

# register or wrap common methods on 'ConcreteProxy'
# for method in magic_methods:
# torch.fx.graph.inplace_methods may not exist on some verion of pytorch
inplace_methods = {
    'iadd': '{} += {}',
    'iand': '{} &= {}',
    'ifloordiv': '{} //= {}',
    'ilshift': '{} <<= {}',
    'imod': '{} %= {}',
    'imul': '{} *= {}',
    'imatmul': '{} @= {}',
    'ior': '{} |= {}',
    'ipow': '{} **= {}',
    'irshift': '{} >>= {}',
    'isub': '{} -= {}',
    'itruediv': '{} /= {}',
    'ixor': '{} ^= {}',
    'setitem': '{}[{}] = {}',
}
for method in {**magic_methods, **inplace_methods}:
    def _scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = _orig_getattr(operator, method)
            return tracer.create_proxy('call_function', target, args, kwargs)
        impl.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(ConcreteProxy, as_magic, impl)
    _scope(method)


def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name.strip("_")}__'

    def impl(self, rhs):
        target = _orig_getattr(operator, orig_method_name)
        return self.tracer.create_proxy('call_function', target, (rhs, self), {})
    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(ConcreteProxy, method_name, impl)


for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)
