# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import sys
import inspect
import operator
import math
import functools
import builtins

from itertools import chain
from types import FunctionType, MethodType, ModuleType
from typing import Any, Dict, Optional, Set, Tuple, Type, List, Callable, Union

import torch
from torch._C import ScriptObject
from torch.fx import GraphModule
from torch.fx._compatibility import compatibility
from torch.fx._symbolic_trace import _Patcher, _proxyable_classes
from torch.fx import graph as fx_graph
from torch.fx.graph import Graph
from torch.fx.node import Target, Node
from torch.fx.proxy import TracerBase

from . import concrete_proxy as ep
# import .concrete_proxy as ep
from .operator_patcher import OperatorPatcher

HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS

# These need to run in global scope to handle nested calls correctly
_orig_module_call: Callable = torch.nn.Module.__call__
_orig_module_getattr: Callable = torch.nn.Module.__getattr__
_orig_isinstance: Callable = builtins.isinstance
_orig_bool: Type[Any] = builtins.bool
_orig_tuple: Type[Any] = builtins.tuple
_orig_list: Type[Any] = builtins.list
_orig_set: Type[Any] = builtins.set
_orig_frozenset: Type[Any] = builtins.frozenset
_orig_dict: Type[Any] = builtins.dict
_orig_len: Callable = builtins.len
_orig_not: Callable = operator.not_
_orig_is: Callable = operator.is_
_orig_is_not: Callable = operator.is_not
_orig_contains: Callable = operator.contains


@compatibility(is_backward_compatible=True)
class ConcreteTracer(TracerBase):
    """
    A model tracer similar to _symbolic_trace.Tracer, but with concrete execution and real value so we can pass complecate conditions
    and go into correct brunches.
    """

    default_autowrap_modules = (
        math,
    )
    default_autowap_funcs = (
        # no necessary input for tensor, so __torch_function__ will not be called. need to be wrapped manually.
        # todo:
        #   1. more needed to be tested.
        #   2. some with no proxy input functions such as 'torch.rand' should also be traced.
        torch.arange,
        torch.meshgrid,
    )
    @compatibility(is_backward_compatible=True)
    def __init__(self, autowrap_modules: Tuple[ModuleType] = default_autowrap_modules,
                 autowrap_functions: Tuple[Callable, ...] = default_autowap_funcs) -> None:
        """
        similar to _symbolic_trace.Tracer.__init__.
        remove the 'param_shapes_constant' because we can get real shape when executing.
        """
        super().__init__()

        # Functions we will eagerly wrap when we see them while tracing
        # this captures both `math.sqrt()` and `from math import sqrt` automatically
        self._autowrap_function_ids: Set[int] = {
            id(value) for name, value in chain(*[m.__dict__.items() for m in autowrap_modules])
            if not name.startswith("_") and callable(value)}
        self._autowrap_function_ids.update(set([id(f) for f in autowrap_functions]))

        # Python modules to apply autowrap to at the start, in addition to
        # modules we see while tracing
        self._autowrap_search: List[ModuleType] = list(autowrap_modules)
        for func in autowrap_functions:
            if hasattr(func, '__module__'):
                self._autowrap_search.append(sys.modules[func.__module__])

        self.submodule_paths: Optional[Dict[torch.nn.Module, str]] = None

    @compatibility(is_backward_compatible=True)
    def fetch_attr(self, target: str):
        """
        to get the attr in self.root. only for execution of 'call_module' nodes.
        """
        temp_disable_attr = self.temp_disable_attr
        self.temp_disable_attr = True
        target_atoms = target.split('.')
        attr_itr = self.root
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        self.temp_disable_attr = temp_disable_attr
        return attr_itr

    def run_target(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        """
        actually execute the code.
        apply the patcher, and the _autowrap_check to the target function.
        """
        if kind == 'call_function':
            fn = target
            if hasattr(fn, '__globals__'):
                _autowrap_check(self.patcher, fn.__globals__, self._autowrap_function_ids)
            fn = self.op_patcher.patch(fn)
            return fn(*args, **kwargs)
        elif kind == 'call_method':
            self_obj, *args_tail = args
            fn = getattr(self_obj, target)
            if hasattr(fn, '__globals__'):
                _autowrap_check(self.patcher, fn.__globals__, self._autowrap_function_ids)
            fn = self.op_patcher.patch(fn)
            return fn(*args_tail, **kwargs)
        elif kind == 'call_module':
            fn = self.fetch_attr(target)
            if hasattr(fn, '__globals__'):
                _autowrap_check(self.patcher, fn.__globals__, self._autowrap_function_ids)
            fn = self.op_patcher.patch(fn)
            return fn(*args, **kwargs)
        elif kind == 'get_attr':
            return self.fetch_attr(target)
        elif kind == 'output':
            return args[0]
        elif kind == 'placeholder':
            return self.placeholder_dict[target]
        else:
            raise RuntimeError()

    @compatibility(is_backward_compatible=True)
    def proxy(self, value: Any, node: Node) -> ep.ConcreteProxy:
        """
        overloaded to use custom 'proxy'.
        """
        return ep.ConcreteProxy(node, value, self)

    @compatibility(is_backward_compatible=True)
    def create_proxy(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any],
                     name: Optional[str] = None, type_expr: Optional[Any] = None):
        """
        similar to _symbolic_trace.Tracer.create_proxy.
        use the 'run_target' to actually execute the code, and store the value in 'value' field.
        """
        def upwrapper(obj: Any):
            if isinstance(obj, ep.ConcreteProxy):
                return obj.value
            return obj
        args_unwrapped = ep.map_aggregate_not_proxy(args, upwrapper)
        kwargs_unwrapped = ep.map_aggregate_not_proxy(kwargs, upwrapper)

        # real value by execution
        value_unwrapped = self.run_target(kind, target, args_unwrapped, kwargs_unwrapped)

        args_noded = self.create_arg(args)
        kwargs_noded = self.create_arg(kwargs)

        assert isinstance(args_noded, tuple)
        assert isinstance(kwargs_noded, dict)

        node = self.create_node(kind, target, args_noded, kwargs_noded, name, type_expr)
        return self.proxy(value_unwrapped, node)

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> Node:
        """
        similar to _symbolic_trace.Tracer.create_arg
        move the base case to the top in case the wrapping of the function 'isinstance'
        """
        # base case: we unwrap the Proxy object
        if isinstance(a, ep.ConcreteProxy):
            return a.node

        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node('get_attr', n, (), {})
            raise NameError('parameter is not a member of this module')
        elif isinstance(a, torch.Tensor):
            for n_, p_ in self.root.named_buffers():
                if a is p_:
                    return self.create_node('get_attr', n_, (), {})
        elif isinstance(a, torch.nn.Module):
            for n_, p_ in self.root.named_modules():
                if a is p_:
                    return self.create_node('get_attr', n_, (), {})
        # For NamedTuple instances that appear literally as args, we emit
        # a node to construct the NamedTuple and use that Node as the argument.
        if isinstance(a, tuple) and hasattr(a, '_fields'):
            args = tuple(self.create_arg(elem) for elem in a)
            return self.create_node('call_function', a.__class__, args, {})

        # Tensors do not have a reliable string repr() from which they can be
        # constructed (and we probably don't want to rely on that, either), so
        # for any constant Tensor values we encounter, first search for if they
        # are an attribute of some module in the module hierarchy. If so, emit
        # a get_attr to retrieve that tensor. Otherwise, we'll store away the
        # tensor value into a special attribute on the Module s.t. we can
        # retrieve it with a get_attr.
        if isinstance(a, (torch.Tensor, ScriptObject)):
            qualname: Optional[str] = self.tensor_attrs.get(a)

            # Tensor was not found in the Module hierarchy, stow it away in a
            # special attribute and set the qualname to refer to that
            if not qualname:
                i = 0
                while True:
                    qualname = f'_tensor_constant{i}'
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                self.tensor_attrs[a] = qualname
                setattr(self.root, qualname, a)

            return self.create_node('get_attr', qualname, (), {})

        if type(a) in _proxyable_classes:
            # This is an instance of a proxyable class for which we did not
            # witness its construction. Intern this as a constant attribute

            # TODO: binary search
            i = 0
            while True:
                qualname = f'_{a.__class__.__name__}_constant_{i}'
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)

            return self.create_node('get_attr', qualname, (), {})

        return super().create_arg(a)

    @compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        similar to _symbolic_trace.Tracer.is_leaf_module
        """
        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)

    @compatibility(is_backward_compatible=True)
    def path_of_module(self, mod: torch.nn.Module) -> str:
        """
        similar to _symbolic_trace.Tracer.path_of_module
        """
        # Prefer the O(1) algorithm
        if self.submodule_paths:
            path = self.submodule_paths.get(mod)
            if path is None:
                raise NameError('module is not installed as a submodule')
            assert isinstance(path, str)
            return path
        # O(N^2) fallback in the case that we didn't store the submodule
        # paths.
        else:
            for n, p in self.root.named_modules():
                if mod is p:
                    return n
            raise NameError('module is not installed as a submodule')

    @compatibility(is_backward_compatible=True)
    def _module_call(self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """
        similar to _symbolic_trace.Tracer.call_module
        """
        module_qualified_name = self.path_of_module(m)
        if not self.is_leaf_module(m, module_qualified_name):
            return forward(*args, **kwargs)
        else:
            # disable patches
            temp_disable_call = self.temp_disable_call
            self.temp_disable_call = True
            # execute leaf module
            proxy = self.create_proxy('call_module', module_qualified_name, args, kwargs)
            # enable patches
            self.temp_disable_call = temp_disable_call

            return proxy

    def _module_getattr(self, attr_val, parameter_proxy_cache):
        def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        val_proxy = self.create_proxy('get_attr', n, (), {})
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None

        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_parameters(), parameter_proxy_cache)
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy

        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_buffers(), parameter_proxy_cache)
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy

        return attr_val


    # This method will be refactored
    @compatibility(is_backward_compatible=False)
    def create_args_for_root(self, root_fn, is_module, concrete_args=None):
        """
        for wrapping all the parameters of the function with dummy_input.
        in concrete tracer, we need all the parameters input by users.

        todo: this function should be refactored after the same function in torch.fx be refactored.
        """
        # In some cases, a function or method has been decorated with a wrapper
        # defined via ``functools.wraps``. In this case, the outer code object
        # will likely not contain the actual parameters we care about, so unwrap
        # the function to get to the innermost callable.
        fn_for_analysis = inspect.unwrap(root_fn)
        default_value_list = fn_for_analysis.__defaults__
        if default_value_list is None:
            default_value_list = tuple()
        co = fn_for_analysis.__code__
        total_args = co.co_argcount + co.co_kwonlyargcount
        # orig_args = list(co.co_varnames)
        names_iter = iter(co.co_varnames)
        args: List[Any] = []
        more_args = []
        kwargs = {}
        skip_arg_idx = 0
        if is_module:
            if total_args == 0:
                raise RuntimeError('``self`` argument cannot be part of *args expansion!')
            skip_arg_idx = 1
            next(names_iter)  # skip self
            args.append(self.root)

        cnt = 0
        self.placeholder_dict = {}
        def proxy_placeholder(name: str):
            nonlocal cnt
            cnt += 1

            if name in concrete_args:
                self.placeholder_dict[f'{name}_{str(cnt)}'] = concrete_args[name]
            else:
                assert name in default_args
                self.placeholder_dict[f'{name}_{str(cnt)}'] = default_args[name]
            return self.create_proxy('placeholder', f'{name}_{str(cnt)}', (), {})
        arg_names = [next(names_iter) for idx in range(skip_arg_idx, total_args)]
        diff_len = len(arg_names) - len(default_value_list)
        default_args = {arg_names[idx + diff_len]: default_value_list[idx] for idx in range(len(default_value_list))}
        if isinstance(concrete_args, tuple):
            if len(arg_names) != len(concrete_args):
                raise RuntimeError(f"Tracing expected {len(arg_names)} arguments but got {len(concrete_args)} concrete arguments")
            concrete_args = {name: val for name, val in zip(arg_names, concrete_args)}
        args.extend(proxy_placeholder(names) for names in arg_names)


        if hasattr(co, 'co_kwonlyargcount') and (
            co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF):
            # TODO: type annotations for *args and **kwargs
            if co.co_flags & inspect.CO_VARARGS:
                name = '*' + next(names_iter)
                default_args[name] = ()
                more_args = proxy_placeholder(name)
            if co.co_flags & inspect.CO_VARKEYWORDS:
                name = '**' + next(names_iter)
                default_args[name] = {}
                kwargs = proxy_placeholder(name)

        return root_fn, args, more_args, kwargs

    @compatibility(is_backward_compatible=True)
    def trace(self, root: Union[torch.nn.Module, Callable[..., Any]],
                   concrete_args: Optional[Dict[str, Any]],
                   use_operator_patch: bool = True,
                   operator_patch_backlist: List[str] = []) -> Graph:
        """
        similar to _symbolic_trace.Tracer.trace
        different args:
            use_operator_patch:
                the operators 'not/is/is not/in/not in' cannot be wrapped after
                    compiled. so we re-parse the functions, replace these operators
                    with functions 'operator.not_/is_/is_not/contains', then we
                    could wrap and trace these.
                for example: in ``if x is None:``, if x is a proxy, the tracer will
                    never go into the branch, even x is a proxy with value 'None'.
                values:
                true: before executing a func, the func will be patched if the func
                    is not in operator_patch_backlist
                false: before executing a func, the func will be patched if the func
                    is in operator_patch_backlist

            operator_patch_backlist:
                such as '__main__.FooModel' or '__main__.bar_func'. the namespace is
                always needed.
        """
        self.op_patcher = OperatorPatcher(use_operator_patch, operator_patch_backlist)
        if isinstance(root, torch.nn.Module):
            self.root = root

            assert hasattr(
                root, self.traced_func_name
            ), f"traced_func_name={self.traced_func_name} doesn't exist in {type(root).__name__}"

            fn = getattr(root, self.traced_func_name)
            self.submodule_paths = {mod: name for name, mod in root.named_modules()}
        else:
            self.root = torch.nn.Module()
            fn = root

        tracer_cls: Optional[Type['ConcreteTracer']] = getattr(self, '__class__', None)
        self.graph = Graph(tracer_cls=tracer_cls)

        # When we encounter a Tensor value that's not a parameter, we look if it
        # is some other attribute on the model. Construct a dict mapping Tensor
        # values to the qualified name here for efficiency. This is used downstream
        # in create_arg
        self.tensor_attrs: Dict[Union[torch.Tensor, ScriptObject], str] = {}

        def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
            for k, v in m.__dict__.items():
                if isinstance(v, (torch.Tensor, ScriptObject)):
                    self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
            for k, v in m.named_children():
                collect_tensor_attrs(v, prefix_atoms + [k])

        collect_tensor_attrs(self.root, [])

        if isinstance(fn, MethodType):
            fn = fn.__func__
        assert isinstance(fn, FunctionType)

        fn_globals = fn.__globals__  # run before it gets patched
        fn, args, more_args, kwargs = self.create_args_for_root(fn, isinstance(root, torch.nn.Module), concrete_args)
        # print('args:', args)
        fn = self.op_patcher.patch(fn)

        parameter_proxy_cache: Dict[str, ep.ConcreteProxy] = {}  # Reduce number of get_attr calls

        # Method dispatch on parameters is not recorded unless it's directly used.
        # Thus, we need to insert a proxy when __getattr__ requests a parameter.
        @functools.wraps(_orig_module_getattr)
        def module_getattr_wrapper(mod, attr):
            attr_val = _orig_module_getattr(mod, attr)
            if self.temp_disable_call | self.temp_disable_attr:
                return attr_val
            else:
                return self._module_getattr(attr_val, parameter_proxy_cache)

        @functools.wraps(_orig_module_call)
        def module_call_wrapper(mod, *args, **kwargs):
            def forward(*args, **kwargs):
                return _orig_module_call(mod, *args, **kwargs)

            if self.temp_disable_call:
                return _orig_module_call(mod, *args, **kwargs)
            else:
                _autowrap_check(self.patcher, getattr(getattr(mod, "forward", mod), "__globals__", {}),
                                self._autowrap_function_ids)
                return self._module_call(mod, forward, args, kwargs)

        @functools.wraps(_orig_bool)
        def bool_wrapper(obj):
            if isinstance(obj, ep.ConcreteProxy):
                return obj.tracer.create_proxy('call_function',
                    _orig_bool, (obj,),
                    {})
            else:
                return _orig_bool(obj)

        @functools.wraps(_orig_tuple)
        def tuple_wrapper(*args, **kwargs):
            if _orig_len(args) != 0 and isinstance(args[0], ep.ConcreteProxy):
                return args[0].tracer.create_proxy('call_function',
                    _orig_tuple, args, kwargs)
            else:
                return _orig_tuple(*args, **kwargs)

        @functools.wraps(_orig_list)
        def list_wrapper(*args, **kwargs):
            if _orig_len(args) != 0 and isinstance(args[0], ep.ConcreteProxy):
                return args[0].tracer.create_proxy('call_function',
                    _orig_list, args, kwargs)
            else:
                return _orig_list(*args, **kwargs)

        @functools.wraps(_orig_set)
        def set_wrapper(*args, **kwargs):
            if _orig_len(args) != 0 and isinstance(args[0], ep.ConcreteProxy):
                return args[0].tracer.create_proxy('call_function',
                    _orig_set, args, kwargs)
            else:
                return _orig_set(*args, **kwargs)

        @functools.wraps(_orig_frozenset)
        def frozenset_wrapper(*args, **kwargs):
            if _orig_len(args) != 0 and isinstance(args[0], ep.ConcreteProxy):
                return args[0].tracer.create_proxy('call_function',
                    _orig_frozenset, args, kwargs)
            else:
                return _orig_frozenset(*args, **kwargs)

        @functools.wraps(_orig_dict)
        def dict_wrapper(*args, **kwargs):
            if _orig_len(args) != 0 and isinstance(args[0], ep.ConcreteProxy):
                return args[0].tracer.create_proxy('call_function',
                    _orig_dict, args, kwargs)
            else:
                return _orig_dict(*args, **kwargs)

        @functools.wraps(_orig_len)
        def len_wrapper(obj) -> int:
            if isinstance(obj, ep.ConcreteProxy):
                return obj.tracer.create_proxy('call_function',
                    _orig_len, (obj,),
                    {})
            else:
                return _orig_len(obj)

        @functools.wraps(_orig_not)
        def not_wrapper(obj):
            if isinstance(obj, ep.ConcreteProxy):
                return obj.tracer.create_proxy('call_function',
                    _orig_not, (obj,),
                    {})
            else:
                return _orig_not(obj)

        @functools.wraps(_orig_is)
        def is_wrapper(obj_a, obj_b):
            if isinstance(obj_a, ep.ConcreteProxy):
                return obj_a.tracer.create_proxy('call_function',
                    _orig_is, (obj_a, obj_b),
                    {})
            else:
                return _orig_is(obj_a, obj_b)

        @functools.wraps(_orig_is_not)
        def is_not_wrapper(obj_a, obj_b):
            if isinstance(obj_a, ep.ConcreteProxy):
                return obj_a.tracer.create_proxy('call_function',
                    _orig_is_not, (obj_a, obj_b),
                    {})
            else:
                return _orig_is_not(obj_a, obj_b)

        @functools.wraps(_orig_contains)
        def contains_wrapper(obj_a, obj_b):
            # 'obj_a in obj_b' ==> 'contains(obj_b obj_a)'
            if isinstance(obj_b, ep.ConcreteProxy):
                return obj_b.tracer.create_proxy('call_function',
                    _orig_contains, (obj_a, obj_b),
                    {})
            else:
                return _orig_contains(obj_a, obj_b)

        @functools.wraps(_orig_isinstance)
        def isinstance_wrapper(instance, clz):
            type_wrappers = {
                bool_wrapper:       _orig_bool,
                list_wrapper:       _orig_list,
                tuple_wrapper:      _orig_tuple,
                set_wrapper:        _orig_set,
                frozenset_wrapper:  _orig_frozenset,
                dict_wrapper:       _orig_dict,
            }
            if type(clz) in (tuple, list, slice, _orig_tuple, _orig_list):
                clz_wappers = []
                for type_wrapper, orig_type in type_wrappers.items():
                    if type_wrapper in clz:
                        clz_wappers.append(orig_type)
                clz = (*clz_wappers, *(aclz for aclz in clz if aclz not in type_wrappers))
                # use _orig_isinstance(clz, Iterable) will cause an endless recursive loop
                for cls in (object, ep.ConcreteProxy, ep.ConcreteAttrProxy, ep.ConcreteUnpackIterProxy):
                    if cls in clz and _orig_isinstance(instance, cls):
                        return True
                if _orig_isinstance(instance, ep.ConcreteProxy):
                    return _orig_isinstance(instance.value, clz)
                else:
                    return _orig_isinstance(instance, clz)
            else:
                if clz in type_wrappers:
                    return _orig_isinstance(instance, type_wrappers[clz])
                elif clz in (object, ep.ConcreteProxy, ep.ConcreteAttrProxy, ep.ConcreteUnpackIterProxy):
                    return _orig_isinstance(instance, clz)
                elif _orig_isinstance(instance, ep.ConcreteProxy):
                    return _orig_isinstance(instance.value, clz)
                else:
                    return _orig_isinstance(instance, clz)

        # for passing the tracing of leaf modules
        self.temp_disable_call = False
        self.temp_disable_attr = False
        with _Patcher() as self.patcher:
            # allow duplicate patches to support the case of nested calls
            self.patcher.patch_method(torch.nn.Module, "__getattr__", module_getattr_wrapper, deduplicate=False)
            self.patcher.patch_method(torch.nn.Module, "__call__", module_call_wrapper, deduplicate=False)
            self.patcher.patch_method(builtins, "isinstance", isinstance_wrapper, deduplicate=False)
            self.patcher.patch_method(builtins, "bool", bool_wrapper, deduplicate=False)
            # self.patcher.patch_method(builtins, "slice", slice_wrapper, deduplicate=False)
            self.patcher.patch_method(builtins, "tuple", tuple_wrapper, deduplicate=False)
            self.patcher.patch_method(builtins, "list", list_wrapper, deduplicate=False)
            self.patcher.patch_method(builtins, "set", set_wrapper, deduplicate=False)
            self.patcher.patch_method(builtins, "frozenset", frozenset_wrapper, deduplicate=False)
            self.patcher.patch_method(builtins, "dict", dict_wrapper, deduplicate=False)
            self.patcher.patch_method(builtins, "len", len_wrapper, deduplicate=False)
            self.patcher.patch_method(operator, "not_", not_wrapper, deduplicate=False)
            self.patcher.patch_method(operator, "is_", is_wrapper, deduplicate=False)
            self.patcher.patch_method(operator, "is_not", is_not_wrapper, deduplicate=False)
            self.patcher.patch_method(operator, "contains", contains_wrapper, deduplicate=False)
            _patch_wrapped_functions(self.patcher)
            _autowrap_check(self.patcher, fn_globals, self._autowrap_function_ids)
            for module in self._autowrap_search:
                _autowrap_check(self.patcher, module.__dict__, self._autowrap_function_ids)
            self.create_node('output', 'output', (self.create_arg(fn(*args, *more_args, **kwargs)),), {},
                             type_expr=fn.__annotations__.get('return', None))

        self.submodule_paths = None

        return self.graph

# List of pairs of (global dict, function name) functions
# to patch for the purposes of the wrap() API.
_wrapped_fns_to_patch : List[Tuple[dict, str]] = []

# List of methods on classes to wrap (class type, function name)
# this currently only works for Tensor.* methods that aren't traced properly
_wrapped_methods_to_patch : List[Tuple[type, str]] = []


def _find_proxy(*objects_to_search):
    """
    Recursively search a data structure for a Proxy() and return it,
    return None if not found.
    """
    proxy = None

    def find_proxy(x):
        nonlocal proxy
        if isinstance(x, ep.ConcreteProxy):
            proxy = x

    ep.map_aggregate_not_proxy(objects_to_search, find_proxy)
    return proxy

def _create_wrapped_func(orig_fn):
    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        """
        Given an closed-over ``orig_function`` to invoke, search the args and kwargs for
        a Proxy object. If there is one, emit a ``call_function`` node to preserve the
        call to this leaf function directly. Otherwise, just return the results of
        this function call, as this function is not being traced.
        """
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return_proxy = proxy.tracer.create_proxy('call_function', orig_fn, args, kwargs)
            return_proxy.node.meta['is_wrapped'] = True
            return return_proxy
        return orig_fn(*args, **kwargs)

    return wrapped

def _patch_wrapped_functions(patcher : _Patcher):
    """
    Go through ``_wrapped_fn_patch_table`` and, for each frame object, wrap
    the listed global functions in the `_create_wrapped_func` wrapper.
    """
    for frame_dict, name in _wrapped_fns_to_patch:
        if name not in frame_dict and hasattr(builtins, name):
            orig_fn = getattr(builtins, name)
        else:
            orig_fn = frame_dict[name]
        patcher.patch(frame_dict, name, _create_wrapped_func(orig_fn))

    for cls, name in _wrapped_methods_to_patch:
        patcher.patch_method(cls, name, _create_wrapped_method(cls, name))


def _autowrap_check(patcher : _Patcher, frame_dict : Dict[str, Any], function_ids : Set[int]):
    """
    Some methods, like `math.sqrt` are common enough we want to automatically wrap them as we see them.
    This method searches a scope for them and patches them if found.
    """
    if patcher.visit_once(frame_dict):
        for name, value in frame_dict.items():
            if not name.startswith("_") and callable(value) and id(value) in function_ids:
                patcher.patch(frame_dict, name, _create_wrapped_func(value))


def _create_wrapped_method(cls, name):
    orig_fn = getattr(cls, name)

    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        """
        Search the args and kwargs for a Proxy object. If there is one,
        emit a ``call_method`` node to preserve the call to this method
        directly. Otherwise, just return the results of this function
        call, as this function is not being traced.
        """
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return proxy.tracer.create_proxy('call_method', name, args, kwargs)
        return orig_fn(*args, **kwargs)

    return wrapped


@compatibility(is_backward_compatible=True)
class GraphAppendingConcreteTracer(ConcreteTracer):
    def __init__(self, graph: Graph):
        super().__init__()
        self.graph = graph


class MagicMethodPatcher:
    magic_methods_ori = fx_graph.magic_methods
    magic_methods_new = {
        **fx_graph.magic_methods,
        'not_': 'not {}',
        'is_': '{} is {}',
        'is_not': '{} is not {}',
        'contains': '{1} in {0}',
    }

    def __enter__(self):
        fx_graph.magic_methods = self.magic_methods_new

    def __exit__(self, exc_type, exc_value, tb):
        fx_graph.magic_methods = self.magic_methods_ori
        return exc_type is None


def concrete_trace(root : Union[torch.nn.Module, Callable[..., Any]],
                   concrete_args: Optional[Dict[str, Any]],
                   use_function_patch: bool = True,
                   function_patch_backlist: List[str] = []) -> GraphModule:
    tracer = ConcreteTracer()
    graph = tracer.trace(root, concrete_args, use_function_patch, function_patch_backlist)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    with MagicMethodPatcher():
        traced = GraphModule(tracer.root, graph, name)
    return traced