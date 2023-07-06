# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .concrete_tracer import ConcreteTracer

import ast
import builtins
import inspect
import logging
import platform

from textwrap import dedent
from types import MethodType, FunctionType
from typing import List, Optional, Callable, Dict

import torch

from .utils import (
    _orig_type,
    _orig_isinstance,
    _orig_len,
    _orig_dict,
    _orig_zip,
    _orig_tuple,
)

_logger = logging.getLogger(__name__)

class TransformerOp(ast.NodeTransformer):
    """
    An ast transformer, to check and replace the python ops 'not/is/is not/in/not in' to functions in 'operator' module.
    """

    def visit_start(self, node):
        # to mark if the ast is changed
        self.is_transformed = False

        # detect the expr now is in a branch test expr
        # 0: not in a branch test expr.
        # 1: in propagate if not in func 'visit', or not in a branch test expr in func 'visit'
        # 2: in a branch test expr
        self.is_incond_status = 0
        ret = super().visit(node)
        return self.is_transformed, ret

    def visit(self, node):
        if self.is_incond_status != 0:
            # if the status is 'in branch test',
            self.is_incond_status -= 1
        return super().visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == 'super' and _orig_len(node.args) == 0:
            return self.generic_visit(ast.Call(
                func=ast.Name(id='super', ctx=ast.Load()),
                args=[
                    ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='__class__', ctx=ast.Load()),
                    ast.Name(id='self', ctx=ast.Load()),
                ],
                keywords=node.keywords,
            ))
        elif not isinstance(node.func, ast.Name) or node.func.id != 'patch_run':
            self.is_transformed = True
            return self.generic_visit(ast.Call(
                func=ast.Name(id='patch_run', ctx=ast.Load()),
                args=[node.func, *node.args],
                keywords=node.keywords,
            ))
        else:
            return self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self.is_incond_status = 2
        node.test = self.visit(node.test)
        self.is_incond_status = 0
        node.body = [self.visit(item) for item in node.body]
        node.orelse = [self.visit(item) for item in node.orelse]
        return node

    def visit_If(self, node: ast.If):
        self.is_incond_status = 2
        node.test = self.visit(node.test)
        self.is_incond_status = 0
        node.body = [self.visit(item) for item in node.body]
        node.orelse = [self.visit(item) for item in node.orelse]
        return node

    def visit_IfExp(self, node: ast.IfExp):
        node.body = self.visit(node.body)
        self.visit(node.body)
        self.is_incond_status = 2
        node.test = self.visit(node.test)
        self.is_incond_status = 0
        node.orelse = self.visit(node.orelse)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if self.is_incond_status != 0:
            # in branch cond test expr, need no replacement
            self.is_incond_status = 2
            return self.generic_visit(node)
        elif _orig_isinstance(node.op, ast.Not):
            self.is_transformed = True
            return self.generic_visit(ast.Call(
                func=ast.Name(id='not_', ctx=ast.Load()),
                args=[node.operand],
                keywords=[],
            ))
        else:
            return self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        if self.is_incond_status != 0:
            # in branch cond test expr, need no replacement
            self.is_incond_status = 2
            return self.generic_visit(node)
        else:
            if not _orig_isinstance(node.values[1], (ast.Call, ast.BoolOp)):
                _logger.warning('warning: "and/or" will generate branch expr. The 2nd arg can\'t be traced if the 1st arg returns a True.'
                                ' Don\'t mix up "and/or" and "&/|"!')
            return self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        should_replace = False
        for op in node.ops:
            if _orig_type(op) in (ast.Is, ast.IsNot, ast.In, ast.NotIn):
                should_replace = True
                break
        if should_replace:
            if _orig_len(node.ops) != 1:
                raise RuntimeError(
                    'not supported in "{} cmp_op {} cmp_op {}" when cmp_op contains "is/is not/in/not in"')
            self.is_transformed = True
            func_id = {
                ast.Is: 'is_',
                ast.IsNot: 'is_not',
                ast.In: 'contains',
                ast.NotIn: 'contains',
            }[_orig_type(node.ops[0])]
            if _orig_isinstance(node.ops[0], (ast.In, ast.NotIn)):
                args = [node.comparators[0], node.left]
            else:
                args = [node.left, node.comparators[0]]
            ret_node = ast.Call(
                func=ast.Name(id=func_id, ctx=ast.Load()),
                args=args,
                keywords=[],
            )
            if _orig_isinstance(node.ops[0], ast.NotIn):
                ret_node = ast.Call(
                    func=ast.Name(id='not_', ctx=ast.Load()),
                    args=[ret_node],
                    keywords=[],
                )
            return self.generic_visit(ret_node)
        else:
            return self.generic_visit(node)


class OperatorPatcher:
    """
    An function patcher, to patch the un-wrappable operator 'not/is/is not/in/not in' to wrappable functions.
    """

    transformer_op = TransformerOp()

    def __init__(self, use_operator_patch: bool, operator_patch_backlist: List[str]):
        self.use_operator_patch = use_operator_patch
        self.operator_patch_backlist = operator_patch_backlist
        self.function_cache: Dict[int, Callable] = {}
        self.function_cache_orig: Dict[int, Callable] = {}

    def patch_inner(self, func):
        if _orig_isinstance(func, torch.nn.Module):
            return self.patch_inner_helper(func)    # better not cache this
        if id(func) not in self.function_cache:
            self.function_cache[id(func)] = self.patch_inner_helper(func)
            self.function_cache_orig[id(func)] = func
        return self.function_cache[id(func)]

    def patch_inner_helper(self, func):
        if not hasattr(func, '__module__') or func.__module__ is None or func.__module__.startswith('torch'):
            return func
        if hasattr(func, '_Patcher__fx_already_patched'):
            return func
        if self.use_operator_patch == (func in self.operator_patch_backlist):
            return func
        if _orig_isinstance(func, torch.nn.Module):
            func = func.forward
        if _orig_isinstance(func, MethodType):
            func_inner = func.__func__
            the_self = func.__self__
        else:
            func_inner = func
            the_self = None
        if not _orig_isinstance(func_inner, FunctionType) or not hasattr(func_inner, '__code__'):
            return func

        lines, lnum = inspect.findsource(func_inner)
        # align with original source code
        source = ''.join(('\n' * lnum, *inspect.getblock(lines[lnum:])))
        dedent_src = dedent(source)
        tree = ast.parse(dedent_src)

        is_transformed, new_tree = OperatorPatcher.transformer_op.visit_start(tree)
        if not is_transformed:
            return func
        else:
            body0: ast.FunctionDef = new_tree.body[0]
            body0.body = [
                # equals to:
                # from operator import not_, is_, is_not, contains
                ast.ImportFrom(
                    module='operator',
                    names=[
                        ast.alias(name='not_'),
                        ast.alias(name='is_'),
                        ast.alias(name='is_not'),
                        ast.alias(name='contains'),
                    ],
                    level=0
                ),
                *body0.body
            ]
            body0.name = 'new_func'
            # for deleting some annotations like 'add_start_docstrings_to_model_forward' or 'add_code_sample_docstrings'
            body0.decorator_list = [i for i in body0.decorator_list
                if isinstance(i, ast.Call) and isinstance(i.func, ast.Name) and i.func.id == 'patch_run' and
                    isinstance(i.args[0], ast.Name) and
                    i.args[0].id not in ('add_start_docstrings_to_model_forward', 'add_code_sample_docstrings')]
            ast.fix_missing_locations(new_tree)

            # closure info
            closure_dict = {}
            closures = func_inner.__closure__
            co_freevars = func_inner.__code__.co_freevars
            if (closures != None and _orig_len(closures) != 0) or _orig_len(co_freevars) != 0:
                assert _orig_len(closures) == _orig_len(co_freevars)
                closure_dict = _orig_dict(_orig_zip(co_freevars, [c.cell_contents for c in closures]))

            tuple_wrapped = tuple
            try:
                if platform.python_version_tuple() < ('3', '9'):
                    setattr(builtins, 'tuple', _orig_tuple)
                var_dict = {}
                exec(
                    # use func.__code__.co_filename to make the new function easily debuggable.
                    compile(new_tree, func_inner.__code__.co_filename, 'exec'),
                    {
                        'patch_run': OperatorPatcherContext.patch_run,
                        **func_inner.__globals__,
                        **closure_dict,
                    },
                    var_dict)
                if the_self is not None:
                    return var_dict['new_func'].__get__(the_self)
                else:
                    return var_dict['new_func']
            finally:
                if platform.python_version_tuple() < ('3', '9'):
                    setattr(builtins, 'tuple', tuple_wrapped)

class OperatorPatcherContext:
    ctx_tracer: Optional['ConcreteTracer'] = None
    ctx_patcher: Optional[OperatorPatcher] = None

    def __init__(self, tracer: 'ConcreteTracer', use_operator_patch: bool, operator_patch_backlist: List[str]):
        self.tracer = tracer
        self.patcher = OperatorPatcher(use_operator_patch, operator_patch_backlist)

    def __enter__(self):
        assert OperatorPatcherContext.ctx_tracer is None
        assert OperatorPatcherContext.ctx_patcher is None
        OperatorPatcherContext.ctx_tracer = self.tracer
        OperatorPatcherContext.ctx_patcher = self.patcher

    def __exit__(self, exc_type, exc_value, tb):
        assert OperatorPatcherContext.ctx_tracer == self.tracer
        OperatorPatcherContext.ctx_tracer = None
        OperatorPatcherContext.ctx_patcher = None
        return exc_type is None

    @staticmethod
    def patch_run(func, *args, **kwargs):
        assert OperatorPatcherContext.ctx_tracer is not None
        assert OperatorPatcherContext.ctx_patcher is not None
        with OperatorPatcherContext.ctx_tracer.do_temp_disable(True, True, True):
            new_func = OperatorPatcherContext.ctx_patcher.patch_inner(func)
        return new_func(*args, **kwargs)
