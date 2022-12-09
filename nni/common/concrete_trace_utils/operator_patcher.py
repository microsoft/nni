# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import inspect
import logging

from functools import lru_cache
from textwrap import dedent
from types import MethodType, FunctionType
from typing import List, Optional

import torch


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
            # if the status is 'in brunch test',
            self.is_incond_status -= 1
        return super().visit(node)

    def visit_While(self, node: ast.While):
        self.is_incond_status = 2
        self.visit(node.test)
        self.is_incond_status = 0
        for item in node.body:
            self.visit(item)
        for item in node.orelse:
            self.visit(item)
        return node

    def visit_If(self, node: ast.If):
        self.is_incond_status = 2
        self.visit(node.test)
        self.is_incond_status = 0
        for item in node.body:
            self.visit(item)
        for item in node.orelse:
            self.visit(item)
        return node

    def visit_IfExp(self, node: ast.IfExp):
        self.visit(node.body)
        self.is_incond_status = 2
        self.visit(node.test)
        self.is_incond_status = 0
        self.visit(node.orelse)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if self.is_incond_status != 0:
            # in branch cond test expr, need no replacement
            self.is_incond_status = 2
            return self.generic_visit(node)
        elif isinstance(node.op, ast.Not):
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
            if not isinstance(node.values[1], (ast.Call, ast.BoolOp)):
                _logger.warning('warning: "and/or" will generate branch expr. The 2nd arg can\'t be traced if the 1st arg returns a True.'
                                ' Don\'t mix up "and/or" and "&/|"!')
            return self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        if self.is_incond_status != 0:
            self.is_incond_status = 2
            return self.generic_visit(node)
        should_replace = False
        for op in node.ops:
            if type(op) in (ast.Is, ast.IsNot, ast.In, ast.NotIn):
                should_replace = True
                break
        if should_replace:
            if len(node.ops) != 1:
                raise RuntimeError(
                    'not supported in "{} cmp_op {} cmp_op {}" when cmp_op contains "is/is not/in/not in"')
            self.is_transformed = True
            func_id = {
                ast.Is: 'is_',
                ast.IsNot: 'is_not',
                ast.In: 'contains',
                ast.NotIn: 'contains',
            }[type(node.ops[0])]
            if isinstance(node.ops[0], (ast.In, ast.NotIn)):
                args = [node.comparators[0], node.left]
            else:
                args = [node.left, node.comparators[0]]
            ret_node = ast.Call(
                func=ast.Name(id=func_id, ctx=ast.Load()),
                args=args,
                keywords=[],
            )
            if isinstance(node.ops[0], ast.NotIn):
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
    Todo: patch this operators:
        LIST_TO_TUPLE
        LIST_EXTEND(i)
        SET_UPDATE(i)
        DICT_UPDATE(i)
        DICT_MERGE
    """

    transformer_op = TransformerOp()

    def __init__(self, use_operator_patch: bool, operator_patch_backlist: List[str]):
        self.use_operator_patch = use_operator_patch
        self.operator_patch_backlist = operator_patch_backlist

    @lru_cache
    def patch_inner(self, func):
        if not hasattr(func, '__module__') or func.__module__ is None or func.__module__.startswith('torch'):
            return func
        if self.use_operator_patch == (func in self.operator_patch_backlist):
            return func
        if isinstance(func, torch.nn.Module):
            func = func.forward
        if isinstance(func, MethodType):
            func = func.__func__
        if not isinstance(func, FunctionType) or not hasattr(func, '__code__'):
            return func

        lines, lnum = inspect.findsource(func)
        # align with original source code
        lines_cut_start = ['\n' * lnum, *lines[lnum:]]
        lines_cut_start_end = inspect.getblock(lines_cut_start)

        source = ''.join(lines_cut_start_end)
        dedent_src = dedent(source)
        tree = ast.parse(dedent_src)

        is_transformed, new_tree = self.transformer_op.visit_start(tree)
        if not is_transformed:
            return func
        else:
            new_tree.body[0].body = [
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
                *new_tree.body[0].body
            ]
            new_tree.body[0].name = 'new_func'
            ast.fix_missing_locations(new_tree)
            var_dict = {}
            # use func.__code__.co_filename to make the new function easily debuggable.
            exec(compile(new_tree, func.__code__.co_filename, 'exec'),
                 func.__globals__, var_dict)
            return var_dict['new_func']

class OperatorPatcherContext:
    patcher: Optional[OperatorPatcher] = None

    @staticmethod
    def create(use_operator_patch: bool, operator_patch_backlist: List[str]):
        assert OperatorPatcherContext.patcher is None
        OperatorPatcherContext.patcher = OperatorPatcher(use_operator_patch, operator_patch_backlist)

    @staticmethod
    def patch(func):
        assert OperatorPatcherContext.patcher is not None
        return OperatorPatcherContext.patcher.patch_inner(func)