# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
from sys import version_info


if version_info >= (3, 8):
    ast_Num = ast_Str = ast_Bytes = ast_NameConstant = ast_Ellipsis = ast.Constant

    def lineno(ast_node):
        return ast_node.end_lineno

else:
    ast_Num = ast.Num
    ast_Str = ast.Str
    ast_Bytes = ast.Bytes
    ast_NameConstant = ast.NameConstant
    ast_Ellipsis = ast.Ellipsis

    def lineno(ast_node):
        return ast_node.lineno
