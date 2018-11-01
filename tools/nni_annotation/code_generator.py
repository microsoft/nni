# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================


import ast

import astor

# pylint: disable=unidiomatic-typecheck


def parse_annotation(code):
    """Parse an annotation string.
    Return an AST Expr node.
    code: annotation string (excluding '@')
    """
    module = ast.parse(code)
    assert type(module) is ast.Module, 'internal error #1'
    assert len(module.body) == 1, 'Annotation contains more than one expression'
    assert type(module.body[0]) is ast.Expr, 'Annotation is not expression'
    return module.body[0]


def parse_annotation_function(code, func_name):
    """Parse an annotation function.
    Return the value of `name` keyword argument and the AST Call node.
    func_name: expected function name
    """
    expr = parse_annotation(code)
    call = expr.value
    assert type(call) is ast.Call, 'Annotation is not a function call'

    assert type(call.func) is ast.Attribute, 'Unexpected annotation function'
    assert type(call.func.value) is ast.Name, 'Invalid annotation function name'
    assert call.func.value.id == 'nni', 'Annotation is not a NNI function'
    assert call.func.attr == func_name, 'internal error #2'

    assert len(call.keywords) == 1, 'Annotation function contains more than one keyword argument'
    assert call.keywords[0].arg == 'name', 'Annotation keyword argument is not "name"'
    name = call.keywords[0].value

    return name, call


def parse_nni_variable(code):
    """Parse `nni.variable` expression.
    Return the name argument and AST node of annotated expression.
    code: annotation string
    """
    name, call = parse_annotation_function(code, 'variable')

    assert len(call.args) == 1, 'nni.variable contains more than one arguments'
    arg = call.args[0]
    assert type(arg) is ast.Call, 'Value of nni.variable is not a function call'
    assert type(arg.func) is ast.Attribute, 'nni.variable value is not a NNI function'
    assert type(arg.func.value) is ast.Name, 'nni.variable value is not a NNI function'
    assert arg.func.value.id == 'nni', 'nni.variable value is not a NNI function'

    name_str = astor.to_source(name).strip()
    keyword_arg = ast.keyword(arg='name', value=ast.Str(s=name_str))
    arg.keywords.append(keyword_arg)

    return name, arg


def parse_nni_function(code):
    """Parse `nni.function_choice` expression.
    Return the AST node of annotated expression and a list of dumped function call expressions.
    code: annotation string
    """
    name, call = parse_annotation_function(code, 'function_choice')
    funcs = [ast.dump(func, False) for func in call.args]
    call.args = [make_lambda(arg) for arg in call.args]

    name_str = astor.to_source(name).strip()
    call.keywords[0].value = ast.Str(s=name_str)

    return call, funcs


def make_lambda(call):
    """Wrap an AST Call node to lambda expression node.
    call: ast.Call node
    """
    assert type(call) is ast.Call, 'Argument of nni.function_choice is not function call'
    empty_args = ast.arguments(args=[], vararg=None, kwarg=None, defaults=[])
    return ast.Lambda(args=empty_args, body=call)


def test_variable_equal(var1, var2):
    """Test whether two variables are the same."""
    if type(var1) is not type(var2):
        return False
    if type(var1) is ast.Name:
        return var1.id == var2.id
    if type(var1) is ast.Attribute:
        return var1.attr == var2.attr and test_variable_equal(var1.value, var2.value)
    return False


def replace_variable_node(node, annotation):
    """Replace a node annotated by `nni.variable`.
    node: the AST node to replace
    annotation: annotation string
    """
    assert type(node) is ast.Assign, 'nni.variable is not annotating assignment expression'
    assert len(node.targets) == 1, 'Annotated assignment has more than one left-hand value'
    name, expr = parse_nni_variable(annotation)
    assert test_variable_equal(node.targets[0], name), 'Annotated variable has wrong name'
    node.value = expr
    return node


def replace_function_node(node, annotation):
    """Replace a node annotated by `nni.function_choice`.
    node: the AST node to replace
    annotation: annotation string
    """
    target, funcs = parse_nni_function(annotation)
    FuncReplacer(funcs, target).visit(node)
    return node


class FuncReplacer(ast.NodeTransformer):
    """To replace target function call expressions in a node annotated by `nni.function_choice`"""

    def __init__(self, funcs, target):
        """Constructor.
        funcs: list of dumped function call expressions to replace
        target: use this AST node to replace matching expressions
        """
        self.funcs = set(funcs)
        self.target = target

    def visit_Call(self, node):  # pylint: disable=invalid-name
        if ast.dump(node, False) in self.funcs:
            return self.target
        return node


class Transformer(ast.NodeTransformer):
    """Transform original code to annotated code"""

    def __init__(self):
        self.stack = []
        self.last_line = 0
        self.annotated = False

    def visit(self, node):
        if isinstance(node, (ast.expr, ast.stmt)):
            self.last_line = node.lineno

        # do nothing for root
        if not self.stack:
            return self._visit_children(node)

        annotation = self.stack[-1]

        # this is a standalone string, may be an annotation
        if type(node) is ast.Expr and type(node.value) is ast.Str:
            # must not annotate an annotation string
            assert annotation is None, 'Annotating an annotation'
            return self._visit_string(node)

        if annotation is not None:  # this expression is annotated
            self.stack[-1] = None  # so next expression is not
            if annotation.startswith('nni.variable'):
                return replace_variable_node(node, annotation)
            if annotation.startswith('nni.function_choice'):
                return replace_function_node(node, annotation)

        return self._visit_children(node)


    def _visit_string(self, node):
        string = node.value.s
        if string.startswith('@nni.'):
            self.annotated = True
        else:
            return node  # not an annotation, ignore it

        if string.startswith('@nni.report_intermediate_result(')  \
                or string.startswith('@nni.report_final_result(') \
                or string.startswith('@nni.get_next_parameter('):
            return parse_annotation(string[1:])  # expand annotation string to code

        if string.startswith('@nni.variable(') \
                or string.startswith('@nni.function_choice('):
            self.stack[-1] = string[1:]  # mark that the next expression is annotated
            return None

        raise AssertionError('Unexpected annotation function')


    def _visit_children(self, node):
        self.stack.append(None)
        self.generic_visit(node)
        annotation = self.stack.pop()
        assert annotation is None, 'Annotation has no target'
        return node


def parse(code):
    """Annotate user code.
    Return annotated code (str) if annotation detected; return None if not.
    code: original user code (str)
    """
    try:
        ast_tree = ast.parse(code)
    except Exception:
        raise RuntimeError('Bad Python code')

    transformer = Transformer()
    try:
        transformer.visit(ast_tree)
    except AssertionError as exc:
        raise RuntimeError('%d: %s' % (ast_tree.last_line, exc.args[0]))

    if not transformer.annotated:
        return None

    last_future_import = -1
    import_nni = ast.Import(names=[ast.alias(name='nni', asname=None)])
    nodes = ast_tree.body
    for i, _ in enumerate(nodes):
        if type(nodes[i]) is ast.ImportFrom and nodes[i].module == '__future__':
            last_future_import = i
    nodes.insert(last_future_import + 1, import_nni)

    return astor.to_source(ast_tree)
