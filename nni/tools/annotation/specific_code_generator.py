# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import astor
from nni.tools.nnictl.common_utils import print_warning

from .utils import ast_Num, ast_Str, lineno

# pylint: disable=unidiomatic-typecheck

para_cfg = None
prefix_name = None


def parse_annotation_mutable_layers(code, lineno):
    """Parse the string of mutable layers in annotation.
    Return a list of AST Expr nodes
    code: annotation string (excluding '@')
    """
    module = ast.parse(code)
    assert type(module) is ast.Module, 'internal error #1'
    assert len(module.body) == 1, 'Annotation mutable_layers contains more than one expression'
    assert type(module.body[0]) is ast.Expr, 'Annotation is not expression'
    call = module.body[0].value
    nodes = []
    mutable_id = prefix_name + '/mutable_block_' + str(lineno)
    mutable_layer_cnt = 0
    for arg in call.args:
        fields = {'layer_choice': False,
                  'fixed_inputs': False,
                  'optional_inputs': False,
                  'optional_input_size': False,
                  'layer_output': False}
        mutable_layer_id = 'mutable_layer_' + str(mutable_layer_cnt)
        mutable_layer_cnt += 1
        func_call = None
        for k, value in zip(arg.keys, arg.values):
            if k.id == 'layer_choice':
                assert not fields['layer_choice'], 'Duplicated field: layer_choice'
                assert type(value) is ast.List, 'Value of layer_choice should be a list'
                for call in value.elts:
                    assert type(call) is ast.Call, 'Element in layer_choice should be function call'
                    call_name = astor.to_source(call).strip()
                    if call_name == para_cfg[mutable_id][mutable_layer_id]['chosen_layer']:
                        func_call = call
                        assert not call.args, 'Number of args without keyword should be zero'
                        break
                fields['layer_choice'] = True
            elif k.id == 'fixed_inputs':
                assert not fields['fixed_inputs'], 'Duplicated field: fixed_inputs'
                assert type(value) is ast.List, 'Value of fixed_inputs should be a list'
                fixed_inputs = value
                fields['fixed_inputs'] = True
            elif k.id == 'optional_inputs':
                assert not fields['optional_inputs'], 'Duplicated field: optional_inputs'
                assert type(value) is ast.List, 'Value of optional_inputs should be a list'
                var_names = [astor.to_source(var).strip() for var in value.elts]
                chosen_inputs = para_cfg[mutable_id][mutable_layer_id]['chosen_inputs']
                elts = []
                for i in chosen_inputs:
                    index = var_names.index(i)
                    elts.append(value.elts[index])
                optional_inputs = ast.List(elts=elts)
                fields['optional_inputs'] = True
            elif k.id == 'optional_input_size':
                pass
            elif k.id == 'layer_output':
                assert not fields['layer_output'], 'Duplicated field: layer_output'
                assert type(value) is ast.Name, 'Value of layer_output should be ast.Name type'
                layer_output = value
                fields['layer_output'] = True
            else:
                raise AssertionError('Unexpected field in mutable layer')
        # make call for this mutable layer
        assert fields['layer_choice'], 'layer_choice must exist'
        assert fields['layer_output'], 'layer_output must exist'

        if not fields['fixed_inputs']:
            fixed_inputs = ast.List(elts=[])
        if not fields['optional_inputs']:
            optional_inputs = ast.List(elts=[])
        inputs = ast.List(elts=[fixed_inputs, optional_inputs])

        func_call.args.append(inputs)
        node = ast.Assign(targets=[layer_output], value=func_call)
        nodes.append(node)
    return nodes


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
    keyword_arg = ast.keyword(arg='name', value=ast_Str(s=name_str))
    arg.keywords.append(keyword_arg)
    if arg.func.attr == 'choice':
        convert_args_to_dict(arg)

    return name, arg


def parse_nni_function(code):
    """Parse `nni.function_choice` expression.
    Return the AST node of annotated expression and a list of dumped function call expressions.
    code: annotation string
    """
    name, call = parse_annotation_function(code, 'function_choice')
    funcs = [ast.dump(func, False) for func in call.args]
    convert_args_to_dict(call, with_lambda=True)

    name_str = astor.to_source(name).strip()
    call.keywords[0].value = ast_Str(s=name_str)

    return call, funcs


def convert_args_to_dict(call, with_lambda=False):
    """Convert all args to a dict such that every key and value in the dict is the same as the value of the arg.
    Return the AST Call node with only one arg that is the dictionary
    """
    keys, values = list(), list()
    for arg in call.args:
        if type(arg) in [ast_Str, ast_Num]:
            arg_value = arg
        else:
            # if arg is not a string or a number, we use its source code as the key
            arg_value = astor.to_source(arg).strip('\n"')
            arg_value = ast_Str(str(arg_value))
        arg = make_lambda(arg) if with_lambda else arg
        keys.append(arg_value)
        values.append(arg)
    del call.args[:]
    call.args.append(ast.Dict(keys=keys, values=values))

    return call


def make_lambda(call):
    """Wrap an AST Call node to lambda expression node.
    call: ast.Call node
    """
    empty_args = ast.arguments(args=[], vararg=None, kwarg=None, defaults=[])
    return ast.Lambda(args=empty_args, body=call)


def test_variable_equal(node1, node2):
    """Test whether two variables are the same."""
    if type(node1) is not type(node2):
        return False
    if isinstance(node1, ast.AST):
        for k, v in vars(node1).items():
            if k in ('lineno', 'col_offset', 'ctx', 'end_lineno', 'end_col_offset'):
                continue
            if not test_variable_equal(v, getattr(node2, k)):
                return False
        return True
    if isinstance(node1, list):
        if len(node1) != len(node2):
            return False
        return all(test_variable_equal(n1, n2) for n1, n2 in zip(node1, node2))

    return node1 == node2


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
            self.last_line = lineno(node)

        # do nothing for root
        if not self.stack:
            return self._visit_children(node)

        annotation = self.stack[-1]

        # this is a standalone string, may be an annotation
        if type(node) is ast.Expr and type(node.value) is ast_Str:
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

        if string.startswith('@nni.get_next_parameter'):
            deprecated_message = "'@nni.get_next_parameter' is deprecated in annotation due to inconvenience. " \
                                 "Please remove this line in the trial code."
            print_warning(deprecated_message)
            return ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()),
                                           args=[ast_Str(s='Get next parameter here...')], keywords=[]))

        if string.startswith('@nni.training_update'):
            return ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()),
                                           args=[ast_Str(s='Training update here...')], keywords=[]))

        if string.startswith('@nni.report_intermediate_result'):
            module = ast.parse(string[1:])
            arg = module.body[0].value.args[0]
            return ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()),
                                           args=[ast_Str(s='nni.report_intermediate_result: '), arg], keywords=[]))

        if string.startswith('@nni.report_final_result'):
            module = ast.parse(string[1:])
            arg = module.body[0].value.args[0]
            return ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()),
                                           args=[ast_Str(s='nni.report_final_result: '), arg], keywords=[]))

        if string.startswith('@nni.mutable_layers'):
            return parse_annotation_mutable_layers(string[1:], lineno(node))

        if string.startswith('@nni.variable') \
                or string.startswith('@nni.function_choice'):
            self.stack[-1] = string[1:]  # mark that the next expression is annotated
            return None

        raise AssertionError('Unexpected annotation function')

    def _visit_children(self, node):
        self.stack.append(None)
        self.generic_visit(node)
        annotation = self.stack.pop()
        assert annotation is None, 'Annotation has no target'
        return node


def parse(code, para, module):
    """Annotate user code.
    Return annotated code (str) if annotation detected; return None if not.
    code: original user code (str)
    """
    global para_cfg
    global prefix_name
    para_cfg = para
    prefix_name = module
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

    return astor.to_source(ast_tree)
