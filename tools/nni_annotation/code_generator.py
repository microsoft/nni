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
from nni_cmd.common_utils import print_warning

# pylint: disable=unidiomatic-typecheck

layer_dict_name = 'nni_layer_info'


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

    assert len(
        call.keywords) == 1, 'Annotation function contains more than one keyword argument'
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
    assert type(
        arg.func) is ast.Attribute, 'nni.variable value is not a NNI function'
    assert type(
        arg.func.value) is ast.Name, 'nni.variable value is not a NNI function'
    assert arg.func.value.id == 'nni', 'nni.variable value is not a NNI function'

    name_str = astor.to_source(name).strip()
    keyword_arg = ast.keyword(arg='name', value=ast.Str(s=name_str))
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
    call.keywords[0].value = ast.Str(s=name_str)

    return call, funcs


def convert_args_to_dict(call, with_lambda=False):
    """Convert all args to a dict such that every key and value in the dict is the same as the value of the arg.
    Return the AST Call node with only one arg that is the dictionary
    """
    keys, values = list(), list()
    for arg in call.args:
        if type(arg) in [ast.Str, ast.Num]:
            arg_value = arg
        else:
            # if arg is not a string or a number, we use its source code as the key
            arg_value = astor.to_source(arg).strip('\n"')
            arg_value = ast.Str(str(arg_value))
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
            if k in ('lineno', 'col_offset', 'ctx'):
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
    assert type(
        node) is ast.Assign, 'nni.variable is not annotating assignment expression'
    assert len(
        node.targets) == 1, 'Annotated assignment has more than one left-hand value'
    name, expr = parse_nni_variable(annotation)
    assert test_variable_equal(
        node.targets[0], name), 'Annotated variable has wrong name'
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

# New code added


def parse_architecture(string, layer_dict_initialized):
    return_node_list = list()
    # fix indentation
    first_char_index = 0
    for char in string.split('\n')[0]:
        if char in [' ', '\t']:
            first_char_index += 1
    strings = [line[first_char_index:] for line in string.split('\n')]
    platform = strings.pop(1).split(':')[1].strip().strip(',')
    assert platform in ['tensorflow', 'others'], "Platform:'%s' must be 'tensorflow' or 'others'" % platform
    dict_node = ast.parse('\n'.join(strings))
    dict_node = NameReplacer().visit(dict_node)
    dict_node = dict_node.body[0].value
    if not layer_dict_initialized:
        initialization = ast.parse(layer_dict_name+"=dict()").body[0]
        return_node_list.append(initialization)

    def get_update_dict_node(content):
        '''update nni_layer_info dict with the content node'''
        call_node = make_attr_call(layer_dict_name, 'update', [content])
        return ast.Expr(value=call_node)
    return_node_list.append(get_update_dict_node(dict_node))
    # store locals and globals
    return_node_list.append(ast.parse('_nni_locals=locals()').body[0])
    return_node_list.append(ast.parse('_nni_globals=globals()').body[0])
    #layer_names = {layer_name.s: eval(layer_name.s) for layer_name in dict_node.keys}
    return_node_list.extend(make_nodes_for_each_layer(dict_node, platform))

    return (*return_node_list,)


def make_call(func, args=[], keywords=[]):
    '''generate an call with func:str, args:list, keywords:list'''
    return ast.Call(func=ast.Name(id=func), args=args, keywords=keywords)


def make_attr_call(attr_name, attr_attr, args=[], keywords=[]):
    '''generate an attribute call with attr_name:str, attr_attr:str and args:list'''
    attribute = ast.Attribute(value=ast.Name(id=attr_name), attr=attr_attr)
    return ast.Call(func=attribute, args=args, keywords=keywords)


def make_layer_info_node(layer_name, key):
    '''generate an dict node like nni_layer_info['layer_name'][key]'''
    inner_value = ast.Name(id=layer_dict_name, ctx=ast.Load())
    inner_slice = ast.Index(value=ast.Str(s=layer_name))
    inner_dict = ast.Subscript(
        value=inner_value, slice=inner_slice, ctx=ast.Load())

    outer_slice = ast.Index(value=ast.Str(s=key))
    outer_dict = ast.Subscript(
        value=inner_dict, slice=outer_slice, ctx=ast.Store())

    return outer_dict


def eval_items(layername, key, return_dict=False, is_list=True):
    """Eval an item or all items in a list 
    Return the Ast node of the evaluation list
    return_dict: if true, the Ast node will be a dict whose key-value is like {variable_name: variable_value}
    is_list: eval an item or all items in a list
    """
    target = "{}['{}']['{}']".format(layer_dict_name, layername, key)
    if is_list:
        if return_dict:
            template = "%s={item: _nni_locals[item] if item in _nni_locals else _nni_globals[item] for item in %s}" % (
                target, target)
        else:
            template = "%s=[_nni_locals[item] if item in _nni_locals else _nni_globals[item] for item in %s]" % (
                target, target)
    else:
        template = "{0}=_nni_locals[{0}] if {0} in _nni_locals else _nni_globals[{0}]".format(
            target)

    return ast.parse(template).body[0]


def make_nodes_for_each_layer(dict_node, platform):
    def other_node(layer_name, info):
        # left value
        output_node = ast.Name(id=info['outputs'].s)
        # right value
        args = [ast.Name(id=layer_dict_name), ast.Str(layer_name)]
        value_node = make_attr_call('nni', 'get_layer_output', args)
        # assign statement
        assign_node = ast.Assign(targets=[output_node], value=value_node)
        return [assign_node]
    layer_nodes = list()
    for layer_name, info in zip(dict_node.keys, dict_node.values):
        # transform info from ast node to dict
        info = {key.s: val for key, val in zip(info.keys, info.values)}
        layer_name = layer_name.s
        # evaluate all inputs and functions
        layer_nodes.append(ast.parse('locals()').body[0])
        layer_nodes.append(ast.Assign(targets=[make_layer_info_node(
            layer_name, 'input_candidates_str')], value=make_layer_info_node(layer_name, 'input_candidates')))
        layer_nodes.append(ast.Assign(targets=[make_layer_info_node(
            layer_name, 'layer_choice_str')], value=make_layer_info_node(layer_name, 'layer_choice')))
        return_dict = False if platform=="tensorflow" else True
        layer_nodes.append(eval_items(layer_name, 'layer_choice', return_dict=return_dict))
        layer_nodes.append(eval_items(layer_name, 'input_candidates', return_dict=return_dict))
        layer_nodes.append(eval_items(layer_name, 'post_process_outputs', is_list=False))
        # left value
        output_node = ast.Name(id=info['outputs'].s)
        # right value
        args = [ast.Name(id=layer_dict_name), ast.Str(layer_name)]
        if platform=="tensorflow":
            args.append(ast.Name(id="tf"))
        value_node = make_attr_call('nni', 'get_layer_output', args)
        # assign statement
        assign_node = ast.Assign(targets=[output_node], value=value_node)
        layer_nodes.append(assign_node)

    return layer_nodes


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


class NameReplacer(ast.NodeTransformer):
    '''To replace all ast.Name to ast.Str using ast.Name.id'''

    def visit_Name(self, node):
        self.generic_visit(node)
        return ast.Str(node.id)


class Transformer(ast.NodeTransformer):
    """Transform original code to annotated code"""

    def __init__(self):
        self.stack = []
        self.last_line = 0
        self.annotated = False
        self.layer_dict_initialized = False

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

        if string.startswith('@nni.get_next_parameter('):
            return parse_annotation("nni.reload_tf_variable(" + string.split('(')[1])

        if string.startswith('@nni.report_intermediate_result(')  \
                or string.startswith('@nni.report_final_result('):
            # expand annotation string to code
            return parse_annotation(string[1:])

        if string.startswith('@nni.variable(') \
                or string.startswith('@nni.function_choice('):
            # mark that the next expression is annotated
            self.stack[-1] = string[1:]
            return None

        if string.startswith('@nni.architecture'):
            assert string != '@nni.architecture', 'nni.architecture annotation should not be empty'
            nodes = parse_architecture(
                string[len('@nni.architecture')+1:], self.layer_dict_initialized)
            self.layer_dict_initialized = True
            return nodes

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
    import_tensorflow = ast.Import(
        names=[ast.alias(name='tensorflow', asname='tf')])
    nodes = ast_tree.body
    for i, _ in enumerate(nodes):
        if type(nodes[i]) is ast.ImportFrom and nodes[i].module == '__future__':
            last_future_import = i
    nodes.insert(last_future_import + 1, import_nni)
    nodes.insert(last_future_import + 2, import_tensorflow)

    return astor.to_source(ast_tree)
