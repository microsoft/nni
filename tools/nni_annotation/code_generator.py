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

def parse_annotation_mutable_layers(code, lineno, nas_mode):
    """Parse the string of mutable layers in annotation.
    Return a list of AST Expr nodes
    code: annotation string (excluding '@')
    nas_mode: the mode of NAS
    """
    module = ast.parse(code)
    assert type(module) is ast.Module, 'internal error #1'
    assert len(module.body) == 1, 'Annotation mutable_layers contains more than one expression'
    assert type(module.body[0]) is ast.Expr, 'Annotation is not expression'
    call = module.body[0].value
    nodes = []
    mutable_id = 'mutable_block_' + str(lineno)
    mutable_layer_cnt = 0
    for arg in call.args:
        fields = {'layer_choice': False,
                  'fixed_inputs': False,
                  'optional_inputs': False,
                  'optional_input_size': False,
                  'layer_output': False}
        for k, value in zip(arg.keys, arg.values):
            if k.id == 'layer_choice':
                assert not fields['layer_choice'], 'Duplicated field: layer_choice'
                assert type(value) is ast.List, 'Value of layer_choice should be a list'
                call_funcs_keys = []
                call_funcs_values = []
                call_kwargs_values = []
                for call in value.elts:
                    assert type(call) is ast.Call, 'Element in layer_choice should be function call'
                    call_name = astor.to_source(call).strip()
                    call_funcs_keys.append(ast.Str(s=call_name))
                    call_funcs_values.append(call.func)
                    assert not call.args, 'Number of args without keyword should be zero'
                    kw_args = []
                    kw_values = []
                    for kw in call.keywords:
                        kw_args.append(ast.Str(s=kw.arg))
                        kw_values.append(kw.value)
                    call_kwargs_values.append(ast.Dict(keys=kw_args, values=kw_values))
                call_funcs = ast.Dict(keys=call_funcs_keys, values=call_funcs_values)
                call_kwargs = ast.Dict(keys=call_funcs_keys, values=call_kwargs_values)
                fields['layer_choice'] = True
            elif k.id == 'fixed_inputs':
                assert not fields['fixed_inputs'], 'Duplicated field: fixed_inputs'
                assert type(value) is ast.List, 'Value of fixed_inputs should be a list'
                fixed_inputs = value
                fields['fixed_inputs'] = True
            elif k.id == 'optional_inputs':
                assert not fields['optional_inputs'], 'Duplicated field: optional_inputs'
                assert type(value) is ast.List, 'Value of optional_inputs should be a list'
                var_names = [ast.Str(s=astor.to_source(var).strip()) for var in value.elts]
                optional_inputs = ast.Dict(keys=var_names, values=value.elts)
                fields['optional_inputs'] = True
            elif k.id == 'optional_input_size':
                assert not fields['optional_input_size'], 'Duplicated field: optional_input_size'
                assert type(value) is ast.Num or type(value) is ast.List, 'Value of optional_input_size should be a number or list'
                optional_input_size = value
                fields['optional_input_size'] = True
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
        mutable_layer_id = 'mutable_layer_' + str(mutable_layer_cnt)
        mutable_layer_cnt += 1
        target_call_attr = ast.Attribute(value=ast.Name(id='nni', ctx=ast.Load()), attr='mutable_layer', ctx=ast.Load())
        target_call_args = [ast.Str(s=mutable_id),
                            ast.Str(s=mutable_layer_id),
                            call_funcs,
                            call_kwargs]
        if fields['fixed_inputs']:
            target_call_args.append(fixed_inputs)
        else:
            target_call_args.append(ast.List(elts=[]))
        if fields['optional_inputs']:
            target_call_args.append(optional_inputs)
            assert fields['optional_input_size'], 'optional_input_size must exist when optional_inputs exists'
            target_call_args.append(optional_input_size)
        else:
            target_call_args.append(ast.Dict(keys=[], values=[]))
            target_call_args.append(ast.Num(n=0))
        target_call_args.append(ast.Str(s=nas_mode))
        if nas_mode in ['enas_mode', 'oneshot_mode']:
            target_call_args.append(ast.Name(id='tensorflow'))
        target_call = ast.Call(func=target_call_attr, args=target_call_args, keywords=[])
        node = ast.Assign(targets=[layer_output], value=target_call)
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

    def __init__(self, nas_mode=None):
        self.stack = []
        self.last_line = 0
        self.annotated = False
        self.nas_mode = nas_mode

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

        if string.startswith('@nni.get_next_parameter'):
            call_node = parse_annotation(string[1:]).value
            if call_node.args:
                # it is used in enas mode as it needs to retrieve the next subgraph for training
                call_attr = ast.Attribute(value=ast.Name(id='nni', ctx=ast.Load()), attr='reload_tensorflow_variables', ctx=ast.Load())
                return ast.Expr(value=ast.Call(func=call_attr, args=call_node.args, keywords=[]))

        if string.startswith('@nni.report_intermediate_result')  \
                or string.startswith('@nni.report_final_result') \
                or string.startswith('@nni.get_next_parameter'):
            return parse_annotation(string[1:])  # expand annotation string to code

        if string.startswith('@nni.mutable_layers'):
            nodes = parse_annotation_mutable_layers(string[1:], node.lineno, self.nas_mode)
            return nodes

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


def parse(code, nas_mode=None):
    """Annotate user code.
    Return annotated code (str) if annotation detected; return None if not.
    code: original user code (str),
    nas_mode: the mode of NAS given that NAS interface is used
    """
    try:
        ast_tree = ast.parse(code)
    except Exception:
        raise RuntimeError('Bad Python code')

    transformer = Transformer(nas_mode)
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
    # enas and oneshot modes for tensorflow need tensorflow module, so we import it here
    if nas_mode in ['enas_mode', 'oneshot_mode']:
        import_tf = ast.Import(names=[ast.alias(name='tensorflow', asname=None)])
        nodes.insert(last_future_import + 1, import_tf)

    return astor.to_source(ast_tree)
