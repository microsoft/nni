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

# pylint: disable=unidiomatic-typecheck


# list of functions related to search space generating
_ss_funcs = [
    'choice',
    'randint',
    'uniform',
    'quniform',
    'loguniform',
    'qloguniform',
    'normal',
    'qnormal',
    'lognormal',
    'qlognormal',
    'function_choice'
]


class SearchSpaceGenerator(ast.NodeVisitor):
    """Generate search space from smart parater APIs"""

    def __init__(self, module_name):
        self.module_name = module_name
        self.search_space = {}
        self.last_line = 0  # last parsed line, useful for error reporting

    def visit_Call(self, node):  # pylint: disable=invalid-name
        self.generic_visit(node)

        # ignore if the function is not 'nni.*'
        if type(node.func) is not ast.Attribute:
            return
        if type(node.func.value) is not ast.Name:
            return
        if node.func.value.id != 'nni':
            return

        # ignore if its not a search space function (e.g. `report_final_result`)
        func = node.func.attr
        if func not in _ss_funcs:
            return

        self.last_line = node.lineno

        if node.keywords:
            # there is a `name` argument
            assert len(node.keywords) == 1, 'Smart parameter has keyword argument other than "name"'
            assert node.keywords[0].arg == 'name', 'Smart paramater\'s keyword argument is not "name"'
            assert type(node.keywords[0].value) is ast.Str, 'Smart parameter\'s name must be string literal'
            name = node.keywords[0].value.s
            specified_name = True
        else:
            # generate the missing name automatically
            assert len(node.args) > 0, 'Smart parameter expression has no argument'
            name = '__line' + str(node.args[-1].lineno)
            specified_name = False

        if func in ('choice', 'function_choice'):
            # arguments of `choice` may contain complex expression,
            # so use indices instead of arguments
            args = list(range(len(node.args)))
        else:
            # arguments of other functions must be literal number
            assert all(type(arg) is ast.Num for arg in node.args), 'Smart parameter\'s arguments must be number literals'
            args = [arg.n for arg in node.args]

        key = self.module_name + '/' + name + '/' + func
        if func == 'function_choice':
            func = 'choice'
        value = {'_type': func, '_value': args}

        if specified_name:
            # multiple functions with same name must have identical arguments
            old = self.search_space.get(key)
            assert old is None or old == value, 'Different smart parameters have same name'
        else:
            # generated name must not duplicate
            assert key not in self.search_space, 'Only one smart parameter is allowed in a line'

        self.search_space[key] = value


def generate(module_name, code):
    """Generate search space.
    Return a serializable search space object.
    module_name: name of the module (str)
    code: user code (str)
    """
    try:
        ast_tree = ast.parse(code)
    except Exception:
        raise RuntimeError('Bad Python code')

    visitor = SearchSpaceGenerator(module_name)
    try:
        visitor.visit(ast_tree)
    except AssertionError as exc:
        raise RuntimeError('%d: %s' % (visitor.last_line, exc.args[0]))
    return visitor.search_space
