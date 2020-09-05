import ast
import inspect
import os

from .mutator import Mutator
from ..operations import Operation
from ..translate_code import inspect_module_args

def deprecated_module_to_ir(module):
    """
    Use m.__repr__() to retrieve type and inputs arguments of the module
    TODO: module cannot have submodule, which should be resolved in future

    fatal blocking: Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1), padding_mode=reflect)
    here `reflect` should be string

    Parameters
    ----------
    module : torch.nn.Module
    """
    print('zql str: ', module.__repr__())
    tree = ast.parse(module.__repr__())
    assert len(tree.body) == 1
    assert isinstance(tree.body[0].value, ast.Call)
    call = tree.body[0].value
    _type = call.func.id
    params = {}
    # get args, no key name specified, key name is retrieved through inspect
    argspec = inspect.getfullargspec(type(module))
    args_name = argspec.args
    for i, arg in enumerate(call.args):
        params[args_name[i+1]] = ast.literal_eval(arg)
    # get keywords
    for keyword in call.keywords:
        print('zql value: ', keyword.value)
        params[keyword.arg] = ast.literal_eval(keyword.value)
    return _type, params

def module_to_ir(module):
    params = inspect_module_args(module)
    _type = type(module).__name__
    return _type, params

class OperatorMutator(Mutator):
    def __init__(self, target: str, candidates: 'List[torch.nn.Module]', type='name'):
        self.target = target
        parsed_candidates = []
        for each in candidates:
            _type, params = module_to_ir(each)
            parsed_candidates.append({'type': _type, 'params': params})
        self.candidates = parsed_candidates
        self.type = type
        assert self.type == 'name'

    def retrieve_targeted_graph(self, graph: 'Graph') -> 'Graph':
        return graph.find_node(self.target)

    def mutate(self, graph):
        target_node = self.retrieve_targeted_graph(graph)
        new_op = self.choice(self.candidates)
        target_node.update_operation(new_op['type'], **new_op['params'])

class InputMutator(Mutator):
    def mutate(self):
        pass

class InsertMutator(Mutator):
    def mutate(self):
        pass

class HpMutator(Mutator):
    def mutate(self):
        pass

class ModuleMutator(Mutator):
    def __init__(self, target: str, args_choices: 'List'):
        self.target = target
        self.args_choices = args_choices

    def retrieve_targeted_graph(self, graph: 'Graph') -> 'Graph':
        return graph.find_node(self.target)

    def mutate(self, graph):
        target_node = self.retrieve_targeted_graph(graph)
        #new_args = self.choice(self.args_choices)
        #target_node.update_operation(None, **new_args)
        target_node.update_operation_super(self.args_choices)
