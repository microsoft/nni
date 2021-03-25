# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Optional, Tuple

from ...mutator import Mutator
from ...graph import Model, Node
from .api import ValueChoice


class LayerChoiceMutator(Mutator):
    def __init__(self, nodes: List[Node]):
        super().__init__()
        self.nodes = nodes

    def mutate(self, model):
        n_candidates = len(self.nodes[0].operation.parameters['candidates'])
        indices = list(range(n_candidates))
        chosen_index = self.choice(indices)
        for node in self.nodes:
            target = model.get_node_by_name(node.name)
            chosen_cand = node.operation.parameters['candidates'][chosen_index]
            target.update_operation(chosen_cand['type'], chosen_cand['parameters'])


class InputChoiceMutator(Mutator):
    def __init__(self, nodes: List[Node]):
        super().__init__()
        self.nodes = nodes

    def mutate(self, model):
        n_candidates = self.nodes[0].operation.parameters['n_candidates']
        n_chosen =  self.nodes[0].operation.parameters['n_chosen']
        candidates = list(range(n_candidates))
        chosen = [self.choice(candidates) for _ in range(n_chosen)]
        for node in self.nodes:
            target = model.get_node_by_name(node.name)
            target.update_operation('__torch__.nni.retiarii.nn.pytorch.ChosenInputs',
                                    {'chosen': chosen, 'reduction': node.operation.parameters['reduction']})


class ValueChoiceMutator(Mutator):
    def __init__(self, nodes: List[Node], candidates: List[Any]):
        super().__init__()
        self.nodes = nodes
        self.candidates = candidates

    def mutate(self, model):
        chosen = self.choice(self.candidates)
        for node in self.nodes:
            target = model.get_node_by_name(node.name)
            target.update_operation('prim::Constant', {'type': type(chosen).__name__, 'value': chosen})


class ParameterChoiceMutator(Mutator):
    def __init__(self, nodes: List[Tuple[Node, str]], candidates: List[Any]):
        super().__init__()
        self.nodes = nodes
        self.candidates = candidates

    def mutate(self, model):
        chosen = self.choice(self.candidates)
        for node, argname in self.nodes:
            target = model.get_node_by_name(node.name)
            target.update_operation(target.operation.type, {**target.operation.parameters, argname: chosen})


def process_inline_mutation(model: Model) -> Optional[List[Mutator]]:
    applied_mutators = []

    lc_nodes = _group_by_label(model.get_nodes_by_type('__torch__.nni.retiarii.nn.pytorch.api.LayerChoice'))
    for node_list in lc_nodes:
        assert _is_all_equal(map(lambda node: len(node.operation.parameters['candidates']), node_list)), \
            'Layer choice with the same label must have the same number of candidates.'
        mutator = LayerChoiceMutator(node_list)
        applied_mutators.append(mutator)

    ic_nodes = _group_by_label(model.get_nodes_by_type('__torch__.nni.retiarii.nn.pytorch.api.InputChoice'))
    for node_list in ic_nodes:
        assert _is_all_equal(map(lambda node: node.operation.parameters['n_candidates'], node_list)) and \
            _is_all_equal(map(lambda node: node.operation.parameters['n_chosen'], node_list)), \
            'Input choice with the same label must have the same number of candidates.'
        mutator = InputChoiceMutator(node_list)
        applied_mutators.append(mutator)

    vc_nodes = _group_by_label(model.get_nodes_by_type('__torch__.nni.retiarii.nn.pytorch.api.ValueChoice'))
    for node_list in vc_nodes:
        assert _is_all_equal(map(lambda node: node.operation.parameters['candidates'], node_list)), \
            'Value choice with the same label must have the same candidates.'
        mutator = ValueChoiceMutator(node_list, node_list[0].operation.parameters['candidates'])
        applied_mutators.append(mutator)

    pc_nodes = []
    for node in model.get_nodes():
        for name, choice in node.operation.parameters.items():
            if isinstance(choice, ValueChoice):
                pc_nodes.append((node, name))
    pc_nodes = _group_parameters_by_label(pc_nodes)
    for node_list in pc_nodes:
        assert _is_all_equal([node.operation.parameters[name].candidates for node, name in node_list]), \
            'Value choice with the same label must have the same candidates.'
        mutator = ParameterChoiceMutator(node_list, node_list[0][0].operation.parameters[node_list[0][1]].candidates)
        applied_mutators.append(mutator)

    if applied_mutators:
        return applied_mutators
    return None


def _is_all_equal(lst):
    last = None
    for x in lst:
        if last is not None and last != x:
            return False
        last = x
    return True


def _group_by_label(nodes: List[Node]) -> List[List[Node]]:
    result = {}
    for node in nodes:
        label = node.operation.parameters['label']
        if label not in result:
            result[label] = []
        result[label].append(node)
    return list(result.values())


def _group_parameters_by_label(nodes: List[Tuple[Node, str]]) -> List[List[Tuple[Node, str]]]:
    result = {}
    for node, argname in nodes:
        label = node.operation.parameters[argname].label
        if label not in result:
            result[label] = []
        result[label].append((node, argname))
    return list(result.values())
