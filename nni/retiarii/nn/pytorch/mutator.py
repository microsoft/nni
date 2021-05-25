# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Optional, Tuple

from ...mutator import Mutator
from ...graph import Cell, Graph, Model, Node
from .api import ValueChoice


class LayerChoiceMutator(Mutator):
    def __init__(self, nodes: List[Node]):
        super().__init__()
        self.nodes = nodes

    def mutate(self, model):
        candidates = self.nodes[0].operation.parameters['candidates']
        chosen = self.choice(candidates)
        for node in self.nodes:
            # Each layer choice corresponds to a cell, which is unconnected in the base graph.
            # We add the connections here in the mutation logic.
            # Thus, the mutated model should not be mutated again. Everything should be based on the original base graph.
            target = model.graphs[node.operation.cell_name]
            chosen_node = target.get_node_by_name(chosen)
            assert chosen_node is not None
            target.add_edge((target.input_node, 0), (chosen_node, None))
            target.add_edge((chosen_node, None), (target.output_node, None))
            model.get_node_by_name(node.name).update_operation(Cell(node.operation.cell_name))

            # remove redundant nodes
            for rm_node in target.hidden_nodes:
                if rm_node.name != chosen_node.name:
                    rm_node.remove()


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
            chosen_value = node.operation.parameters[argname].access(chosen)
            target = model.get_node_by_name(node.name)
            target.update_operation(target.operation.type, {**target.operation.parameters, argname: chosen_value})


class RepeatMutator(Mutator):
    def __init__(self, nodes: List[Node]):
        # nodes is a subgraph consisting of repeated blocks.
        super().__init__()
        self.nodes = nodes

    def _retrieve_chain_from_graph(self, graph: Graph) -> List[Node]:
        u = graph.input_node
        chain = []
        while u != graph.output_node:
            if u != graph.input_node:
                chain.append(u)
            assert len(u.successors) == 1, f'This graph is an illegal chain. {u} has output {u.successor}.'
            u = u.successors[0]
        return chain

    def mutate(self, model):
        min_depth = self.nodes[0].operation.parameters['min_depth']
        max_depth = self.nodes[0].operation.parameters['max_depth']
        if min_depth < max_depth:
            chosen_depth = self.choice(list(range(min_depth, max_depth + 1)))
        for node in self.nodes:
            # the logic here is similar to layer choice. We find cell attached to each node.
            target: Graph = model.graphs[node.operation.cell_name]
            chain = self._retrieve_chain_from_graph(target)
            for edge in chain[chosen_depth - 1].outgoing_edges:
                edge.remove()
            target.add_edge((chain[chosen_depth - 1], None), (target.output_node, None))
            for rm_node in chain[chosen_depth:]:
                for edge in rm_node.outgoing_edges:
                    edge.remove()
                rm_node.remove()
            # to delete the unused parameters.
            model.get_node_by_name(node.name).update_operation(Cell(node.operation.cell_name))


def process_inline_mutation(model: Model) -> Optional[List[Mutator]]:
    applied_mutators = []

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
        first_node, first_argname = node_list[0]
        mutator = ParameterChoiceMutator(node_list, first_node.operation.parameters[first_argname].candidates)
        applied_mutators.append(mutator)

    # apply layer choice at last as it will delete some nodes
    lc_nodes = _group_by_label(filter(lambda d: d.operation.parameters.get('mutation') == 'layerchoice',
                                      model.get_nodes_by_type('_cell')))
    for node_list in lc_nodes:
        assert _is_all_equal(map(lambda node: len(node.operation.parameters['candidates']), node_list)), \
            'Layer choice with the same label must have the same number of candidates.'
        mutator = LayerChoiceMutator(node_list)
        applied_mutators.append(mutator)

    repeat_nodes = _group_by_label(filter(lambda d: d.operation.parameters.get('mutation') == 'repeat',
                                      model.get_nodes_by_type('_cell')))
    for node_list in repeat_nodes:
        assert _is_all_equal(map(lambda node: node.operation.parameters['max_depth'], node_list)) and \
            _is_all_equal(map(lambda node: node.operation.parameters['min_depth'], node_list)), \
            'Repeat with the same label must have the same number of candidates.'
        mutator = RepeatMutator(node_list)
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
