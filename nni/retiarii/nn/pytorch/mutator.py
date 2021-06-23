# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
from typing import Any, List, Optional, Tuple

import torch.nn as nn

from ...mutator import Mutator
from ...graph import Cell, Graph, Model, ModelStatus, Node
from .api import LayerChoice, InputChoice, ValueChoice, Placeholder
from .component import Repeat
from ...utils import uid


class LayerChoiceMutator(Mutator):
    def __init__(self, nodes: List[Node]):
        super().__init__(label=nodes[0].operation.parameters['label'])
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
            for rm_node in list(target.hidden_nodes):  # remove from a list on the fly will cause issues
                if rm_node.name != chosen_node.name:
                    rm_node.remove()


class InputChoiceMutator(Mutator):
    def __init__(self, nodes: List[Node]):
        super().__init__(label=nodes[0].operation.parameters['label'])
        self.nodes = nodes

    def mutate(self, model):
        n_candidates = self.nodes[0].operation.parameters['n_candidates']
        n_chosen = self.nodes[0].operation.parameters['n_chosen']
        candidates = list(range(n_candidates))
        chosen = [self.choice(candidates) for _ in range(n_chosen)]
        for node in self.nodes:
            target = model.get_node_by_name(node.name)
            target.update_operation('__torch__.nni.retiarii.nn.pytorch.ChosenInputs',
                                    {'chosen': chosen, 'reduction': node.operation.parameters['reduction']})


class ValueChoiceMutator(Mutator):
    def __init__(self, nodes: List[Node], candidates: List[Any]):
        super().__init__(label=nodes[0].operation.parameters['label'])
        self.nodes = nodes
        self.candidates = candidates

    def mutate(self, model):
        chosen = self.choice(self.candidates)
        for node in self.nodes:
            target = model.get_node_by_name(node.name)
            target.update_operation('prim::Constant', {'type': type(chosen).__name__, 'value': chosen})


class ParameterChoiceMutator(Mutator):
    def __init__(self, nodes: List[Tuple[Node, str]], candidates: List[Any]):
        node, argname = nodes[0]
        super().__init__(label=node.operation.parameters[argname].label)
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
        super().__init__(label=nodes[0].operation.parameters['label'])
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


# The following are written for pure-python mode


class ManyChooseManyMutator(Mutator):
    """
    Choose based on labels. Will not affect the model itself.
    """

    def __init__(self, label: Optional[str]):
        super().__init__(label=label)

    @staticmethod
    def candidates(node):
        if 'n_candidates' in node.operation.parameters:
            return list(range(node.operation.parameters['n_candidates']))
        else:
            return node.operation.parameters['candidates']

    @staticmethod
    def number_of_chosen(node):
        if 'n_chosen' in node.operation.parameters:
            return node.operation.parameters['n_chosen']
        return 1

    def mutate(self, model: Model):
        # this mutate does not have any effect, but it is recorded in the mutation history
        for node in model.get_nodes_by_label(self.label):
            for _ in range(self.number_of_chosen(node)):
                self.choice(self.candidates(node))
            break


def extract_mutation_from_pt_module(pytorch_model: nn.Module) -> Tuple[Model, Optional[List[Mutator]]]:
    model = Model(_internal=True)
    graph = Graph(model, uid(), '_model', _internal=True)._register()
    model.python_class = pytorch_model.__class__
    if len(inspect.signature(model.python_class.__init__).parameters) > 1:
        if not hasattr(pytorch_model, '_init_parameters'):
            raise ValueError('Please annotate the model with @serialize decorator in python execution mode '
                             'if your model has init parameters.')
        model.python_init_params = pytorch_model._init_parameters
    else:
        model.python_init_params = {}

    for name, module in pytorch_model.named_modules():
        # tricky case: value choice that serves as parameters are stored in _init_parameters
        if hasattr(module, '_init_parameters'):
            for key, value in module._init_parameters.items():
                if isinstance(value, ValueChoice):
                    node = graph.add_node(name + '.init.' + key, 'ValueChoice', {'candidates': value.candidates})
                    node.label = value.label

        if isinstance(module, (LayerChoice, InputChoice, ValueChoice)):
            # TODO: check the label of module and warn if it's auto-generated
            pass
        if isinstance(module, LayerChoice):
            node = graph.add_node(name, 'LayerChoice', {'candidates': module.names})
            node.label = module.label
        if isinstance(module, InputChoice):
            node = graph.add_node(name, 'InputChoice',
                                  {'n_candidates': module.n_candidates, 'n_chosen': module.n_chosen})
            node.label = module.label
        if isinstance(module, ValueChoice):
            node = graph.add_node(name, 'ValueChoice', {'candidates': module.candidates})
            node.label = module.label
        if isinstance(module, Repeat) and module.min_depth <= module.max_depth:
            node = graph.add_node(name, 'Repeat', {
                'candidates': list(range(module.min_depth, module.max_depth + 1))
            })
            node.label = module.label
        if isinstance(module, Placeholder):
            raise NotImplementedError('Placeholder is not supported in python execution mode.')

    model.status = ModelStatus.Frozen
    if not graph.hidden_nodes:
        return model, None

    mutators = []
    for nodes in _group_by_label_and_type(graph.hidden_nodes):
        assert _is_all_equal(map(lambda n: n.operation.type, nodes)), \
            f'Node with label "{nodes[0].label}" does not all have the same type.'
        assert _is_all_equal(map(lambda n: n.operation.parameters, nodes)), \
            f'Node with label "{nodes[0].label}" does not agree on parameters.'
        mutators.append(ManyChooseManyMutator(nodes[0].label))
    return model, mutators


# utility functions


def _is_all_equal(lst):
    last = None
    for x in lst:
        if last is not None and last != x:
            return False
        last = x
    return True


def _group_by_label_and_type(nodes: List[Node]) -> List[List[Node]]:
    result = {}
    for node in nodes:
        key = (node.label, node.operation.type)
        if key not in result:
            result[key] = []
        result[key].append(node)
    return list(result.values())


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
