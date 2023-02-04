# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Convert MutableModules into mutators on graph model space."""

from __future__ import annotations

from typing import Any, List, Tuple, Dict, Iterable, cast

from nni.mutable import MutableExpression, Categorical, frozen_context, label_scope
from nni.nas.space import Graph, GraphModelSpace, Node, StationaryMutator, Mutator
from nni.nas.space.graph_op import Cell


class LayerChoiceMutator(StationaryMutator):
    """Mutate layer choice nodes.

    One mutator corresponds to all layer choices with the same label.
    The choices in layer choice each correspond to a cell in the graph model space,
    which is to support nested layer choice.
    """

    def __init__(self, nodes: List[Node]):
        super().__init__(label=nodes[0].operation.parameters['label'])
        self.nodes = nodes

    def mutate(self, model: GraphModelSpace) -> None:
        candidates = self.nodes[0].operation.parameters['candidates']
        chosen = self.choice(candidates)
        for node in self.nodes:
            # Each layer choice corresponds to a cell, which is unconnected in the base graph.
            # We add the connections here in the mutation logic.
            # Thus, the mutated model should not be mutated again. Everything should be based on the original base graph.
            target = model.graphs[cast(Cell, node.operation).cell_name]
            chosen_node = target.get_node_by_name(chosen)
            assert chosen_node is not None
            target.add_edge((target.input_node, 0), (chosen_node, None))
            target.add_edge((chosen_node, None), (target.output_node, None))
            operation = cast(Cell, node.operation)
            target_node = cast(Node, model.get_node_by_name(node.name))
            target_node.update_operation(Cell(operation.cell_name))

            # remove redundant nodes
            for rm_node in list(target.hidden_nodes):  # remove from a list on the fly will cause issues
                if rm_node.name != chosen_node.name:
                    rm_node.remove()


class InputChoiceMutator(StationaryMutator):
    """
    Mutate the input choice nodes.
    """

    def __init__(self, nodes: List[Node]):
        super().__init__(label=nodes[0].operation.parameters['label'])
        self.nodes = nodes

    def mutate(self, model: GraphModelSpace) -> None:
        n_candidates = self.nodes[0].operation.parameters['n_candidates']
        n_chosen = self.nodes[0].operation.parameters['n_chosen']
        candidates = list(range(n_candidates))
        if n_chosen is None:
            chosen = [i for i in candidates if self.choice([False, True])]
            # FIXME This is a hack to make choice align with the previous format
            self._cur_samples = chosen
        else:
            chosen = [self.choice(candidates) for _ in range(n_chosen)]
        for node in self.nodes:
            target = cast(Node, model.get_node_by_name(node.name))
            target.update_operation('__torch__.nni.nas.nn.pytorch.ChosenInputs',
                                    {'chosen': chosen, 'reduction': node.operation.parameters['reduction']})


class ParameterChoiceLeafMutator(StationaryMutator):
    """
    Mutate the leaf node (i.e., ValueChoice) of parameter choices.

    Should be used together with :class:`ParameterChoiceMutator`.
    """

    def __init__(self, candidates: List[Any], label: str):
        super().__init__(label=label)
        self.candidates = candidates

    def mutate(self, model: GraphModelSpace) -> None:
        # NOTE: leave a record here
        # real mutations will be done in ParameterChoiceMutator
        self.choice(self.candidates)


class ParameterChoiceMutator(StationaryMutator):
    """
    To deal with ValueChoice used as a parameter of a basic unit.

    Should be used together with :class:`ParameterChoiceLeafMutator`.
    :class:`ParameterChoiceMutator` is an empty-shell mutator.
    It calculates all the parameter values based on previous mutations of :class:`ParameterChoiceLeafMutator`.
    """

    def __init__(self, nodes: List[Tuple[Node, str]]):
        super().__init__()

        self.nodes = nodes

        self._within_dry_run = False

    def dry_run(self, model: GraphModelSpace) -> tuple[dict[str, Categorical], GraphModelSpace]:
        try:
            self._within_dry_run = True
            return super().dry_run(model)
        finally:
            self._within_dry_run = False

    def mutate(self, model: GraphModelSpace) -> None:
        # Retrieve the mutation records from history.
        # looks like {"label1": "cat", "label2": 123}
        value_choice_decisions = {}
        for mutation in model.history:
            if isinstance(mutation.mutator, ParameterChoiceLeafMutator):
                value_choice_decisions[mutation.mutator.label] = mutation.samples[0]

        for node, argname in self.nodes:
            # argname is the location of the argument
            # e.g., Conv2d(out_channels=nn.ValueChoice([1, 2, 3])) => argname = "out_channels"
            value_choice: MutableExpression = node.operation.parameters[argname]

            if self._within_dry_run:
                # Dry-run mode. Fake the value based on the frozen context.
                context = frozen_context.current()
                assert context is not None
                context_before_keys = set(context.keys())
                result_value = value_choice.robust_default(context)
                frozen_context.update(
                    {key: value for key, value in context.items() if key not in context_before_keys}
                )
            else:
                # calculate all the values on the leaf node of ValueChoiceX computation graph
                result_value = value_choice.freeze(value_choice_decisions)

            # update model with graph mutation primitives
            target = cast(Node, model.get_node_by_name(node.name))
            target.update_operation(target.operation.type, {**target.operation.parameters, argname: result_value})


class RepeatMutator(StationaryMutator):
    """
    Dealing with Repeat.

    The depth choice should already have been handled in :class:`ParameterChoiceLeafMutator` and :class:`ParameterChoiceMutator`.
    """

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
            assert len(u.successors) == 1, f'This graph is an illegal chain. {u} has output {u.successors}.'
            u = u.successors[0]
        return chain

    def mutate(self, model):
        for node in self.nodes:
            # the logic here is similar to layer choice. We find cell attached to each node.
            target: Graph = model.graphs[cast(Cell, node.operation).cell_name]
            chain = self._retrieve_chain_from_graph(target)
            # and we get the chosen depth (by value choice)
            node_in_model = cast(Node, model.get_node_by_name(node.name))
            # depth is a value choice in base model
            # but it's already mutated by a ParameterChoiceMutator here
            chosen_depth: int = node_in_model.operation.parameters['depth']
            for edge in chain[chosen_depth - 1].outgoing_edges:
                edge.remove()
            target.add_edge((chain[chosen_depth - 1], None), (target.output_node, None))
            for rm_node in chain[chosen_depth:]:
                for edge in rm_node.outgoing_edges:
                    edge.remove()
                rm_node.remove()

            # to delete the unused parameters.
            target_node = cast(Node, model.get_node_by_name(node.name))
            cell_operation = cast(Cell, node.operation)
            target_node.update_operation(Cell(cell_operation.cell_name))


def process_inline_mutation(model: GraphModelSpace) -> List[Mutator]:
    """Generate mutators based on the parsed model space.

    Model space should already have some hints on where the mutators should be plugged in.
    This function will generate the mutators based on those hints.
    """

    applied_mutators = []

    assert label_scope.current() is None, 'label_scope should be empty before processing inline mutation.'

    ic_nodes = _group_by_label(model.get_nodes_by_type('__torch__.nni.nas.nn.pytorch.choice.InputChoice'))
    for node_list in ic_nodes:
        assert _is_all_equal(map(lambda node: node.operation.parameters['n_candidates'], node_list)) and \
            _is_all_equal(map(lambda node: node.operation.parameters['n_chosen'], node_list)), \
            'Input choice with the same label must have the same number of candidates.'
        mutator = InputChoiceMutator(node_list)
        applied_mutators.append(mutator)

    # `pc_nodes` are arguments of basic units. They can be compositions.
    pc_nodes: List[Tuple[Node, str, MutableExpression]] = []
    for node in model.get_nodes():
        # arguments used in operators like Conv2d
        # argument `valuechoice` used in generated repeat cell
        for name, choice in node.operation.parameters.items():
            if isinstance(choice, MutableExpression):
                # e.g., (conv_node, "out_channels", ValueChoice([1, 3]))
                pc_nodes.append((node, name, choice))

    # Break `pc_nodes` down to leaf value choices. They should be what we want to sample.
    leaf_value_choices: Dict[str, List[Any]] = {}
    for _, __, choice in pc_nodes:
        for inner_choice in choice.simplify().values():
            if not isinstance(inner_choice, Categorical):
                raise TypeError(f'Arguments in basic units only support expressions made up of choices, but {inner_choice} found.')
            if inner_choice.label not in leaf_value_choices:
                leaf_value_choices[inner_choice.label] = inner_choice.values
            else:
                assert leaf_value_choices[inner_choice.label] == inner_choice.values, \
                    'Value choice with the same label must have the same candidates, but found ' \
                    f'{leaf_value_choices[inner_choice.label]} vs. {inner_choice.values}'

    for label, candidates in leaf_value_choices.items():
        applied_mutators.append(ParameterChoiceLeafMutator(candidates, label))

    # in the end, add another parameter choice mutator for "real" mutations
    if pc_nodes:
        applied_mutators.append(ParameterChoiceMutator([(node, name) for node, name, _ in pc_nodes]))

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
        # this check is not completely reliable, because it only checks max and min
        assert _is_all_equal(map(lambda node: node.operation.parameters['max_depth'], node_list)) and \
            _is_all_equal(map(lambda node: node.operation.parameters['min_depth'], node_list)), \
            'Repeat with the same label must have the same candidates.'
        mutator = RepeatMutator(node_list)
        applied_mutators.append(mutator)

    return applied_mutators


# utility functions


def _is_all_equal(lst):
    last = None
    for x in lst:
        if last is not None and last != x:
            return False
        last = x
    return True


def _group_by_label(nodes: Iterable[Node]) -> List[List[Node]]:
    result = {}
    for node in nodes:
        label = node.operation.parameters['label']
        if label not in result:
            result[label] = []
        result[label].append(node)
    return list(result.values())
