# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
from typing import Any, List, Optional, Tuple, Dict, Iterator

import torch.nn as nn

from nni.common.serializer import is_traceable, is_wrapped_with_trace
from nni.retiarii.graph import Cell, Graph, Model, ModelStatus, Node, Evaluator
from nni.retiarii.mutator import Mutator
from nni.retiarii.serializer import is_basic_unit, is_model_wrapped
from nni.retiarii.utils import ModelNamespace, uid

from .api import LayerChoice, InputChoice, ValueChoice, ValueChoiceX, Placeholder
from .component import NasBench101Cell, NasBench101Mutator


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
        if n_chosen is None:
            chosen = [i for i in candidates if self.choice([False, True])]
            # FIXME This is a hack to make choice align with the previous format
            self._cur_samples = chosen
        else:
            chosen = [self.choice(candidates) for _ in range(n_chosen)]
        for node in self.nodes:
            target = model.get_node_by_name(node.name)
            target.update_operation('__torch__.nni.retiarii.nn.pytorch.ChosenInputs',
                                    {'chosen': chosen, 'reduction': node.operation.parameters['reduction']})


class ValueChoiceMutator(Mutator):
    def __init__(self, nodes: List[Node], candidates: List[Any]):
        # use nodes[0] as an example to get label
        super().__init__(label=nodes[0].operation.parameters['label'])
        self.nodes = nodes
        self.candidates = candidates

    def mutate(self, model):
        chosen = self.choice(self.candidates)
        # no need to support transformation here,
        # because it is naturally done in forward loop
        for node in self.nodes:
            target = model.get_node_by_name(node.name)
            target.update_operation('prim::Constant', {'type': type(chosen).__name__, 'value': chosen})


class ParameterChoiceLeafMutator(Mutator):
    # mutate the leaf node (i.e., ValueChoice) of parameter choices
    # should be used together with ParameterChoiceMutator

    def __init__(self, candidates: List[Any], label: str):
        super().__init__(label=label)
        self.candidates = candidates

    def mutate(self, model: Model) -> Model:
        # leave a record here
        # real mutations will be done in ParameterChoiceMutator
        self.choice(self.candidates)


class ParameterChoiceMutator(Mutator):
    # To deal with ValueChoice used as a parameter of a basic unit
    # should be used together with ParameterChoiceLeafMutator
    # parameter choice mutator is an empty-shell-mutator
    # calculate all the parameter values based on previous mutations of value choice mutator

    def __init__(self, nodes: List[Tuple[Node, str]]):
        super().__init__()

        self.nodes = nodes

    def mutate(self, model: Model) -> Model:
        # looks like {"label1": "cat", "label2": 123}
        value_choice_decisions = {}
        for mutation in model.history:
            if isinstance(mutation.mutator, ParameterChoiceLeafMutator):
                value_choice_decisions[mutation.mutator.label] = mutation.samples[0]

        for node, argname in self.nodes:
            # argname is the location of the argument
            # e.g., Conv2d(out_channels=nn.ValueChoice([1, 2, 3])) => argname = "out_channels"
            value_choice: ValueChoiceX = node.operation.parameters[argname]

            # calculate all the values on the leaf node of ValueChoiceX computation graph
            leaf_node_values = []
            for choice in value_choice.inner_choices():
                leaf_node_values.append(value_choice_decisions[choice.label])
            result_value = value_choice.evaluate(leaf_node_values)

            # update model with graph mutation primitives
            target = model.get_node_by_name(node.name)
            target.update_operation(target.operation.type, {**target.operation.parameters, argname: result_value})


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
        for node in self.nodes:
            # the logic here is similar to layer choice. We find cell attached to each node.
            target: Graph = model.graphs[node.operation.cell_name]
            chain = self._retrieve_chain_from_graph(target)
            # and we get the chosen depth (by value choice)
            node_in_model = model.get_node_by_name(node.name)
            # depth is a value choice in base model
            # but it's already mutated by a ParameterChoiceMutator here
            chosen_depth = node_in_model.operation.parameters['depth']
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

    # `pc_nodes` are arguments of basic units. They can be compositions.
    pc_nodes: List[Tuple[Node, str, ValueChoiceX]] = []
    for node in model.get_nodes():
        # arguments used in operators like Conv2d
        # argument `valuechoice` used in generated repeat cell
        for name, choice in node.operation.parameters.items():
            if isinstance(choice, ValueChoiceX):
                # e.g., (conv_node, "out_channels", ValueChoice([1, 3]))
                pc_nodes.append((node, name, choice))

    # Break `pc_nodes` down to leaf value choices. They should be what we want to sample.
    leaf_value_choices: Dict[str, List[Any]] = {}
    for _, __, choice in pc_nodes:
        for inner_choice in choice.inner_choices():
            if inner_choice.label not in leaf_value_choices:
                leaf_value_choices[inner_choice.label] = inner_choice.candidates
            else:
                assert leaf_value_choices[inner_choice.label] == inner_choice.candidates, \
                    'Value choice with the same label must have the same candidates, but found ' \
                    f'{leaf_value_choices[inner_choice.label]} vs. {inner_choice.candidates}'

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
            n_chosen = self.number_of_chosen(node)
            if n_chosen is None:
                candidates = [i for i in self.candidates(node) if self.choice([False, True])]
                # FIXME This is a hack to make choice align with the previous format
                # For example, it will convert [False, True, True] into [1, 2].
                self._cur_samples = candidates
            else:
                for _ in range(n_chosen):
                    self.choice(self.candidates(node))
            break


def extract_mutation_from_pt_module(pytorch_model: nn.Module) -> Tuple[Model, Optional[List[Mutator]]]:
    model = Model(_internal=True)
    graph = Graph(model, uid(), '_model', _internal=True)._register()
    model.python_class = pytorch_model.__class__
    if len(inspect.signature(model.python_class.__init__).parameters) > 1:
        if not is_model_wrapped(pytorch_model):
            raise ValueError('Please annotate the model with @model_wrapper decorator in python execution mode '
                             'if your model has init parameters.')
        model.python_init_params = pytorch_model.trace_kwargs
    else:
        model.python_init_params = {}

    # hyper-parameter choice
    namespace: ModelNamespace = pytorch_model._model_namespace
    for param_spec in namespace.parameter_specs:
        assert param_spec.categorical and param_spec.type == 'choice'
        node = graph.add_node(f'param_spec_{param_spec.name}', 'ModelParameterChoice', {'candidates': param_spec.values})
        node.label = param_spec.name

    for name, module in pytorch_model.named_modules():
        # tricky case: value choice that serves as parameters are stored in traced arguments
        if is_basic_unit(module):
            for key, value in module.trace_kwargs.items():
                if isinstance(value, ValueChoiceX):
                    for i, choice in enumerate(value.inner_choices()):
                        node = graph.add_node(f'{name}.init.{key}.{i}', 'ValueChoice', {'candidates': choice.candidates})
                        node.label = choice.label

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
        if isinstance(module, ValueChoiceX):
            for i, choice in enumerate(module.inner_choices()):
                node = graph.add_node(f'{name}.{i}', 'ValueChoice', {'candidates': choice.candidates})
                node.label = choice.label
        if isinstance(module, NasBench101Cell):
            node = graph.add_node(name, 'NasBench101Cell', {
                'max_num_edges': module.max_num_edges
            })
            node.label = module.label
        if isinstance(module, Placeholder):
            raise NotImplementedError('Placeholder is not supported in python execution mode.')

    model.status = ModelStatus.Frozen
    if not graph.hidden_nodes:
        return model, None

    mutators = []
    mutators_final = []
    for nodes in _group_by_label_and_type(graph.hidden_nodes):
        assert _is_all_equal(map(lambda n: n.operation.type, nodes)), \
            f'Node with label "{nodes[0].label}" does not all have the same type.'
        assert _is_all_equal(map(lambda n: n.operation.parameters, nodes)), \
            f'Node with label "{nodes[0].label}" does not agree on parameters.'
        if nodes[0].operation.type == 'NasBench101Cell':
            mutators_final.append(NasBench101Mutator(nodes[0].label))
        else:
            mutators.append(ManyChooseManyMutator(nodes[0].label))
    return model, mutators + mutators_final


# mutations for evaluator

class EvaluatorValueChoiceLeafMutator(Mutator):
    # see "ParameterChoiceLeafMutator"
    # works in the same way

    def __init__(self, candidates: List[Any], label: str):
        super().__init__(label=label)
        self.candidates = candidates

    def mutate(self, model: Model) -> Model:
        # leave a record here
        # real mutations will be done in ParameterChoiceMutator
        self.choice(self.candidates)


class EvaluatorValueChoiceMutator(Mutator):
    # works in the same way as `ParameterChoiceMutator`
    # we only need one such mutator for one model/evaluator

    def _mutate_traceable_object(self, obj: Any, value_choice_decisions: Dict[str, Any]) -> Any:
        if not _is_traceable_object(obj):
            return obj

        updates = {}

        # For each argument that is a composition of value choice
        # we find all the leaf-value-choice in the mutation
        # and compute the final updates
        for key, param in obj.trace_kwargs.items():
            if isinstance(param, ValueChoiceX):
                leaf_node_values = [value_choice_decisions[choice.label] for choice in param.inner_choices()]
                updates[key] = param.evaluate(leaf_node_values)
            elif is_traceable(param):
                # Recursively
                sub_update = self._mutate_traceable_object(param, value_choice_decisions)
                if sub_update is not param:  # if mutated
                    updates[key] = sub_update

        if updates:
            mutated_obj = obj.trace_copy()                  # Make a copy
            mutated_obj.trace_kwargs.update(updates)        # Mutate
            mutated_obj = mutated_obj.get()                 # Instantiate the full mutated object

            return mutated_obj

        return obj

    def mutate(self, model: Model):
        value_choice_decisions = {}
        for mutation in model.history:
            if isinstance(mutation.mutator, EvaluatorValueChoiceLeafMutator):
                value_choice_decisions[mutation.mutator.label] = mutation.samples[0]

        model.evaluator = self._mutate_traceable_object(model.evaluator, value_choice_decisions)


def process_evaluator_mutations(evaluator: Evaluator, existing_mutators: List[Mutator]) -> List[Mutator]:
    # take all the value choice in the kwargs of evaluaator into a list
    # `existing_mutators` can mutators generated from `model`
    if not _is_traceable_object(evaluator):
        return []
    mutator_candidates = {}
    for param in _expand_nested_trace_kwargs(evaluator):
        if isinstance(param, ValueChoiceX):
            for choice in param.inner_choices():
                # merge duplicate labels
                for mutator in existing_mutators:
                    if mutator.label == choice.label:
                        raise ValueError(
                            f'Found duplicated labels “{choice.label}”. When two value choices have the same name, '
                            'they would share choices. However, sharing choices between model and evaluator is not supported.'
                        )
                if choice.label in mutator_candidates and mutator_candidates[choice.label] != choice.candidates:
                    raise ValueError(
                        f'Duplicate labels for evaluator ValueChoice {choice.label}. They should share choices.'
                        f'But their candidate list is not equal: {mutator_candidates[choice.label][1]} vs. {choice.candidates}'
                    )
                mutator_candidates[choice.label] = choice.candidates
    mutators = []
    for label, candidates in mutator_candidates.items():
        mutators.append(EvaluatorValueChoiceLeafMutator(candidates, label))
    if mutators:
        # one last mutator to actually apply the mutations
        mutators.append(EvaluatorValueChoiceMutator())
    return mutators


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


def _expand_nested_trace_kwargs(obj: Any) -> Iterator[Any]:
    # Get items from `trace_kwargs`.
    # If some item is traceable itself, get items recursively.

    if _is_traceable_object(obj):
        for param in obj.trace_kwargs.values():
            yield param
            yield from _expand_nested_trace_kwargs(param)


def _is_traceable_object(obj: Any) -> bool:
    # Is it a traceable "object" (not class)?
    return is_traceable(obj) and not is_wrapped_with_trace(obj)
