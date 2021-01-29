from typing import Any, List, Union

from ...mutator import Mutator
from ...graph import Model


class LayerChoiceMutator(Mutator):
    def __init__(self, node_name: str, candidates: List[Any]):
        super().__init__()
        self.node_name = node_name
        self.candidates = candidates

    def mutate(self, model):
        target = model.get_node_by_name(self.node_name)
        indexes = [i for i in range(len(self.candidates))]
        chosen_index = self.choice(indexes)
        chosen_cand = self.candidates[chosen_index]
        target.update_operation(chosen_cand['type'], chosen_cand['parameters'])


class InputChoiceMutator(Mutator):
    def __init__(self, node_name: str, n_candidates: int, n_chosen: int, reduction: str):
        super().__init__()
        self.node_name = node_name
        self.n_candidates = n_candidates
        self.n_chosen = n_chosen
        self.reduction = reduction

    def mutate(self, model):
        target = model.get_node_by_name(self.node_name)
        candidates = [i for i in range(self.n_candidates)]
        chosen = [self.choice(candidates) for _ in range(self.n_chosen)]
        target.update_operation('__torch__.nni.retiarii.nn.pytorch.ChosenInputs',
                                {'chosen': chosen, 'reduction': self.reduction})


class ValueChoiceMutator(Mutator):
    def __init__(self, node_name: str, candidates: List[Any]):
        super().__init__()
        self.node_name = node_name
        self.candidates = candidates

    def mutate(self, model):
        target = model.get_node_by_name(self.node_name)
        chosen = self.choice(self.candidates)
        target.update_operation('prim::Constant', {'value': chosen})


def process_inline_mutation(model: Model) -> Union[None, List[Mutator]]:
    lc_nodes = model.get_nodes_by_type('__torch__.nni.retiarii.nn.pytorch.nn.LayerChoice')
    ic_nodes = model.get_nodes_by_type('__torch__.nni.retiarii.nn.pytorch.nn.InputChoice')
    vc_nodes = model.get_nodes_by_type('__torch__.nni.retiarii.nn.pytorch.nn.ValueChoice')
    if not lc_nodes and not ic_nodes and not vc_nodes:
        return None
    applied_mutators = []
    for node in lc_nodes:
        mutator = LayerChoiceMutator(node.name, node.operation.parameters['choices'])
        applied_mutators.append(mutator)
    for node in ic_nodes:
        mutator = InputChoiceMutator(node.name,
                                     node.operation.parameters['n_candidates'],
                                     node.operation.parameters['n_chosen'],
                                     node.operation.parameters['reduction'])
        applied_mutators.append(mutator)
    for node in vc_nodes:
        mutator = ValueChoiceMutator(node.name, node.operation.parameters['candidates'])
        applied_mutators.append(mutator)
    return applied_mutators
