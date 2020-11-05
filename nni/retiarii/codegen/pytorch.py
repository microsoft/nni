from typing import *

from ..graph import IllegalGraphError, Edge, Graph, Node, Model
from ..operation import Operation, Cell


def model_to_pytorch_script(model: Model) -> str:
    graphs = [graph_to_pytorch_model(name, cell) for name, cell in model.graphs.items()]
    return _PyTorchScriptTemplate.format('\n\n'.join(graphs)).strip()


def _sorted_incoming_edges(node: Node) -> List[Edge]:
    edges = [edge for edge in node.graph.edges if edge.tail is node]
    if not edges:
        return []
    if all(edge.tail_slot is None for edge in edges):
        return edges
    if all(isinstance(edge.tail_slot, int) for edge in edges):
        edges = sorted(edges, key=(lambda edge: edge.tail_slot))
        if [edge.tail_slot for edge in edges] == list(range(len(edges))):
            return edges
    raise IllegalGraphError(node.graph, 'Node {} has bad inputs'.format(node.name))


def _format_inputs(node: Node) -> str:
    edges = _sorted_incoming_edges(node)
    inputs = []
    for edge in edges:
        if edge.head.name == '_inputs':
            assert isinstance(edge.head_slot, int)
            if node.graph.input_names is not None:
                # when input has names, e.g., forward(self, tensor1, tensor2, another_one)
                inputs.append(node.graph.input_names[edge.head_slot])
            else:
                # when input has no name, e.g., forward(*_inputs)
                inputs.append('_inputs[{}]'.format(edge.head_slot))
        else:
            if edge.head_slot is None:
                # when the input comes from a single-output operator
                inputs.append('{}'.format(edge.head.name))
            else:
                # when the input comes from a multi-output operator: needs to know which one it comes from
                inputs.append('{}[{}]'.format(edge.head.name, edge.head_slot))
    return ', '.join(inputs)


def graph_to_pytorch_model(graph_name: str, graph: Graph) -> str:
    nodes = graph.nodes  # FIXME: topological sort is needed here

    # handle module node and function node differently
    # only need to generate code for module here
    node_codes = []
    for node in nodes:
        if node.operation:
            node_codes.append(node.operation.to_init_code(node.name))

    if graph.input_names is None:
        input_code = '*_inputs'
    else:
        input_code = ', '.join(graph.input_names)

    edge_codes = []

    for node in nodes:
        if node.operation:
            inputs = _format_inputs(node)
            edge_codes.append(node.operation.to_forward_code(node.name, node.name, inputs))

    output_code = _format_inputs(graph.output_node)
    if not output_code:
        output_code = 'None'

    linebreak = '\n        '
    return _PyTorchModelTemplate.format(
        graph_name=('Graph' if graph_name == '_graph' else graph_name),
        inputs=input_code,
        outputs=output_code,
        nodes=linebreak.join(node_codes),
        edges=linebreak.join(edge_codes)
    )


# TODO: handle imports

_PyTorchScriptTemplate = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

{}
'''

_PyTorchModelTemplate = '''
class {graph_name}(nn.Module):
    def __init__(self):
        super().__init__()
        {nodes}

    def forward(self, {inputs}):
        {edges}
        return {outputs}
'''
