from typing import *

from ..graph import IllegalGraphError, Edge, Graph, Node, Model
from ..operation import Operation, Cell

# TODO: fix: inputs is a list, how to deal with single element list and single element
# TODO: sort edges in topological order

def model_to_pytorch_script(model: Model) -> str:
    graphs = []
    total_pkgs = set()
    for name, cell in model.graphs.items():
        import_pkgs, graph_code = graph_to_pytorch_model(name, cell)
        graphs.append(graph_code)
        total_pkgs.update(import_pkgs)
    # TODO: set correct PATH for the packages
    pkgs_code = '\n'.join(['import {}'.format(pkg) for pkg in total_pkgs])
    return _PyTorchScriptTemplate.format(pkgs_code, '\n\n'.join(graphs)).strip()

def _convert_name(name: str) -> str:
    """
    Convert the names using separator '.' to valid variable name in code
    """
    return name.replace('.', '__')

def _convert_names(names: List[str]) -> List[str]:
    return [_convert_name(name) for name in names]

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

def _format_inputs(node: Node) -> List[str]:
    edges = _sorted_incoming_edges(node)
    inputs = []
    for edge in edges:
        if edge.head.name == '_inputs':
            assert isinstance(edge.head_slot, int)
            if node.graph.input_names is not None:
                # when input has names, e.g., forward(self, tensor1, tensor2, another_one)
                inputs.append(_convert_name(node.graph.input_names[edge.head_slot]))
            else:
                # when input has no name, e.g., forward(*_inputs)
                inputs.append('_inputs[{}]'.format(edge.head_slot))
        else:
            if edge.head_slot is None:
                # when the input comes from a single-output operator
                inputs.append('{}'.format(_convert_name(edge.head.name)))
            else:
                # when the input comes from a multi-output operator: needs to know which one it comes from
                inputs.append('{}[{}]'.format(_convert_name(edge.head.name), edge.head_slot))
    return inputs

def graph_to_pytorch_model(graph_name: str, graph: Graph) -> str:
    nodes = graph.nodes  # FIXME: topological sort is needed here

    # handle module node and function node differently
    # only need to generate code for module here
    import_pkgs = set()
    node_codes = []
    for node in nodes:
        if node.operation:
            pkg_name = node.operation.get_import_pkg()
            if pkg_name is not None:
                import_pkgs.add(pkg_name)
            node_code = node.operation.to_init_code(_convert_name(node.name))
            if node_code is not None:
                node_codes.append(node_code)

    if graph.input_names is None:
        input_code = '*_inputs'
    else:
        input_code = ', '.join(_convert_names(graph.input_names))

    edge_codes = []
    sorted_nodes = graph.topo_sort()
    for node in sorted_nodes:
        if node.operation:
            inputs = _format_inputs(node)
            edge_codes.append(node.operation.to_forward_code(_convert_name(node.name), _convert_name(node.name), inputs))

    output_names = _format_inputs(graph.output_node)
    if not output_names:
        output_names = ['None']

    linebreak = '\n        '
    return import_pkgs, _PyTorchModelTemplate.format(
        graph_name=('Graph' if graph_name == '_graph' else _convert_name(graph_name)),
        inputs=input_code,
        outputs=', '.join(output_names),
        nodes=linebreak.join(node_codes),
        edges=linebreak.join(edge_codes)
    )


# TODO: handle imports

_PyTorchScriptTemplate = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append("test/convert_test")

{}

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
