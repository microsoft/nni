# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import List, Tuple, Any

from ..graph import IllegalGraphError, Edge, Graph, Node, Model

_logger = logging.getLogger(__name__)


def model_to_pytorch_script(model: Model, placement=None) -> str:
    graphs = []
    total_pkgs = set()
    for name, cell in model.graphs.items():
        import_pkgs, graph_code = graph_to_pytorch_model(name, cell, placement=placement)
        graphs.append(graph_code)
        total_pkgs.update(import_pkgs)
    pkgs_code = '\n'.join(['import {}'.format(pkg) for pkg in total_pkgs])
    return _PyTorchScriptTemplate.format(pkgs_code, '\n\n'.join(graphs)).strip()


def _sorted_incoming_edges(node: Node) -> List[Edge]:
    edges = [edge for edge in node.graph.edges if edge.tail is node]
    _logger.debug('sorted_incoming_edges: %s', str(edges))
    if not edges:
        return []
    _logger.debug('all tail_slots are None: %s', str([edge.tail_slot for edge in edges]))
    if all(edge.tail_slot is None for edge in edges):
        return edges
    if all(isinstance(edge.tail_slot, int) for edge in edges):
        edges = sorted(edges, key=(lambda edge: edge.tail_slot))
        if [edge.tail_slot for edge in edges] == list(range(len(edges))):
            return edges
    raise IllegalGraphError(node.graph, 'Node {} has bad inputs'.format(node.name))


def _format_inputs(node: Node) -> Tuple[List[str], List[Any]]:
    """
    Format the inputs of a given node

    Parameters
    ----------
    node : Node
        a graph node, get and format its inputs

    Returns
    -------
    list
        the list of input names
    list
        the list of input values, if an input is simple type, record its value,
        otherwise the value is None
    """
    edges = _sorted_incoming_edges(node)
    inputs = []
    inputs_value = []
    for edge in edges:
        if edge.head.name == '_inputs':
            assert isinstance(edge.head_slot, int)
            if edge.head.operation.io_names is not None:
                # when input has names, e.g., forward(self, tensor1, tensor2, another_one)
                inputs.append(edge.head.operation.io_names[edge.head_slot])
            else:
                # when input has no name, e.g., forward(*_inputs)
                inputs.append('_inputs[{}]'.format(edge.head_slot))
            inputs_value.append(None)
        else:
            if edge.head_slot is None:
                # when the input comes from a single-output operator
                inputs.append('{}'.format(edge.head.name))
                if edge.head.operation.type in ('prim::Constant', 'prim::GetAttr') and \
                    'value' in edge.head.operation.parameters:
                    inputs_value.append(edge.head.operation.parameters['value'])
                else:
                    inputs_value.append(None)
            else:
                # when the input comes from a multi-output operator: needs to know which one it comes from
                inputs.append('{}[{}]'.format(edge.head.name, edge.head_slot))
                inputs_value.append(None)
    return inputs, inputs_value


def _remove_prefix(names, graph_name):
    """
    variables name (full name space) is too long,
    shorten the name by removing the prefix ```graph_name```
    """
    if isinstance(names, list):
        converted_names = []
        for name in names:
            if name.startswith(graph_name):
                converted_names.append(name[len(graph_name):])
            else:
                converted_names.append(name)
        return converted_names
    else:
        return names[len(graph_name):] if names.startswith(graph_name) else names


def graph_to_pytorch_model(graph_name: str, graph: Graph, placement=None) -> str:
    nodes = graph.topo_sort()

    # handle module node and function node differently
    # only need to generate code for module here
    import_pkgs = set()
    node_codes = []
    for node in nodes:
        if node.operation:
            if node.operation.type == 'shared':
                continue
            pkg_name = node.operation.get_import_pkg()
            if pkg_name is not None:
                import_pkgs.add(pkg_name)
            node_code = node.operation.to_init_code(_remove_prefix(node.name, graph_name))
            if node_code is not None:
                if placement and node in placement and len(node_code) > 0:
                    node_codes.append(f"{node_code}.to('{placement[node].device}')")
                else:
                    node_codes.append(node_code)

    if graph.input_node.operation.io_names is None:
        input_code = '*_inputs'
    else:
        for name in graph.input_node.operation.io_names:
            assert not name.startswith(graph_name)
        input_code = ', '.join(graph.input_node.operation.io_names)

    edge_codes = []
    sorted_nodes = graph.topo_sort()
    for node in sorted_nodes:
        if node.operation:
            inputs, inputs_value = _format_inputs(node)
            inputs = _remove_prefix(inputs, graph_name)
            node_name = _remove_prefix(node.name, graph_name)
            submodule_name = node_name
            if node.operation.type == 'shared':
                submodule_name = _remove_prefix(node.operation.parameters['reference'], graph_name)
            edge_codes.append(node.operation.to_forward_code(submodule_name, node_name, inputs, inputs_value))

    output_names, _ = _format_inputs(graph.output_node)
    output_names = _remove_prefix(output_names, graph_name)
    if not output_names:
        raise RuntimeError('"forward" function should have return value(s): {}, {}, {}'.format(output_names, graph_name, graph.output_node))
    output_code = ', '.join(output_names)

    linebreak = '\n        '
    return import_pkgs, _PyTorchModelTemplate.format(
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

import nni.retiarii.nn.pytorch

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
