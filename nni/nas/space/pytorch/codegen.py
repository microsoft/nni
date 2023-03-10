# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = ['model_to_pytorch_script']

import logging
import re
from typing import Dict, List, Tuple, Any, cast

from nni.common.device import Device, GPUDevice
from nni.nas.space.graph import IllegalGraphError, Edge, Graph, Node, GraphModelSpace
from nni.nas.space.graph_op import PyTorchOperation
from nni.nas.utils import STATE_DICT_PY_MAPPING

from .op_def import ToDevice

_logger = logging.getLogger(__name__)


def model_to_pytorch_script(model: GraphModelSpace) -> str:
    graphs = []
    total_pkgs = set()
    for name, cell in model.graphs.items():
        import_pkgs, graph_code = graph_to_pytorch_model(name, cell, placement=model.placement)
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
        edges = sorted(edges, key=(lambda edge: cast(int, edge.tail_slot)))
        if [edge.tail_slot for edge in edges] == list(range(len(edges))):
            return edges
    raise IllegalGraphError(node.graph, 'Node {} has bad inputs'.format(node.name))


def _format_inputs(node: Node, graph_name: str) -> Tuple[List[str], List[Any]]:
    """
    Format the inputs of a given node.
    Inputs will be formatted with ``_format_variable_name``

    Parameters
    ----------
    node : Node
        a graph node, get and format its inputs
    graph_name : str
        subgraph name, to format variable names

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
                inputs.append(_format_variable_name(edge.head.operation.io_names[edge.head_slot], graph_name))
            else:
                # when input has no name, e.g., forward(*_inputs)
                inputs.append('_inputs[{}]'.format(edge.head_slot))
            inputs_value.append(None)
        else:
            if edge.head_slot is None:
                # when the input comes from a single-output operator
                inputs.append(_format_variable_name(edge.head.name, graph_name))
                if edge.head.operation.type in ('prim::Constant', 'prim::GetAttr') and \
                        'value' in edge.head.operation.parameters:
                    inputs_value.append(edge.head.operation.parameters['value'])
                else:
                    inputs_value.append(None)
            else:
                # when the input comes from a multi-output operator: needs to know which one it comes from
                inputs.append('{}[{}]'.format(_format_variable_name(edge.head.name, graph_name), edge.head_slot))
                inputs_value.append(None)
    return inputs, inputs_value


def _format_variable_name(name: str, graph_name: str) -> str:
    """
    1. replace invalid characters in node name
    2. variables name (full name space) is too long, shorten the name by removing the prefix ```graph_name```
    """
    name = name[len(graph_name):] if name.startswith(graph_name) else name
    name = name.replace('/', '__')

    # https://stackoverflow.com/questions/3303312/how-do-i-convert-a-string-to-a-valid-variable-name-in-python
    name = re.sub(r'\W|^(?=\d)', '_', name)

    if name.startswith('__') and (len(name) > 2 and name[2] != '_'):
        # name can't start with double underscore
        # it's reserved in Python: https://stackoverflow.com/a/1301409/6837658
        # but it's actually very common in our generated code
        name = name[1:]
    elif name.startswith('_'):
        # to avoid conflicts between '_' and '__'
        name = 'i' + name

    return name


def generate_cuda_mapping(placement: Dict[Node, Device]) -> Dict[Device, int]:
    '''
    Since CUDA_VISIBLE_DEVICES will be set to the list of real GPU ID,
    we need to remap the GPU ID when generating code to match them correctly.
    For example, when CUDA_VISIBLE_DEVICES="0,3", we need to use "cuda:0", "cuda:1" in the generated code.
    '''
    unique_devices = sorted(list(set([e for e in placement.values() if isinstance(e, GPUDevice)])))
    node_gpu_cnt = {}
    cuda_remapped_id = {}
    for d in unique_devices:
        if d.node_id not in node_gpu_cnt:
            node_gpu_cnt[d.node_id] = 0
        node_gpu_cnt[d.node_id] += 1
        cuda_remapped_id[d] = node_gpu_cnt[d.node_id] - 1

    return cuda_remapped_id


def graph_to_pytorch_model(graph_name: str, graph: Graph, placement=None) -> Tuple[set, str]:
    nodes = graph.topo_sort()

    # handle module node and function node differently
    # only need to generate code for module here
    import_pkgs = set()
    node_codes = []
    node_python_mappings = {}
    cuda_remapped_id = None
    if placement:
        cuda_remapped_id = generate_cuda_mapping(placement)
    for node in nodes:
        if node.operation:
            if placement and isinstance(node.operation, ToDevice):
                cuda_remapped_id = cast(dict, cuda_remapped_id)
                node.operation.override_device_repr("cuda:%d" % cuda_remapped_id[node.operation.device])

            if node.operation.type == 'shared':
                continue
            pkg_name = cast(PyTorchOperation, node.operation).get_import_pkg()
            if pkg_name is not None:
                import_pkgs.add(pkg_name)

            py_variable_name = _format_variable_name(node.name, graph_name)
            node_code = node.operation.to_init_code(py_variable_name)
            if node_code is not None:
                if placement and node in placement and len(node_code) > 0:
                    if isinstance(placement[node], GPUDevice):
                        assert cuda_remapped_id is not None
                        device_repr = "cuda:%d" % cuda_remapped_id[placement[node]]
                    else:
                        device_repr = placement[node].device_repr()
                    node_codes.append(f"{node_code}.to('{device_repr}')")
                else:
                    node_codes.append(node_code)

                # Map to module hierarchies in original search space python code
                node_python_mappings[py_variable_name] = node.python_name

    node_codes.append(f'self.{STATE_DICT_PY_MAPPING} = {node_python_mappings}')

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
            inputs, inputs_value = _format_inputs(node, graph_name)
            node_name = _format_variable_name(node.name, graph_name)
            submodule_name = node_name
            if node.operation.type == 'shared':
                submodule_name = _format_variable_name(node.operation.parameters['reference'], graph_name)
            edge_codes.append(node.operation.to_forward_code(submodule_name, node_name, inputs, inputs_value))

    output_names, _ = _format_inputs(graph.output_node, graph_name)
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

import nni.nas.nn.pytorch

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
