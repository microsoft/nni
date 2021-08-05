# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: skip-file

"""
FIXME
This file is inherited from last version.

I expect it can work with a few modifications to incorporate with the latest API, but it hasn't
been tested and I'm not sure.
"""

from ..graph_v2 import IllegalGraphError, Cell, Edge, Graph, Node
from ..operations_tf import Operation
from ..type_utils import *


def graph_to_tensorflow_script(graph: Graph) -> str:
    graphs = [graph_to_tensorflow_model(name, cell) for name, cell in graph.cell_templates.items()]
    return _TensorFlowScriptTemplate.format('\n\n'.join(graphs)).strip()


def _sort_incoming_edges(node: Node) -> List[Edge]:
    edges = [edge for edge in node.graph.edges if edge.tail is node]
    if not edges:
        return []
    if all(edge.tail_idx is None for edge in edges):
        return edges
    if all(isinstance(edge.tail_idx, int) for edge in edges):
        edges = sorted(edges, key=(lambda edge: edge.tail_idx))
        if [edge.tail_idx for edge in edges] == list(range(len(edges))):
            return edges
    raise IllegalGraphError(node.graph, 'Node {} has bad inputs'.format(node.name))

def _format_inputs(node: Node) -> str:
    edges = _sort_incoming_edges(node)
    inputs = []
    for edge in edges:
        if edge.head.name == '_inputs':
            assert isinstance(edge.head_idx, int)
            if node.graph.input_names is not None:
                inputs.append(node.graph.input_names[edge.head_idx])
            else:
                inputs.append('_inputs[{}]'.format(edge.head_idx))
        else:
            if edge.head_idx is None:
                inputs.append('{}'.format(edge.head.name))
            else:
                inputs.append('{}[{}]'.format(edge.head.name, edge.head_idx))
    return ', '.join(inputs)


def graph_to_tensorflow_model(graph_name: str, graph: Graph) -> str:
    nodes = graph.topo_sort()

    # handle module node and function node differently
    # only need to generate code for module here
    node_codes = []
    for node in nodes:
        if isinstance(node, Cell):
            node_codes.append('self.{} = {}()'.format(node.name, node.template_name))
        else:
            node_codes.append('self.{} = {}'.format(node.name, cast(Operation, node.operation).to_tensorflow_init()))

    edge_codes = []

    for node in nodes:
        inputs = _format_inputs(node)
        edge_codes.append('{} = self.{}({})'.format(node.name, node.name, inputs))

    output_code = _format_inputs(graph.output_node)
    if not output_code:
        output_code = 'None'

    if graph.input_names is None:
        input_code = '*_inputs'
    else:
        input_code = ', '.join(graph.input_names)

    linebreak = '\n        '
    return _TensorFlowModelTemplate.format(
        graph_name=('Graph' if graph_name == '_graph' else graph_name),
        inputs=input_code,
        outputs=output_code,
        nodes=linebreak.join(node_codes),
        edges=linebreak.join(edge_codes)
    )


_TensorFlowScriptTemplate = '''
import tensorflow as tf
import tensorflow.keras as K

import sdk.custom_ops_tf as CUSTOM

{}
'''

_TensorFlowModelTemplate = '''
class {graph_name}(K.Model):
    def __init__(self):
        super().__init__()
        {nodes}

    def call(self, {inputs}):
        {edges}
        return {outputs}
'''