# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from typing_extensions import TypeGuard

from nni.nas.space.graph import Cell, GraphModelSpace, Graph, Node, Edge


def build_full_name(prefix, name, seq=None):
    if isinstance(name, list):
        name = '__'.join(name)
    if seq is None:
        return '{}__{}'.format(prefix, name)
    else:
        return '{}__{}{}'.format(prefix, name, str(seq))


def build_python_name(prefix, name):
    if isinstance(name, list):
        name = '.'.join(name)
    if prefix:
        return '{}.{}'.format(prefix, name)
    else:  # predix could be None
        return name


def build_cand_name(name, label):
    return f"layerchoice_{label.replace('/', '__')}_{name}"


def _convert_name(name: str) -> str:
    """
    Convert the names using separator '.' to valid variable name in code
    """
    return name.replace('.', '__')


def _extract_info_from_trace_node(trace_node):
    """
    Extract parameters from a trace node.

    Parameters
    ----------
    trace_node: torch._C.Value
    """
    input_shape = []
    output_shape = []

    inputs = list(trace_node.inputs())

    # cat input tensors are in a strange place
    if trace_node.kind() == 'aten::cat':
        input_shape = [input.type().sizes() for input in inputs[0].node().inputs()]
    else:
        for _input in inputs:
            input_type = _input.type()
            if input_type.kind() == 'TensorType':
                shape = input_type.sizes()
                if shape:
                    input_shape.append(shape)

    for _output in trace_node.outputs():
        output_type = _output.type()
        if output_type.kind() == 'TensorType':
            shape = output_type.sizes()
            if shape:
                output_shape.append(shape)

    shape_parameters = {
        'input_shape': input_shape,
        'output_shape': output_shape,
    }

    if trace_node.kind() == 'aten::cat':
        parameters = {'dim': inputs[1].toIValue()}
        return shape_parameters, parameters
    else:
        return shape_parameters, None


def is_layerchoice_node(ir_node: Optional[Node]) -> TypeGuard[Node]:
    if ir_node is not None and isinstance(ir_node.operation, Cell) and ir_node.operation.parameters.get('mutation') == 'layerchoice':
        return True
    else:
        return False


def get_full_name_by_scope_name(ir_model: GraphModelSpace, scope_names, prefix=''):
    full_name = prefix

    for last_scope in range(len(scope_names)):
        ir_node = ir_model.get_node_by_name(full_name)
        # check if it's layerchoice
        if is_layerchoice_node(ir_node):
            full_name = f'layerchoice_{ir_node.operation.parameters["label"]}_{scope_names[last_scope]}'
        else:
            full_name = build_full_name(full_name, scope_names[last_scope])

    return full_name


def match_node(ir_model: GraphModelSpace, torch_node, prefix=''):
    """
    Match the corresponding node of a torch._C.Value
    """
    scope_names = torch_node.scopeName().split('/')[-1].split('.')[1:]
    full_name = get_full_name_by_scope_name(ir_model, scope_names, prefix)
    # handle the case when node is not nn.Module, but directly used in forward()
    # Because name can't be directly matched, so I use a hacky way.
    # I match the first unshaped node of that kind
    graph = ir_model.graphs.get(full_name)
    if graph is not None:
        for node in graph.get_nodes_by_type(torch_node.kind()):
            if not node.operation.attributes['input_shape']:
                return node
        return None
    else:
        return ir_model.get_node_by_name(full_name)


def _without_shape_info(node: Node):
    return not node.operation.attributes['input_shape'] and not node.operation.attributes['output_shape']


def flatten_model_graph(ir_model: GraphModelSpace):
    """
    Flatten the subgraph into root graph.
    """
    def _flatten(graph: Graph):
        """
        flatten this graph
        """
        model = graph.model
        node_to_remove = []

        for node in graph.hidden_nodes:
            node_graph = model.graphs.get(node.name)
            if node_graph is not None:
                _flatten(node_graph)

                # flatten node graph into this graph
                id_to_new_node = {}
                for node_graph_node in node_graph.hidden_nodes:
                    new_node = Node(graph, node_graph_node.id, node_graph_node.name, node_graph_node.operation, _internal=True)
                    new_node.update_label(node_graph_node.label)
                    new_node._register()
                    id_to_new_node[new_node.id] = new_node

                # reconnect node edges
                for in_edge in node.incoming_edges:
                    graph.del_edge(in_edge)
                    for input_node_edge in node_graph.input_node.outgoing_edges:
                        if input_node_edge.head_slot == in_edge.tail_slot:
                            graph.add_edge(
                                head=(in_edge.head, in_edge.head_slot),
                                tail=(id_to_new_node[input_node_edge.tail.id], input_node_edge.tail_slot))

                for out_edge in node.outgoing_edges:
                    graph.del_edge(out_edge)
                    for output_node_edge in node_graph.output_node.incoming_edges:
                        if output_node_edge.head_slot == out_edge.tail_slot:
                            graph.add_edge(
                                head=(id_to_new_node[output_node_edge.head.id], output_node_edge.head_slot),
                                tail=(out_edge.tail, out_edge.tail_slot))

                for edge in node_graph.edges:
                    if edge.head == node_graph.input_node or edge.tail == node_graph.output_node:
                        continue
                    new_head = id_to_new_node[edge.head.id]
                    new_tail = id_to_new_node[edge.tail.id]
                    Edge((new_head, edge.head_slot), (new_tail, edge.tail_slot), _internal=True)._register()

                node_to_remove.append(node)
                del model.graphs[node.name]

        for node in node_to_remove:
            node.remove()

    new_ir_model = ir_model.fork()
    _flatten(new_ir_model.root_graph)

    # remove subgraphs
    new_ir_model.graphs = {new_ir_model._root_graph_name: new_ir_model.root_graph}
    return new_ir_model


def flatten_model_graph_without_layerchoice(ir_model: GraphModelSpace):
    """
    Flatten the subgraph into root graph and jump all layerchoice
    """
    def _flatten_without_layerchoice(graph: Graph):
        """
        flatten this graph
        """
        model = graph.model
        node_to_remove = []

        for node in graph.hidden_nodes:
            if is_layerchoice_node(node):
                for in_edge in node.incoming_edges:
                    graph.del_edge(in_edge)
                for out_edge in node.outgoing_edges:
                    graph.del_edge(out_edge)
                del model.graphs[node.name]
                node.remove()
                return

            node_graph = model.graphs.get(node.name)
            if node_graph is not None:
                _flatten_without_layerchoice(node_graph)

                # flatten node graph into this graph
                id_to_new_node = {}
                for node_graph_node in node_graph.hidden_nodes:
                    new_node = Node(graph, node_graph_node.id, node_graph_node.name, node_graph_node.operation, _internal=True)
                    new_node.update_label(node_graph_node.label)
                    new_node._register()
                    id_to_new_node[new_node.id] = new_node

                # reconnect node edges
                for in_edge in node.incoming_edges:
                    graph.del_edge(in_edge)
                    for input_node_edge in node_graph.input_node.outgoing_edges:
                        if input_node_edge.head_slot == in_edge.tail_slot:
                            graph.add_edge(
                                head=(in_edge.head, in_edge.head_slot),
                                tail=(id_to_new_node[input_node_edge.tail.id], input_node_edge.tail_slot))

                for out_edge in node.outgoing_edges:
                    graph.del_edge(out_edge)
                    for output_node_edge in node_graph.output_node.incoming_edges:
                        if output_node_edge.head_slot == out_edge.tail_slot:
                            graph.add_edge(
                                head=(id_to_new_node[output_node_edge.head.id], output_node_edge.head_slot),
                                tail=(out_edge.tail, out_edge.tail_slot))

                for edge in node_graph.edges:
                    if edge.head == node_graph.input_node or edge.tail == node_graph.output_node:
                        continue
                    new_head = id_to_new_node[edge.head.id]
                    new_tail = id_to_new_node[edge.tail.id]
                    Edge((new_head, edge.head_slot), (new_tail, edge.tail_slot), _internal=True)._register()

                node_to_remove.append(node)
                del model.graphs[node.name]

        for node in node_to_remove:
            node.remove()

    new_ir_model = ir_model.fork()
    _flatten_without_layerchoice(new_ir_model.root_graph)

    # remove subgraphs
    new_ir_model.graphs = {new_ir_model._root_graph_name: new_ir_model.root_graph}
    return new_ir_model
