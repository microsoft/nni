# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from typing import Dict, Tuple, List, Any

from nni.retiarii.utils import uid
from ...graph import Cell, Edge, Graph, Model, Node
from ...operation import Operation, _IOPseudoOperation


class PhysicalDevice:
    def __init__(self, server: str, device: str):
        self.server = server
        self.device = device

    def __eq__(self, o) -> bool:
        return self.server == o.server and self.device == o.device

    def __hash__(self) -> int:
        return hash(self.server + '_' + self.device)


class AbstractLogicalNode(Node):
    def __init__(self, graph, node_id, name, operation, _internal=False):
        super().__init__(graph, node_id, name, operation, _internal=_internal)

    def assemble(self, multi_model_placement: Dict[Model, PhysicalDevice]) -> Tuple[Node, PhysicalDevice]:
        raise NotImplementedError

    def _fork_to(self, graph: Graph):
        raise NotImplementedError


class LogicalGraph(Graph):
    def __init__(self, model: Model, graph_id: int, name: str = None, _internal: bool = False):
        super().__init__(model, graph_id, name='logical_' + name, _internal=_internal)

    def _dump(self) -> Any:
        nodes_dump = {}
        for node in self.hidden_nodes:
            if isinstance(node, OriginNode):
                nodes_dump[f"{node.original_graph.model.model_id}_{node.name}"] = node._dump(
                )
            else:
                nodes_dump[f"{node.graph.model.model_id}_{node.name}"] = node._dump()

        edges_dump = []
        for edge in self.edges:
            if isinstance(edge.head, OriginNode):
                head_info = f'{edge.head.original_graph.model.model_id}_{edge.head.name}'
            else:
                head_info = edge.head.name
            if isinstance(edge.tail, OriginNode):
                tail_info = f'{edge.tail.original_graph.model.model_id}_{edge.tail.name}'
            else:
                tail_info = edge.tail.name
            edges_dump.append((head_info, tail_info))
        return {
            'inputs': self.input_node.operation.io_names,
            'outputs': self.output_node.operation.io_names,
            'nodes': nodes_dump,
            'edges': edges_dump
        }

    def _fork_to(self, model: Model) -> Graph:
        new_graph = Graph(model, self.id, self.name,
                          _internal=True)._register()

        for node in self.hidden_nodes:
            if isinstance(node, AbstractLogicalNode):
                node._fork_to(new_graph)
            else:
                Node(new_graph, node.id, node.name,
                     node.operation, _internal=True)._register()

        id_to_new_node = {node.__repr__(): node for node in new_graph.nodes}

        for edge in self.edges:
            new_head = id_to_new_node[edge.head.__repr__()]
            new_tail = id_to_new_node[edge.tail.__repr__()]
            Edge((new_head, edge.head_slot),
                 (new_tail, edge.tail_slot), _internal=True)._register()

        return new_graph


class OriginNode(AbstractLogicalNode):
    def __init__(self, logical_graph: LogicalGraph,
                 original_graph: Graph, original_node: Node,
                 name: str, operation, _internal=False):
        super().__init__(logical_graph, original_node.id, name, operation)
        self.original_graph = original_graph
        self.original_node = original_node

    def assemble(self, multi_model_placement: Dict[Model, PhysicalDevice]) -> Tuple[Node, PhysicalDevice]:
        model_id = self.original_node.graph.model.model_id
        new_node = Node(self.original_node.graph, self.original_node.id,
                        f"M_{model_id}_" +
                        self.original_node.name,
                        self.original_node.operation)
        return new_node, multi_model_placement[self.original_node.graph.model]

    def __repr__(self):
        return f'OriginNode(id={self.id}, name={self.name}, \
            operation={self.operation}, origin_model_id={self.original_graph.model.model_id})'

    def _fork_to(self, graph: Graph):
        OriginNode(graph, self.original_graph, self.original_node,
                   self.name, self.operation)._register()


class LogicalPlan:
    def __init__(self, plan_id=0) -> None:
        self.lp_model = Model(_internal=True)
        self.id = plan_id
        self.logical_graph = LogicalGraph(
            self.lp_model, self.id, name=f'{self.id}', _internal=True)._register()
        self.lp_model._root_graph_name = self.logical_graph.name
        self.models = []

    def add_model(self, model: Model):
        self.models.append(model)
        # Only optimize the root graph.
        self._merge_graph(model.root_graph)

    def _merge_graph(self, from_graph):
        to_graph = self.logical_graph
        id_to_new_node = {}  # old node ID -> new node object

        for old_node in from_graph.nodes:
            new_node = OriginNode(to_graph, old_node.graph,
                                  old_node, old_node.name,
                                  old_node.operation, _internal=True)._register()
            id_to_new_node[old_node.id] = new_node

        for edge in from_graph.edges:
            new_head = id_to_new_node[edge.head.id]
            new_tail = id_to_new_node[edge.tail.id]
            Edge((new_head, edge.head_slot), (new_tail,
                                              edge.tail_slot), _internal=True)._register()

    def assemble(self, multi_model_placement: Dict[Model, PhysicalDevice]) \
            -> Tuple[Model, Dict[Node, PhysicalDevice], List[Model]]:
        phy_model = Model(_internal=True)  # self.lp_model.fork()
        phy_graph = self.lp_model.root_graph._fork_to(phy_model)

        # Add a flag to mark multi-model in graph json.
        # Multi-model has a list of training configs in kwargs['model_kwargs']
        if len(multi_model_placement) > 1:
            phy_model.evaluator.kwargs['is_multi_model'] = True
            phy_model.evaluator.kwargs['model_cls'] = phy_graph.name
            phy_model.evaluator.kwargs['model_kwargs'] = []
            # FIXME: allow user to specify
            phy_model.evaluator.module = 'nni.retiarii.trainer.pytorch.PyTorchMultiModelTrainer'

        # merge sub-graphs
        for model in multi_model_placement:
            for graph_name in model.graphs:
                if graph_name != model._root_graph_name:
                    model.graphs[graph_name]._fork_to(
                        phy_model, name_prefix=f'M_{model.model_id}_')

        # When replace logical nodes, merge the training configs when
        # input/output nodes are replaced.
        evaluator_slot = {}  # Model ID -> Slot ID
        input_slot_mapping = {}
        output_slot_mapping = {}
        # Replace all logical nodes to executable physical nodes
        hidden_nodes = phy_graph.hidden_nodes.copy()
        node_placements = {}
        for node in hidden_nodes:
            if isinstance(node, OriginNode):
                model_id = node.original_graph.model.model_id
                if node.original_graph.model not in multi_model_placement:
                    for edge in node.incoming_edges:
                        edge.remove()
                    for edge in node.outgoing_edges:
                        edge.remove()
                    node.remove()
                    continue

            if isinstance(node, AbstractLogicalNode):
                new_node, placement = node.assemble(multi_model_placement)
                if isinstance(new_node.operation, _IOPseudoOperation):
                    model_id = new_node.graph.model.model_id
                    if model_id not in evaluator_slot:
                        phy_model.evaluator.kwargs['model_kwargs'].append(new_node.graph.model.evaluator.kwargs.copy())
                        evaluator_slot[model_id] = len(phy_model.evaluator.kwargs['model_kwargs']) - 1
                        slot = evaluator_slot[model_id]
                        phy_model.evaluator.kwargs['model_kwargs'][slot]['model_id'] = model_id
                        phy_model.evaluator.kwargs['model_kwargs'][slot]['use_input'] = False
                        phy_model.evaluator.kwargs['model_kwargs'][slot]['use_output'] = False
                    else:
                        slot = evaluator_slot[model_id]
                    # If a model's inputs/outputs are not used in the multi-model
                    # the codegen and trainer should not generate and use them
                    # "use_input" and "use_output" are used to mark whether
                    # an input/output of a model is used in a multi-model
                    if new_node.operation.type == '_inputs':
                        input_slot_mapping[new_node] = slot
                        phy_model.evaluator.kwargs['model_kwargs'][slot]['use_input'] = True
                    if new_node.operation.type == '_outputs':
                        output_slot_mapping[new_node] = slot
                        phy_model.evaluator.kwargs['model_kwargs'][slot]['use_output'] = True

                self.node_replace(node, new_node)

                if isinstance(new_node.operation, Cell):
                    old_cell_name = new_node.operation.cell_name
                    new_node.operation = copy.deepcopy(new_node.operation)
                    new_node.operation.cell_name = f'M_{model_id}_{old_cell_name}'
                node_placements[new_node] = placement

                node.remove()

        # If two nodes are placed on different devices, use ToDevice op to copy the node
        existing_edges = phy_graph.edges.copy()
        # Avoid a node is copied multiple times on the same device
        copied_op: Dict[Tuple(Node, PhysicalDevice), Node] = {}
        for edge in existing_edges:
            head_placement = node_placements[edge.head]
            tail_placement = node_placements[edge.tail]
            if head_placement != tail_placement:
                if head_placement.server != tail_placement.server:
                    raise ValueError('Cross-server placement is not supported.')
                # Same server different devices
                if (edge.head, tail_placement) in copied_op:
                    to_node = copied_op[(edge.head, tail_placement)]
                else:
                    to_operation = Operation.new('ToDevice', {"device": tail_placement.device})
                    to_node = Node(phy_graph, uid(), edge.head.name + "_to_" + edge.tail.name, to_operation)._register()
                    Edge((edge.head, edge.head_slot), (to_node, None), _internal=True)._register()
                    copied_op[(edge.head, tail_placement)] = to_node
                edge.head = to_node
                edge.head_slot = None

        # merge all input nodes into one with multiple slots
        input_nodes = []
        for node in phy_graph.hidden_nodes:
            if isinstance(node.operation, _IOPseudoOperation) and node.operation.type == '_inputs':
                input_nodes.append(node)

        for edge in phy_graph.edges:
            if edge.head in input_nodes:
                edge.head_slot = input_slot_mapping[edge.head]
                edge.head = phy_graph.input_node

        # merge all output nodes into one with multiple slots
        output_nodes = []
        for node in phy_graph.hidden_nodes:
            if isinstance(node.operation, _IOPseudoOperation) and node.operation.type == '_outputs':
                output_nodes.append(node)

        for edge in phy_graph.edges:
            if edge.tail in output_nodes:
                edge.tail_slot = output_slot_mapping[edge.tail]
                edge.tail = phy_graph.output_node

        for node in input_nodes:
            node.remove()
        for node in output_nodes:
            node.remove()

        return phy_model, node_placements

    def node_replace(self, old_node: Node, new_node: Node, input_slot_mapping=None, output_slot_mapping=None):
        # TODO: currently, only support single input slot and output slot.
        if input_slot_mapping is not None or output_slot_mapping is not None:
            raise ValueError('Slot mapping is not supported')

        phy_graph = old_node.graph
        new_node.graph = phy_graph

        new_node._register()

        for edge in phy_graph.edges:
            if edge.head == old_node:
                edge.head = new_node
            elif edge.tail == old_node:
                edge.tail = new_node

        # after the replacement, there might be multiple duplicated edges
        # with the same input and output nodes, which should be de-duplicated
        self._remove_duplicated_edges()

    def _remove_duplicated_edges(self):
        # TODO: it does not have duplicated edges if only supporting dedup input
        # Duplicated edges appear when a chain of prefix nodes are deduplicated
        pass
