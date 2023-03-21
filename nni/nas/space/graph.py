# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pylint: skip-file
# type: ignore

"""
GraphModelSpace representation for engines based on graph.
"""

from __future__ import annotations

__all__ = [
    'GraphModelSpace', 'Graph', 'Node', 'Edge', 'IllegalGraphError',
]

import json
from typing import (TYPE_CHECKING, Any, Dict, Callable, Iterable, List,
                    Optional, Set, Tuple, Union, ClassVar, cast, overload)

import nni
from nni.common.device import Device, GPUDevice
from nni.mutable import Mutable, LabeledMutable, Sample, uid
from .graph_op import Cell, Operation, _IOPseudoOperation
from .mutator import MutatorSequence, Mutation
from .space import ExecutableModelSpace, ModelStatus

if TYPE_CHECKING:
    from nni.nas.evaluator import Evaluator


EdgeEndpoint = Tuple['Node', Optional[int]]
"""
Type hint for edge's endpoint. The int indicates nodes' order.
"""


class GraphModelSpace(ExecutableModelSpace):
    """
    Represents a neural network model space with graph.
    Previously known as ``GraphModelSpace``.

    During mutation, one :class:`GraphModelSpace` object is created for each trainable snapshot.
    For example, consider a mutator that insert a node at an edge for each iteration.
    In one iteration, the mutator invokes 4 primitives: add node, remove edge, add edge to head, add edge to tail.
    These 4 primitives operates in one :class:`GraphModelSpace` object.
    When they are all done the model will be set to "frozen" (trainable) status and be submitted to execution engine.
    And then a new iteration starts, and a new :class:`GraphModelSpace` object is created by forking last model.

    Attributes
    ----------
    status
        See :class:`ModelStatus`.
    root_graph
        The outermost graph which usually takes dataset as input and feeds output to loss function.
    graphs
        All graphs (subgraphs) in this model.
    evaluator
        GraphModelSpace evaluator
    mutators
        List of mutators that are applied to this model.
    parent
        A :class:`Mutation` object that contains the mutation that creates this model.
    metrics
        Intermediate as well as final metrics.
    """

    framework_type: ClassVar[str] | None = None

    def __init__(self, *, _internal=False):
        super().__init__()
        assert _internal, '`GraphModelSpace()` is private, use `model.fork()` instead'
        self.model_id: int = uid('model')

        self._root_graph_name: str = '_model'
        self.graphs: Dict[str, Graph] = {}
        self.evaluator: Evaluator | None = None

        self.mutators: MutatorSequence = MutatorSequence([])

        self.parent: Mutation | None = None
        self.sample: Sample | None = None

        # Placement is used in CGO engine.
        self.placement: dict[Node, Device] | None = None

    def extra_repr(self):
        return f'model_id={self.model_id}, status={self.status}, graphs={list(self.graphs.keys())}, ' + \
            f'evaluator={self.evaluator}, mutators={self.mutators}, metrics={self.metrics}'

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        with self.mutators.bind_model(self):
            yield from self.mutators.leaf_mutables(is_leaf)
        if isinstance(self.evaluator, Mutable):
            yield from self.evaluator.leaf_mutables(is_leaf)

    def check_contains(self, sample: dict[str, Any]) -> tuple[bool, str]:
        """Check if the sample is contained in the model space."""
        return self.mutators.check_contains(sample)

    def freeze(self, sample: dict[str, Any]) -> GraphModelSpace:
        """
        Freeze the model by applying the sample to mutators.

        Can only be invoked on a mutating model.
        The new model will be in `Frozen` state.

        This API is used in mutator base class.
        """
        assert not self.status.frozen(), 'Can only freeze a initialized model space'
        with self.mutators.bind_model(self):
            model = self.mutators.freeze(sample)
        if isinstance(self.evaluator, Mutable):
            model.evaluator = self.evaluator.freeze(sample)
        model.status = ModelStatus.Frozen
        model.sample = sample
        return model

    def to_code(self) -> str:
        """Convert the model to code."""
        raise NotImplementedError(f'{self.__class__.__name__} does not support to_code()')

    @property
    def root_graph(self) -> Graph:
        return self.graphs[self._root_graph_name]

    @property
    def history(self) -> list[Mutation]:
        """Mutation history.

        A record of where the model comes from.
        ``self`` comes from the mutation recorded in ``self.history[-1]``.
        ``self.history[0]`` is the first mutation happened on the base graph.
        """
        history: list[Mutation] = []
        model = self
        while model.parent is not None:
            history.append(model.parent)
            model = model.parent.from_
        return list(reversed(history))

    def fork(self) -> GraphModelSpace:
        """
        Create a new model which has same topology, names, and IDs to current one.

        Can only be invoked on a frozen model.
        The new model will be in `Mutating` state.

        This API is used in mutator base class.
        """
        new_model = self.__class__(_internal=True)
        new_model._root_graph_name = self._root_graph_name
        new_model.graphs = {name: graph._fork_to(new_model) for name, graph in self.graphs.items()}
        new_model.mutators = self.mutators
        new_model.evaluator = self.evaluator
        new_model.status = self.status
        # Note: the parent is not updated here. It will be updated when the model is changed, that is in mutator.
        # new_model.parent = self
        return new_model

    @classmethod
    def _load(cls, **ir: Any) -> GraphModelSpace:
        framework_type = ir.pop('framework', nni.get_default_framework())
        if framework_type == 'pytorch':
            from .pytorch.graph import PytorchGraphModelSpace
            model = PytorchGraphModelSpace(_internal=True)
        elif framework_type == 'tensorflow' and '_internal' in ir:  # only test purposes
            from .tensorflow.graph import TensorflowGraphModelSpace
            ir.pop('_internal')
            model = TensorflowGraphModelSpace(_internal=True)
        else:
            raise ValueError(f'Unknown framework type: {framework_type}')
        if 'model_id' in ir:    # backward compatibility
            model.model_id = ir.pop('model_id')
        if '_evaluator' in ir:
            model.evaluator = ir.pop('_evaluator')       # Use evaluator's native load
        if '_mutators' in ir:
            model.mutators = ir.pop('_mutators')
        if '_sample' in ir:
            model.sample = ir.pop('_sample')
        if '_placement' in ir:
            model.placement = ir.pop('_placement')
        for graph_name, graph_data in ir.items():
            Graph._load(model, graph_name, graph_data)._register()
        return model

    def _dump(self) -> Any:
        # Calling dump recursively. Manually handle the serialization of nested objects.
        ret = {name: graph._dump() for name, graph in self.graphs.items()}
        ret['framework'] = self.framework_type
        ret['model_id'] = self.model_id
        if self.status in (ModelStatus.Initialized, ModelStatus.Mutating):
            ret['_mutators'] = self.mutators
        if self.evaluator is not None:
            ret['_evaluator'] = self.evaluator
        if self.sample is not None:
            ret['_sample'] = self.sample
        if self.placement is not None:
            ret['_placement'] = self.placement
        return ret

    def get_nodes(self) -> Iterable['Node']:
        """
        Traverse through all the nodes.
        """
        for graph in self.graphs.values():
            for node in graph.nodes:
                yield node

    def get_nodes_by_label(self, label: str) -> List['Node']:
        """
        Traverse all the nodes to find the matched node(s) with the given label.
        There could be multiple nodes with the same label. Name space name can uniquely
        identify a graph or node.

        NOTE: the implementation does not support the class abstraction
        """
        matched_nodes = []
        for graph in self.graphs.values():
            nodes = graph.get_nodes_by_label(label)
            matched_nodes.extend(nodes)
        return matched_nodes

    def get_nodes_by_type(self, type_name: str) -> List['Node']:
        """
        Traverse all the nodes to find the matched node(s) with the given type.
        """
        matched_nodes = []
        for graph in self.graphs.values():
            nodes = graph.get_nodes_by_type(type_name)
            matched_nodes.extend(nodes)
        return matched_nodes

    def get_node_by_name(self, node_name: str) -> 'Node' | None:
        """
        Traverse all the nodes to find the matched node with the given name.
        """
        matched_nodes = []
        for graph in self.graphs.values():
            nodes = graph.get_nodes_by_name(node_name)
            matched_nodes.extend(nodes)
        assert len(matched_nodes) <= 1
        if matched_nodes:
            return matched_nodes[0]
        else:
            return None

    def get_node_by_python_name(self, python_name: str) -> Optional['Node']:
        """
        Traverse all the nodes to find the matched node with the given python_name.
        """
        matched_nodes = []
        for graph in self.graphs.values():
            nodes = graph.get_nodes_by_python_name(python_name)
            matched_nodes.extend(nodes)
        # assert len(matched_nodes) <= 1
        if matched_nodes:
            return matched_nodes[0]
        else:
            return None

    def get_cell_nodes(self) -> List['Node']:
        matched_nodes = []
        for graph in self.graphs.values():
            nodes = [node for node in graph.nodes if isinstance(node.operation, Cell)]
            matched_nodes.extend(nodes)
        return matched_nodes

    def export_placement_constraint(self):
        """
        Export the placement constraint used in training service.
        """
        if self.placement is None:
            return None
        unique_gpus = sorted(set([e for e in self.placement.values() if isinstance(e, GPUDevice)]))
        placement_constraint = None
        if len(unique_gpus) > 0:
            placement_constraint = {}
            placement_constraint['type'] = 'Device'
            placement_constraint['gpus'] = [(e.node_id, e.gpu_id) for e in unique_gpus]
        return placement_constraint


_InputPseudoUid = -1
_OutputPseudoUid = -2


class Graph:
    """
    Graph topology.

    This class simply represents the topology, with no semantic meaning.
    All other information like metric, non-graph functions, mutation history, etc should go to :class:`GraphModelSpace`.

    Each graph belongs to and only belongs to one :class:`GraphModelSpace`.

    Attributes
    ----------
    model
        The model containing (and owning) this graph.
    id
        Unique ID in the model.
        If two models have graphs of identical ID, they are semantically the same graph.
        Typically this means one graph is mutated from another, or they are both mutated from one ancestor.
    name
        Mnemonic name of this graph. It should have an one-to-one mapping with ID.
    input_names
        Optional mnemonic names of input parameters.
    output_names
        Optional mnemonic names of output values.
    input_node
        Incoming node.
    output_node
        Output node.
    hidden_nodes
        Hidden nodes
    nodes
        All input/output/hidden nodes.
    edges
        Edges.
    python_name
        The name of torch.nn.Module, should have one-to-one mapping with items in python model.
    """

    def __init__(self, model: GraphModelSpace, graph_id: int, name: str = cast(str, None), _internal: bool = False):
        assert _internal, '`Graph()` is private'

        self.model: GraphModelSpace = model
        self.id: int = graph_id
        self.name: str = name or f'_generated_{graph_id}'

        # `python_name` is `None` by default. It should be set after initialization if it is needed.
        self.python_name: Optional[str] = None

        self.input_node: Node = Node(self, _InputPseudoUid, '_inputs', _IOPseudoOperation('_inputs'), _internal=True)
        self.output_node: Node = Node(self, _OutputPseudoUid, '_outputs', _IOPseudoOperation('_outputs'), _internal=True)
        self.hidden_nodes: List[Node] = []

        self.edges: List[Edge] = []

    def __repr__(self):
        return f'Graph(id={self.id}, name={self.name}, ' + \
            f'input_names={self.input_node.operation.io_names}, ' + \
            f'output_names={self.output_node.operation.io_names}, ' + \
            f'num_hidden_nodes={len(self.hidden_nodes)}, num_edges={len(self.edges)})'

    @property
    def nodes(self) -> List['Node']:
        return [self.input_node, self.output_node] + self.hidden_nodes

    def _add_input(self, input_name) -> None:
        if self.input_node.operation.io_names is None:
            self.input_node.operation.io_names = [input_name]
        else:
            self.input_node.operation.io_names.append(input_name)

    def _add_output(self, output_name) -> None:
        if self.output_node.operation.io_names is None:
            self.output_node.operation.io_names = [output_name]
        else:
            self.output_node.operation.io_names.append(output_name)

    @overload
    def add_node(self, name: str, operation: Operation) -> 'Node': ...
    @overload
    def add_node(self, name: str, type_name: str, parameters: Dict[str, Any] = cast(Dict[str, Any], None)) -> 'Node': ...

    def add_node(self, name, operation_or_type, parameters=None):  # type: ignore
        if isinstance(operation_or_type, Operation):
            op = operation_or_type
        else:
            op = Operation.new(operation_or_type, cast(dict, parameters), name)
        return Node(self, uid(), name, op, _internal=True)._register()

    @overload
    def insert_node_on_edge(self, edge: 'Edge', name: str, operation: Operation) -> 'Node': ...

    @overload
    def insert_node_on_edge(self, edge: 'Edge', name: str, type_name: str,
                            parameters: Dict[str, Any] = cast(Dict[str, Any], None)) -> 'Node': ...

    def insert_node_on_edge(self, edge, name, operation_or_type, parameters=None) -> 'Node':  # type: ignore
        if isinstance(operation_or_type, Operation):
            op = operation_or_type
        else:
            op = Operation.new(operation_or_type, cast(dict, parameters), name)
        new_node = Node(self, uid(), name, op, _internal=True)._register()
        # update edges
        self.add_edge((edge.head, edge.head_slot), (new_node, None))
        self.add_edge((new_node, None), (edge.tail, edge.tail_slot))
        self.del_edge(edge)
        return new_node

    # mutation
    def add_edge(self, head: EdgeEndpoint, tail: EdgeEndpoint) -> 'Edge':
        assert head[0].graph is self and tail[0].graph is self
        return Edge(head, tail, _internal=True)._register()

    def del_edge(self, edge: 'Edge') -> None:
        self.edges.remove(edge)

    def get_node_by_name(self, name: str) -> Optional['Node']:
        """
        Returns the node which has specified name; or returns `None` if no node has this name.
        """
        found = [node for node in self.nodes if node.name == name]
        return found[0] if found else None

    def get_node_by_python_name(self, python_name: str) -> Optional['Node']:
        """
        Returns the node which has specified python_name; or returns `None` if no node has this python_name.
        """
        found = [node for node in self.nodes if node.python_name == python_name]
        return found[0] if found else None

    def get_nodes_by_type(self, operation_type: str) -> List['Node']:
        """
        Returns nodes whose operation is specified typed.
        """
        return [node for node in self.hidden_nodes if node.operation.type == operation_type]

    def get_node_by_id(self, node_id: int) -> Optional['Node']:
        """
        Returns the node which has specified name; or returns `None` if no node has this name.
        """
        found = [node for node in self.nodes if node.id == node_id]
        return found[0] if found else None

    def get_nodes_by_label(self, label: str) -> List['Node']:
        return [node for node in self.hidden_nodes if node.label == label]

    def get_nodes_by_name(self, name: str) -> List['Node']:
        return [node for node in self.hidden_nodes if node.name == name]

    def get_nodes_by_python_name(self, python_name: str) -> List['Node']:
        return [node for node in self.nodes if node.python_name == python_name]

    def topo_sort(self) -> List['Node']:
        node_to_fanin = {}
        curr_nodes = []
        for node in self.nodes:
            fanin = len(node.incoming_edges)
            node_to_fanin[node] = fanin
            if fanin == 0:
                curr_nodes.append(node)

        sorted_nodes = []
        while curr_nodes:
            curr_node = curr_nodes.pop(0)
            sorted_nodes.append(curr_node)
            # use successor_slots because a node may connect to another node multiple times
            # to different slots
            for successor_slot in curr_node.successor_slots:
                successor = successor_slot[0]
                node_to_fanin[successor] -= 1
                if node_to_fanin[successor] == 0:
                    curr_nodes.append(successor)

        for key in node_to_fanin:
            assert node_to_fanin[key] == 0, '{}, fanin: {}, predecessor: {}, edges: {}, fanin: {}, keys: {}'.format(
                key,
                node_to_fanin[key],
                key.predecessors[0],
                self.edges,
                node_to_fanin.values(),
                node_to_fanin.keys())

        return sorted_nodes

    def fork(self) -> 'Graph':
        """
        Fork the model and returns corresponding graph in new model.
        This shortcut might be helpful because many algorithms only cares about "stem" subgraph instead of whole model.
        """
        return self.model.fork().graphs[self.name]

    def __eq__(self, other: object) -> bool:
        return self is other

    def _fork_to(self, model: GraphModelSpace, name_prefix='') -> 'Graph':
        new_graph = Graph(model, self.id, name_prefix + self.name, _internal=True)._register()
        # TODO: use node copy instead
        new_graph.input_node.operation.io_names = self.input_node.operation.io_names
        new_graph.output_node.operation.io_names = self.output_node.operation.io_names
        new_graph.input_node.update_label(self.input_node.label)
        new_graph.output_node.update_label(self.output_node.label)
        new_graph.python_name = self.python_name

        for node in self.hidden_nodes:
            new_node = Node(new_graph, node.id, node.name, node.operation, _internal=True)
            new_node.python_name = node.python_name
            new_node.update_label(node.label)
            new_node._register()

        id_to_new_node = {node.id: node for node in new_graph.nodes}

        for edge in self.edges:
            new_head = id_to_new_node[edge.head.id]
            new_tail = id_to_new_node[edge.tail.id]
            Edge((new_head, edge.head_slot), (new_tail, edge.tail_slot), _internal=True)._register()

        return new_graph

    def _copy(self) -> 'Graph':
        # Copy this graph inside the model.
        # The new graph will have identical topology, but its nodes' name and ID will be different.
        new_graph = Graph(self.model, uid(), _internal=True)._register()
        new_graph.input_node.operation.io_names = self.input_node.operation.io_names
        new_graph.output_node.operation.io_names = self.output_node.operation.io_names
        new_graph.input_node.update_label(self.input_node.label)
        new_graph.output_node.update_label(self.output_node.label)
        new_graph.python_name = self.python_name

        id_to_new_node = {}  # old node ID -> new node object

        for old_node in self.hidden_nodes:
            new_node = Node(new_graph, uid(), None, old_node.operation, _internal=True)._register()
            new_node.python_name = old_node.python_name
            new_node.update_label(old_node.label)
            id_to_new_node[old_node.id] = new_node

        for edge in self.edges:
            new_head = id_to_new_node[edge.head.id]
            new_tail = id_to_new_node[edge.tail.id]
            Edge((new_head, edge.head_slot), (new_tail, edge.tail_slot), _internal=True)._register()

        return new_graph

    def _register(self) -> 'Graph':
        self.model.graphs[self.name] = self
        return self

    def _rename_graph(self, old_name, new_name):
        self.model.graphs[old_name].name = new_name
        self.model.graphs[new_name] = self.model.graphs[old_name]
        del self.model.graphs[old_name]

    @classmethod
    def _load(cls, model: GraphModelSpace, name: str, ir: Any) -> 'Graph':
        # This won't be used by nni.load(). Thus it doesn't follow the same pattern as other _load() methods.
        graph = Graph(model, uid(), name, _internal=True)
        graph.input_node.operation.io_names = ir.get('inputs')
        graph.output_node.operation.io_names = ir.get('outputs')
        for node_name, node_data in ir['nodes'].items():
            Node._load(graph, node_name, node_data)._register()
        for edge_data in ir['edges']:
            Edge._load(graph, edge_data)._register()
        return graph

    def _dump(self) -> Any:
        # This dump will NOT be used by nni.dump()
        return {
            'inputs': self.input_node.operation.io_names,
            'outputs': self.output_node.operation.io_names,
            'nodes': {node.name: node._dump() for node in self.hidden_nodes},
            'edges': [edge._dump() for edge in self.edges]
        }


class Node:
    """
    An operation or an opaque subgraph inside a graph.

    Each node belongs to and only belongs to one :class:`Graph`.
    Nodes should never be created with constructor. Use :meth:`Graph.add_node` instead.

    The node itself is for topology only.
    Information of tensor calculation should all go inside ``operation`` attribute.

    TODO: parameter of subgraph (cell)
    It's easy to assign parameters on cell node, but it's hard to "use" them.
    We need to design a way to reference stored cell parameters in inner node operations.
    e.g. ``self.fc = Linear(self.units)``  <-  how to express ``self.units`` in IR?

    Attributes
    ----------
    graph
        The graph containing this node.
    id
        Unique ID in the model.
        If two models have nodes with same ID, they are semantically the same node.
    name
        Mnemonic name. It should have an one-to-one mapping with ID.
    python_name
        The name of torch.nn.Module, should have one-to-one mapping with items in python model.
    label
        Optional. If two nodes have the same label, they are considered same by the mutator.
    operation
        Operation.
    cell
        Read only shortcut to get the referenced subgraph.
        If this node is not a subgraph (is a primitive operation), accessing ``cell`` will raise an error.
    predecessors
        Predecessor nodes of this node in the graph. This is an optional mutation helper.
    successors
        Successor nodes of this node in the graph. This is an optional mutation helper.
    incoming_edges
        Incoming edges of this node in the graph. This is an optional mutation helper.
    outgoing_edges
        Outgoing edges of this node in the graph. This is an optional mutation helper.
    """

    def __init__(self, graph, node_id, name, operation, _internal=False):
        self.graph: Graph = graph
        self.id: int = node_id
        self.name: str = name or f'_generated_{node_id}'
        # `python_name` is `None` by default. It should be set after initialization if it is needed.
        self.python_name: Optional[str] = None
        # TODO: the operation is likely to be considered editable by end-user and it will be hard to debug
        # maybe we should copy it here or make Operation class immutable, in next release
        self.operation: Operation = operation
        self.label: Optional[str] = None

    def __repr__(self):
        return f'Node(id={self.id}, name={self.name}, python_name={self.python_name}, label={self.label}, operation={self.operation})'

    @property
    def predecessors(self) -> List['Node']:
        return sorted(set(edge.head for edge in self.incoming_edges), key=(lambda node: node.id))

    @property
    def successors(self) -> List['Node']:
        return sorted(set(edge.tail for edge in self.outgoing_edges), key=(lambda node: node.id))

    @property
    def successor_slots(self) -> Set[Tuple['Node', Union[int, None]]]:
        return set((edge.tail, edge.tail_slot) for edge in self.outgoing_edges)

    @property
    def incoming_edges(self) -> List['Edge']:
        return [edge for edge in self.graph.edges if edge.tail is self]

    @property
    def outgoing_edges(self) -> List['Edge']:
        return [edge for edge in self.graph.edges if edge.head is self]

    @property
    def cell(self) -> Graph:
        assert isinstance(self.operation, Cell)
        return self.graph.model.graphs[self.operation.parameters['cell']]

    def update_label(self, label: Optional[str]) -> None:
        self.label = label

    @overload
    def update_operation(self, operation: Operation) -> None: ...
    @overload
    def update_operation(self, type_name: str, parameters: Dict[str, Any] = cast(Dict[str, Any], None)) -> None: ...

    def update_operation(self, operation_or_type, parameters=None):  # type: ignore
        if isinstance(operation_or_type, Operation):
            self.operation = operation_or_type
        else:
            self.operation = Operation.new(operation_or_type, cast(dict, parameters))

    # mutation
    def remove(self) -> None:
        assert not self.incoming_edges and not self.outgoing_edges
        self.graph.hidden_nodes.remove(self)

    # mutation
    def specialize_cell(self) -> Graph:
        """
        Only available if the operation is a cell.
        Duplicate the cell template and let this node reference to newly created copy.
        """
        new_cell = self.cell._copy()._register()
        self.operation = Cell(new_cell.name)
        return new_cell

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return hash(id(self))

    def _register(self) -> 'Node':
        self.graph.hidden_nodes.append(self)
        return self

    @classmethod
    def _load(cls, graph: Graph, name: str, ir: Any) -> 'Node':
        if ir['operation']['type'] == '_cell':
            op = Cell(ir['operation']['cell_name'], ir['operation'].get('parameters', {}), attributes=ir['operation'].get('attributes', {}))
        else:
            op = Operation.new(ir['operation']['type'],
                               ir['operation'].get('parameters', {}),
                               attributes=ir['operation'].get('attributes', {}))
        node = Node(graph, uid(), name, op)
        if 'label' in ir:
            node.update_label(ir['label'])
        return node

    def _dump(self) -> Any:
        ret: Dict[str, Any] = {
            'operation': {
                'type': self.operation.type,
                'parameters': self.operation.parameters,
                'attributes': self.operation.attributes
            }
        }
        if isinstance(self.operation, Cell):
            ret['operation']['cell_name'] = self.operation.cell_name
        if self.label is not None:
            ret['label'] = self.label
        if self.python_name is not None:
            ret['python_name'] = self.python_name
        return ret


class Edge:
    """
    A tensor, or "data flow", between two nodes.

    Example forward code snippet: ::

        a, b, c = split(x)
        p = concat(a, c)
        q = sum(b, p)
        z = relu(q)

    Edges in above snippet: ::

        + head: (split, 0), tail: (concat, 0)  # a in concat
        + head: (split, 2), tail: (concat, 1)  # c in concat
        + head: (split, 1), tail: (sum, -1 or 0)  # b in sum
        + head: (concat, null), tail: (sum, -1 or 1)  # p in sum
        + head: (sum, null), tail: (relu, null)  # q in relu

    Attributes
    ----------
    graph
        Graph.
    head
        Head node.
    tail
        Tail node.
    head_slot
        Index of outputs in head node.
        If the node has only one output, this should be ``null``.
    tail_slot
        Index of inputs in tail node.
        If the node has only one input, this should be ``null``.
        If the node does not care about order, this can be ``-1``.
    """

    def __init__(self, head: EdgeEndpoint, tail: EdgeEndpoint, _internal: bool = False):
        assert _internal, '`Edge()` is private'
        self.graph: Graph = head[0].graph
        self.head: Node = head[0]
        self.tail: Node = tail[0]
        self.head_slot: Optional[int] = head[1]
        self.tail_slot: Optional[int] = tail[1]

    def __repr__(self):
        return f'Edge(head=({self.head}, {self.head_slot}), tail=({self.tail}, {self.tail_slot}))'

    # mutation
    def remove(self) -> None:
        self.graph.edges.remove(self)

    def _register(self) -> 'Edge':
        self.graph.edges.append(self)
        return self

    @classmethod
    def _load(cls, graph: Graph, ir: Any) -> 'Edge':
        head = graph.get_node_by_name(ir['head'][0])
        tail = graph.get_node_by_name(ir['tail'][0])
        assert head is not None and tail is not None
        return Edge((head, ir['head'][1]), (tail, ir['tail'][1]), _internal=True)

    def _dump(self) -> Any:
        return {
            'head': [self.head.name, self.head_slot],
            'tail': [self.tail.name, self.tail_slot]
        }


class IllegalGraphError(ValueError):
    def __init__(self, graph, *args):
        self._debug_dump_graph(graph)
        super().__init__(*args)

    @staticmethod
    def _debug_dump_graph(graph):
        if isinstance(graph, Graph):
            graph = graph._dump()
        with open('generated/debug.json', 'w') as dump_file:
            json.dump(graph, dump_file, indent=4)
