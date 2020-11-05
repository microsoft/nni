"""
Model representation.
"""

import copy
from enum import Enum
import json
from typing import (Any, Dict, List, Optional, Tuple, Union, overload)

from .operation import Cell, Operation, _PseudoOperation

__all__ = ['Model', 'ModelStatus', 'Graph', 'Node', 'Edge', 'IllegalGraphError', 'MetricData']


MetricData = Any
"""
Graph metrics like loss, accuracy, etc.

# Maybe we can assume this is a single float number for first iteration.
"""


class TrainingConfig:
    """
    Training training_config of a model.

    Module will be imported, initialized with generated model and arguments in ``kwargs``.

    Attributes
    ----------
    module
        Trainer module
    kwargs
        Trainer keyword arguments
    """

    def __init__(self, module: str, kwargs: Dict[str, Any]):
        self.module = module
        self.kwargs = kwargs

    def __repr__(self):
        return f'TrainingConfig(module={self.module}, kwargs={self.kwargs})'

    @staticmethod
    def _load(ir: Any) -> 'TrainingConfig':
        return TrainingConfig(ir['module'], ir.get('kwargs', {}))

    def _dump(self) -> Any:
        return {
            'module': self.module,
            'kwargs': self.kwargs
        }


class Model:
    """
    Represents a neural network model.

    During mutation, one `Model` object is created for each trainable snapshot.
    For example, consider a mutator that insert a node at an edge for each iteration.
    In one iteration, the mutator invokes 4 primitives: add node, remove edge, add edge to head, add edge to tail.
    These 4 primitives operates in one `Model` object.
    When they are all done the model will be set to "frozen" (trainable) status and be submitted to execution engine.
    And then a new iteration starts, and a new `Model` object is created by forking last model.

    Attributes
    ----------
    status
        See `ModelStatus`.
    root_graph
        The outermost graph which usually takes dataset as input and feeds output to loss function.
    graphs
        All graphs (subgraphs) in this model.
    training_config
        Training config
    history
        Mutation history.
        `self` is directly mutated from `self.history[-1]`;
        `self.history[-1] is mutated from `self.history[-2]`, and so on.
        `self.history[0]` is the base graph.
    metric
        Training result of the model, or `None` if it's not yet trained or has failed to train.
    intermediate_metrics
        Intermediate training metrics. If the model is not trained, it's an empty list.
    """
    _cur_model_id = 0

    def __init__(self, _internal=False):
        assert _internal, '`Model()` is private, use `model.fork()` instead'
        Model._cur_model_id += 1
        self.model_id = Model._cur_model_id

        self.status: ModelStatus = ModelStatus.Mutating

        self._root_graph_name: str = '_model'
        self.graphs: Dict[str, Graph] = {}
        self.training_config: TrainingConfig = TrainingConfig('foo', {})

        self.history: List[Model] = []

        self.metric: Optional[MetricData] = None
        self.intermediate_metrics: List[MetricData] = []

        self._last_uid: int = 0  # FIXME: this should be global, not model-wise

    def __repr__(self):
        return f'Model(model_id={self.model_id}, status={self.status}, graphs={list(self.graphs.keys())}, ' + \
            f'training_config={self.training_config}, metric={self.metric}, intermediate_metrics={self.intermediate_metrics})'

    @property
    def root_graph(self) -> 'Graph':
        return self.graphs[self._root_graph_name]

    def fork(self) -> 'Model':
        """
        Create a new model which has same topology, names, and IDs to current one.

        Can only be invoked on a frozen model.
        The new model will be in `Mutating` state.

        This API is used in mutator base class.
        """
        new_model = Model(_internal=True)
        new_model._root_graph_name = self._root_graph_name
        new_model.graphs = {name: graph._fork_to(new_model) for name, graph in self.graphs.items()}
        new_model.training_config = copy.deepcopy(self.training_config)
        new_model.history = self.history + [self]
        new_model._last_uid = self._last_uid
        return new_model

    def _uid(self) -> int:
        self._last_uid += 1
        return self._last_uid

    @staticmethod
    def _load(ir: Any) -> 'Model':
        model = Model(_internal=True)
        for graph_name, graph_data in ir.items():
            if graph_name != '_training_config':
                Graph._load(model, graph_name, graph_data)._register()
        model.training_config = TrainingConfig._load(ir['_training_config'])
        return model

    def _dump(self) -> Any:
        ret = {name: graph._dump() for name, graph in self.graphs.items()}
        ret['_training_config'] = self.training_config._dump()
        return ret

    def get_nodes_by_label(self, label: str) -> List['Node']:
        """
        Traverse all the nodes to find the matched node(s) with the given name.
        There could be multiple nodes with the same name. Name space name can uniquely
        identify a graph or node.

        NOTE: the implementation does not support the class abstration
        """
        matched_nodes = []
        for graph in self.graphs:
            nodes = graph.get_nodes_by_label(label)
            matched_nodes.extend(nodes)
        return matched_nodes

    def get_by_name(self, name: str) -> Union['Graph', 'Node']:
        """
        Find the graph or node that have the given name space name.
        """


class ModelStatus(Enum):
    """
    The status of model.

    A model is created in `Mutating` status.
    When the mutation is done and the model get ready to train, its status becomes `Frozen`.
    When training started, the model's status becomes `Training`.
    If training is successfully ended, model's `metric` attribute get set and its status becomes `Trained`.
    If training failed, the status becomes `Failed`.
    """
    Mutating = "mutating"
    Frozen = "frozen"
    Training = "training"
    Trained = "trained"
    Failed = "failed"


_InputPseudoUid = -1
_OutputPseudoUid = -2


class Graph:
    """
    Graph topology.

    This class simply represents the topology, with no semantic meaning.
    All other information like metric, non-graph functions, mutation history, etc should go to `Model`.

    Each graph belongs to and only belongs to one `Model`.

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
        ...
    output_node
        ...
    hidden_nodes
        ...
    nodes
        All input/output/hidden nodes.
    edges
        ...
    """

    def __init__(self, model: Model, graph_id: int, name: str = None, _internal: bool = False):
        assert _internal, '`Graph()` is private'

        self.model: Model = model
        self.id: int = graph_id
        self.name: str = name or f'_generated_{graph_id}'

        # TODO: why not merge the names into input_node and output_node???
        self.input_names: Optional[List[str]] = None
        self.output_names: Optional[List[str]] = None

        self.input_node: Node = Node(self, _InputPseudoUid, '_inputs', _PseudoOperation('_inputs'), _internal=True)
        self.output_node: Node = Node(self, _OutputPseudoUid, '_outputs', _PseudoOperation('_outputs'), _internal=True)
        self.hidden_nodes: List[Node] = []

        self.edges: List[Edge] = []

    def __repr__(self):
        return f'Graph(id={self.id}, name={self.name}, input_names={self.input_names}, ' + \
            f'output_names={self.output_names}, num_hidden_nodes={len(self.hidden_nodes)}, num_edges={len(self.edges)})'

    @property
    def nodes(self) -> List['Node']:
        return [self.input_node, self.output_node] + self.hidden_nodes

    def _add_input(self, input_name) -> None:
        if self.input_names is None:
            self.input_names = [input_name]
        else:
            self.input_names.append(input_name)

    def _add_output(self, output_name) -> None:
        if self.output_names is None:
            self.output_names = [output_name]
        else:
            self.output_names.append(output_name)

    @overload
    def add_node(self, name: str, operation: Operation) -> 'Node': ...
    @overload
    def add_node(self, name: str, type_name: str, parameters: Dict[str, Any] = {}) -> 'Node': ...

    def add_node(self, name, operation_or_type, parameters={}):
        if isinstance(operation_or_type, Operation):
            op = operation_or_type
        else:
            op = Operation.new(operation_or_type, name, parameters)
        return Node(self, self.model._uid(), name, op, _internal=True)._register()

    # mutation
    def add_edge(self, head: Tuple['Node', Optional[int]], tail: Tuple['Node', Optional[int]]) -> 'Edge':
        assert head[0].graph is self and tail[0].graph is self
        return Edge(head, tail, _internal=True)._register()

    def insert_node_after(self, node: 'Node', name: str, type: Union[Operation, str], **parameters) -> 'Node':
        if isinstance(type, Operation):
            assert not parameters
            op = type
        else:
            op = Operation.new(type, cell_name=name, **parameters)
        new_node = Node(self, self.model._uid(), name, op, _internal=True)._register()
        return new_node

    def get_node_by_name(self, name: str) -> Optional['Node']:
        """
        Returns the node which has specified name; or returns `None` if no node has this name.
        """
        found = [node for node in self.nodes if node.name == name]
        return found[0] if found else None

    def get_nodes_by_type(self, operation_type: str) -> List['Node']:
        """
        Returns nodes whose operation is specified typed.
        """
        return [node for node in self.hidden_nodes if node.operation.type == operation_type]

    def get_nodes_by_label(self, label: str) -> List['Node']:
        return [node for node in self.hidden_nodes if node.label == label]

    def topo_sort(self) -> List['Node']:  # TODO
        ...

    def fork(self) -> 'Graph':
        """
        Fork the model and returns corresponding graph in new model.
        This shortcut might be helpful because many algorithms only cares about "stem" subgraph instead of whole model.
        """
        return self.model.fork().graphs[self.name]

    def __eq__(self, other: object) -> bool:
        return self is other

    def _fork_to(self, model: Model) -> 'Graph':
        new_graph = Graph(model, self.id, self.name, _internal=True)._register()
        new_graph.input_names = self.input_names
        new_graph.output_names = self.output_names

        for node in self.hidden_nodes:
            Node(new_graph, node.id, node.name, node.operation, _internal=True)._register()

        id_to_new_node = {node.id: node for node in new_graph.nodes}

        for edge in self.edges:
            new_head = id_to_new_node[edge.head.id]
            new_tail = id_to_new_node[edge.tail.id]
            Edge((new_head, edge.head_slot), (new_tail, edge.tail_slot), _internal=True)._register()

        return new_graph

    def _copy(self) -> 'Graph':
        # Copy this graph inside the model.
        # The new graph will have identical topology, but its nodes' name and ID will be different.
        new_graph = Graph(self.model, self.model._uid(), _internal=True)._register()
        new_graph.input_names = self.input_names
        new_graph.output_names = self.output_names

        id_to_new_node = {}  # old node ID -> new node object

        for old_node in self.hidden_nodes:
            new_node = Node(new_graph, self.model._uid(), None, old_node.operation, _internal=True)._register()
            id_to_new_node[old_node.id] = new_node

        for edge in self.edges:
            new_head = id_to_new_node[edge.head.id]
            new_tail = id_to_new_node[edge.tail.id]
            Edge((new_head, edge.head_slot), (new_tail, edge.tail_slot), _internal=True)._register()

        return new_graph

    def _register(self) -> 'Graph':
        self.model.graphs[self.name] = self
        return self

    @staticmethod
    def _load(model: Model, name: str, ir: Any) -> 'Graph':
        graph = Graph(model, model._uid(), name, _internal=True)
        graph.input_names = ir.get('inputs')
        graph.output_names = ir.get('outputs')
        for node_name, node_data in ir['nodes'].items():
            Node._load(graph, node_name, node_data)._register()
        for edge_data in ir['edges']:
            Edge._load(graph, edge_data)._register()
        return graph

    def _dump(self) -> Any:
        return {
            'inputs': self.input_names,
            'outputs': self.output_names,
            'nodes': {node.name: node._dump() for node in self.hidden_nodes},
            'edges': [edge._dump() for edge in self.edges]
        }


class Node:
    """
    An operation or an opaque subgraph inside a graph.

    Each node belongs to and only belongs to one `Graph`.
    Nodes should never be created with constructor. Use `Graph.add_node()` instead.

    The node itself is for topology only.
    Information of tensor calculation should all go inside `operation` attribute.

    TODO: parameter of subgraph (cell)
    It's easy to assign parameters on cell node, but it's hard to "use" them.
    We need to design a way to reference stored cell parameters in inner node operations.
    e.g. `self.fc = Linear(self.units)`  <-  how to express `self.units` in IR?

    Attributes
    ----------
    graph
        The graph containing this node.
    id
        Unique ID in the model.
        If two models have nodes with same ID, they are semantically the same node.
    name
        Mnemonic name. It should have an one-to-one mapping with ID.
    operation
        ...
    cell
        Read only shortcut to get the referenced subgraph.
        If this node is not a subgraph (is a primitive operation), accessing `cell` will raise an error.
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
        self.operation: Operation = operation
        self.label: str = None

    def __repr__(self):
        return f'Node(id={self.id}, name={self.name}, label={self.label}, operation={self.operation})'

    @property
    def predecessors(self) -> List['Node']:
        return sorted(set(edge.head for edge in self.incoming_edges), key=(lambda node: node.id))

    @property
    def successors(self) -> List['Node']:
        return sorted(set(edge.tail for edge in self.outgoing_edges), key=(lambda node: node.id))

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

    def update_label(self, label: str) -> None:
        self.label = label

    @overload
    def update_operation(self, operation: Operation) -> None: ...
    @overload
    def update_operation(self, type_name: str, parameters: Dict[str, Any] = {}) -> None: ...

    def update_operation(self, operation_or_type, parameters={}):
        if isinstance(operation_or_type, Operation):
            self.operation = operation_or_type
        else:
            self.operation = Operation.new(operation_or_type, parameters)

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

    def _register(self) -> 'Node':
        self.graph.hidden_nodes.append(self)
        return self

    @staticmethod
    def _load(graph: Graph, name: str, ir: Any) -> 'Node':
        if ir['operation']['type'] == '_cell':
            op = Cell(ir['operation']['cell'], ir['operation'].get('parameters', {}))
        else:
            op = Operation.new(ir['operation']['type'], ir['operation'].get('parameters', {}))
        return Node(graph, graph.model._uid(), name, op)

    def _dump(self) -> Any:
        ret = {'operation': {'type': self.operation.type, 'parameters': self.operation.parameters}}
        if isinstance(self.operation, Cell):
            ret['operation']['cell_name'] = self.operation.cell_name
        if self.label is not None:
            ret['label'] = self.label
        return ret


class Edge:
    """
    A tensor, or "data flow", between two nodes.

    Example forward code snippet:
    ```
        a, b, c = split(x)
        p = concat(a, c)
        q = sum(b, p)
        z = relu(q)
    ```

    Edges in above snippet:
      + head: (split, 0), tail: (concat, 0)  # a in concat
      + head: (split, 2), tail: (concat, 1)  # c in concat
      + head: (split, 1), tail: (sum, -1 or 0)  # b in sum
      + head: (concat, null), tail: (sum, -1 or 1)  # p in sum
      + head: (sum, null), tail: (relu, null)  # q in relu

    Attributes
    ----------
    graph
        ...
    head
        Head node.
    tail
        Tail node.
    head_slot
        Index of outputs in head node.
        If the node has only one output, this should be `null`.
    tail_slot
        Index of inputs in tail node.
        If the node has only one input, this should be `null`.
        If the node does not care about order, this can be `-1`.
    """

    def __init__(
            self,
            head: Tuple[Node, Optional[int]],
            tail: Tuple[Node, Optional[int]],
            _internal: bool = False):
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

    @staticmethod
    def _load(graph: Graph, ir: Any) -> 'Edge':
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
            graph = graph.dump()
        with open('generated/debug.json', 'w') as dump_file:
            json.dump(graph, dump_file, indent=4)
