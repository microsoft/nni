from collections import defaultdict
import json
import logging

from .operations_tf import Operation
from .type_utils import *
from . import utils

_logger = logging.getLogger(__name__)

def _debug_dump_graph(graph):
    if isinstance(graph, Graph):
        graph = graph.dump()
    with open('generated/debug.json', 'w') as dump_file:
        json.dump(graph, dump_file, indent=4)

EdgeEnd = Tuple['Node', Optional[int]]


class Function:
    def __init__(self):
        self.function: str = ''
        self.kwargs: Dict[str, Any] = {}

    def __call__(self):
        func = utils.import_(self.function)
        return func(**self.kwargs)

    @staticmethod
    def load(data: Any) -> 'Function':
        func = Function()
        if isinstance(data, str):
            func.function = data
        else:
            func.function = data['function']
            func.kwargs = data.get('kwargs', {})
        return func

    def dump(self) -> Any:
        data: Any = {}
        data['function'] = self.function
        data['kwargs'] = self.kwargs
        return data


class Graph:
    def __init__(
            self,
            name: str,
            cell_templates: Dict[str, 'Graph'],
            config: Dict[str, str],
            funcs: Dict[str, Function]) -> None:

        self.id: int = utils.uuid()
        self.name: str = name

        self.input_names: Optional[List[str]] = None
        self.output_names: Optional[List[str]] = None

        self.nodes: List[Node] = [Node(self, '_inputs'), Node(self, '_outputs')]
        self.edges: List[Edge] = []

        # these are "global" dicts, shared by all graphs (cell templates) of current snapshot
        self.config: Dict[str, str] = config
        self.utils: Dict[str, Function] = funcs
        self.cell_templates: Dict[str, Graph] = cell_templates

        # TODO: move to "running graph"  @quanlu
        self.metrics: Any = None
        self.training: Dict[str, str] = {}

        # TODO: move out
        self._script_generated: bool = False

    ## Mutation Primitives Begin ##

    def add_node(self, type: Union[Operation, str], **parameters) -> 'Node':
        name = 'node_{}'.format(utils.uuid())
        node = Node(self, name)
        node.set_operation(type, **parameters)
        self.nodes.append(node)
        return node

    def add_edge(self, head: Union[EdgeEnd, 'Node'], tail: Union[EdgeEnd, 'Node']) -> 'Edge':
        edge = Edge(_decode_edge_end(self, head), _decode_edge_end(self, tail))
        self.edges.append(edge)
        return edge

    def remove_edge(self, edge: 'Edge') -> None:
        self.edges.remove(edge)

    ## Mutation Primitives End ##

    ## Constant Helpers Begin ##

    @property
    def input_node(self) -> 'Node':
        return self.nodes[0]

    @property
    def output_node(self) -> 'Node':
        return self.nodes[1]

    @property
    def hidden_nodes(self) -> List['Node']:
        return self.nodes[2:]

    def find_node(self, node_name: str) -> 'Node':
        found = [node for node in self.nodes if node.name == node_name]
        assert len(found) == 1, 'Found {} nodes of name "{}"'.format(len(found), node_name)
        return found[0]

    def find_nodes_by_type(self, operation_type: str) -> List['Node']:
        return [node for node in self.nodes if node.operation_type == operation_type]

    def get_predecessors(self, node: 'Node') -> List['Node']:
        return [edge.head for edge in self.edges if edge.tail is node]

    def get_incoming_edges(self, node: 'Node') -> List['Edge']:
        return [edge for edge in self.edges if edge.tail is node]

    def topo_sort(self) -> List['Node']:
        sorted_names = []
        rest_names = set()
        in_degree: Dict[str, int] = defaultdict(int)  # node/cell name -> in degree
        successors = defaultdict(list)  # node/cell name -> list of successors' name

        for edge in self.edges:
            in_degree[edge.tail.name] += 1
            successors[edge.head.name].append(edge.tail.name)
            rest_names.add(edge.head.name)
            rest_names.add(edge.tail.name)

        while rest_names:
            heads = [name for name in rest_names if in_degree[name] == 0]
            if not heads:
                _debug_dump_graph(self)
                raise BadGraph(self, 'Cycle detected in graph {}'.format(self.id))
            head = heads[0]
            rest_names.remove(head)
            sorted_names.append(head)
            for successor in successors[head]:
                in_degree[successor] -= 1

        return [self.find_node(name) for name in sorted_names if not name.startswith('_')]

    ## Constant Helpers End ##

    def duplicate(self) -> 'Graph':
        # FIXME: should duplicate all cells as well
        return Graph.load(self.dump(), self.name)

    # TODO: move out
    def generate_code(self, framework: str, output_file: str = None) -> None:
        from . import codegen
        if not self._script_generated:
            script = None
            try:
                if framework.lower() in ('tf', 'tensorflow'):
                    script = codegen.graph_to_tensorflow_script(self)
                if framework.lower() in ('pytorch', 'torch'):
                    script = codegen.graph_to_pytorch_script(self)
            except Exception as e:
                _logger.exception(e)
                _debug_dump_graph(self)
                raise RuntimeError('Failed to generate code for graph {}'.format(self.id))
            if script is None:
                raise ValueError('Unsupported framework: {}'.format(framework))
            if output_file is None:
                # using default
                output_file = 'generated/graph_{}.py'.format(self.id)
            open(output_file, 'w').write(script)
            self._script_generated = True


    @staticmethod
    def load(all_data: Any, graph_name: str = '_graph') -> 'Graph':
        cell_templates: Dict[str, Graph] = {}
        config = all_data.get('_config', {})
        utils = {name: Function.load(data) for name, data in all_data.get('_utils', {}).items()}
        for name, graph_data in all_data.items():
            if name == '_graph' or not name.startswith('_'):
                cell_templates[name] = Graph(name, cell_templates, config, utils)
                cell_templates[name]._load(graph_data)
        return cell_templates[graph_name]

    def _load(self, data: Any) -> None:
        try:
            if 'inputs' in data:
                self.input_names = data['inputs']
            if 'outputs' in data:
                self.output_names = data['outputs']
            for name, node_data in data['hidden_nodes'].items():
                self.nodes.append(Node.load(self, name, node_data))
            for edge_data in data['edges']:
                self.edges.append(Edge.load(self, edge_data))
        except Exception as e:
            _logger.exception(e)
            _debug_dump_graph(data)
            raise ValueError('Bad graph IR')

    def dump(self) -> Any:
        data: Any = {}
        for name, graph in self.cell_templates.items():
            data[name] = graph._dump_graph()
        data['_config'] = self.config
        data['_utils'] = {name: util.dump() for name, util in self.utils.items()}
        return data

    def _dump_graph(self) -> Any:
        data: Any = {}
        data['inputs'] = self.input_names
        data['outputs'] = self.output_names
        data['hidden_nodes'] = {node.name: node.dump() for node in self.nodes if not node.name.startswith('_')}
        data['edges'] = [edge.dump() for edge in self.edges]
        return data

    def __eq__(self, other: object) -> bool:
        return self is other


class Node:
    def __init__(self, graph: Graph, name: str) -> None:
        self.id: int = utils.uuid()
        self.graph: Graph = graph
        self.name: str = name
        self.operation: Optional[Operation] = None

    def set_operation(self, type: Union[Operation, str], **parameters) -> None:
        if isinstance(type, Operation):
            self.operation = type
        else:
            self.operation = Operation.new(type, **parameters)

    def update_operation(self, type: Union[Operation, str], **parameters) -> None:
        if isinstance(type, Operation):
            self.operation = type
        elif self.operation_type != type:
            self.operation = Operation.new(type, **parameters)
        else:
            cast(Operation, self.operation).update_params(**parameters)

    @property
    def operation_type(self) -> Optional[str]:
        return self.operation.type if self.operation is not None else None

    @staticmethod
    def load(graph, name: str, operation_data: Any) -> 'Node':
        if operation_data['type'] == '_cell':
            return Cell.load(graph, name, operation_data)
        node = Node(graph, name)
        node.operation = Operation.new(**operation_data)
        return node

    def dump(self) -> Any:
        assert self.operation is not None
        return self.operation.dump()

    def __eq__(self, other: object) -> bool:
        return self is other


class Cell(Node):
    def __init__(self, graph: Graph, name: str, template_name: str) -> None:
        self.id: int = utils.uuid()
        self.graph: Graph = graph
        self.name: str = name
        self.template_name: str = template_name
        self.operation: None = None

    def set_template(self, template: Union[Graph, str]) -> None:
        self.template_name = template if isinstance(template, str) else template.name

    @property
    def template(self) -> Graph:
        return self.graph.cell_templates[self.template_name]

    @staticmethod
    def load(graph: Graph, name: str, data: Any) -> 'Cell':
        return Cell(graph, name, data['template'])

    def dump(self) -> Any:
        return {'type': '_cell', 'template': self.template_name}


class Edge:
    def __init__(self, head: EdgeEnd, tail: EdgeEnd) -> None:
        self.graph: Graph = head[0].graph
        self.head: Node = head[0]
        self.head_idx: Optional[int] = head[1]
        self.tail: Node = tail[0]
        self.tail_idx: Optional[int] = tail[1]

    def __eq__(self, other: object) -> bool:
        s = (self.head, self.head_idx, self.tail, self.tail_idx)
        if isinstance(other, Edge):
            return s == (other.head, other.head_idx, other.tail, other.tail_idx)
        else:
            return s == other

    @staticmethod
    def load(graph: Graph, data: Any) -> 'Edge':
        head = _decode_edge_end(graph, data['head'])
        tail = _decode_edge_end(graph, data['tail'])
        return Edge(head, tail)

    def dump(self) -> Any:
        head = self.head.name if self.head_idx is None else [self.head.name, self.head_idx]
        tail = self.tail.name if self.tail_idx is None else [self.tail.name, self.tail_idx]
        return {'head': head, 'tail': tail}

def _decode_edge_end(graph: Graph, edge_end: Any) -> EdgeEnd:
    if isinstance(edge_end, str):
        if graph.input_names and edge_end in graph.input_names:
            return (graph.input_node, graph.input_names.index(edge_end))
        if graph.output_names and edge_end in graph.output_names:
            return (graph.output_node, graph.output_names.index(edge_end))
        return (graph.find_node(edge_end), None)
    if isinstance(edge_end, Node):
        return (edge_end, None)
    if isinstance(edge_end[0], str):
        return (graph.find_node(edge_end[0]), edge_end[1])
    return edge_end


class BadGraph(ValueError):
    def __init__(self, graph, *args):
        _debug_dump_graph(graph)
        super().__init__(*args)
