from collections import defaultdict
from enum import Enum
import json
import logging

from . import codegen
from .operations import Operation
from . import utils

_logger = logging.getLogger(__name__)

def _debug_dump_graph(graph):
    if isinstance(graph, Graph):
        graph = graph.dump()
    with open('generated/debug.json', 'w') as dump_file:
        json.dump(graph, dump_file, indent=4)


class Function:
    def __init__(self):
        self.function: str = ''
        self.kwargs: 'Dict[str, Any]' = {}

    def __call__(self):
        func = utils.import_(self.function)
        return func(**self.kwargs)

    @staticmethod
    def load(data: 'Any') -> 'Function':
        func = Function()
        func.function = data['function']
        func.kwargs = data.get('kwargs', {})
        return func

    def dump(self) -> 'Any':
        data = {}
        data['function'] = self.function
        data['kwargs'] = self.kwargs
        return data


class Graph:
    def __init__(self):
        self.id: int = utils.uuid()
        self.input_nodes: 'List[Node]' = []
        self.output_nodes: 'List[Node]' = []
        self.hidden_nodes: 'List[Node]' = []
        self.edges: 'List[Edge]' = []
        self.metrics: 'Any' = None
        self._script_generated: bool = False
        self.configs: 'Dict[str, str]' = {}
        self.utils: 'Dict[str, Union[str, dict]]' = {}
        self.training: 'Dict[str, str]' = {}

    ## Mutation Primitives Begin ##

    def add_node(self, type: 'Union[Operation, str]', **parameters) -> 'Node':
        name = 'node_{}'.format(utils.uuid())
        node = Node(self, NodeType.Hidden, name)
        node.set_operation(type, **parameters)
        self.hidden_nodes.append(node)
        return node

    def remove_node(self, node: 'Node') -> None:
        for head in self.get_predecessors(node):
            self.remove_edge(Edge(head, node))
        for tail in self.get_successors(node):
            self.remove_edge(Edge(node, tail))
        self.hidden_nodes.remove(node)

    def add_edge(self, head: 'Node', tail: 'Node') -> 'Edge':
        edge = Edge(head, tail)
        self.edges.append(edge)
        return edge

    def remove_edge(self, edge: 'Edge') -> None:
        self.edges.remove(edge)

    def insert_after(self, target_node: 'Node', new_node: 'Node') -> None:
        """
        insert new_node after target_node, only support the case that there is only one successor of target_node
        """
        successors = self.get_successors(target_node)
        assert len(successors) == 1
        self.hidden_nodes.append(new_node)
        edge = self.find_edge(target_node, successors[0])
        self.remove_edge(edge)
        self.add_edge(target_node, new_node)
        self.add_edge(new_node, successors[0])

    ## Mutation Primitives End ##

    ## Constant Helpers Begin ##

    def find_node(self, node_name: str) -> 'Node':
        import re
        node_name = re.sub('[^0-9a-zA-Z_]', '__', node_name) # TODO: too hacky
        for node in self.input_nodes:
            if node.name == node_name:
                return node
        for node in self.output_nodes:
            if node.name == node_name:
                return node
        for node in self.hidden_nodes:
            if node.name == node_name:
                return node
        raise ValueError('Bad node name "{}"'.format(node_name))

    def find_nodes_by_type(self, node_type: str) -> 'List[Node]':
        found_nodes = []
        for node in self.hidden_nodes:
            if node.operation.type == node_type:
                found_nodes.append(node)
        return found_nodes

    def find_edge(self, head: 'Node', tail: 'Node') -> 'Edge':
        for edge in self.edges:
            if edge.head == head and edge.tail == tail:
                return edge
        return None

    def get_predecessors(self, node: 'Node') -> 'List[Node]':
        return [edge.head for edge in self.edges if edge.tail is node]

    def get_successors(self, node: 'Node') -> 'List[Node]':
        return [edge.tail for edge in self.edges if edge.head is node]

    def topo_sort(self) -> 'List[Node]':
        sorted_nodes = []  # list of nodes' name
        rest_nodes = set()  # set of nodes' name
        in_degree = defaultdict(int)  # node name -> in degree
        successors = defaultdict(list)  # node name -> list of successors' name

        for edge in self.edges:
            in_degree[edge.tail.name] += 1
            successors[edge.head.name].append(edge.tail.name)
            rest_nodes.add(edge.head.name)
            rest_nodes.add(edge.tail.name)

        while rest_nodes:
            heads = [node for node in rest_nodes if in_degree[node] == 0]
            if not heads:
                _debug_dump_graph(self)
                raise ValueError('Cycle detected in graph {}'.format(self.id))
            head = heads[0]
            rest_nodes.remove(head)
            sorted_nodes.append(head)
            for successor in successors[head]:
                in_degree[successor] -= 1

        nodes = [self.find_node(name) for name in sorted_nodes]
        return [node for node in nodes if node.node_type is NodeType.Hidden]

    ## Constant Helpers End ##

    def duplicate(self) -> 'Graph':
        return Graph.load(self.dump())

    def generate_code(self, framework: str, output_file: str = None) -> None:
        if not self._script_generated:
            script = None
            try:
                if framework.lower() in ('tf', 'tensorflow'):
                    script = codegen.graph_to_tensorflow_script(self)
                if framework.lower() == 'pytorch':
                    script = codegen.graph_to_pytorch_script(self)
            except Exception:
                _debug_dump_graph(self)
                raise RuntimeError('Failed to generate code for graph {}'.format(self.id))
            if script is None:
                raise ValueError('Unsupported framework: {}'.format(framework))
            if output_file is None:
                # using default
                output_file = 'generated/graph_{}.py'.format(self.id)
            with open(output_file, 'w') as fh:
                fh.write(script)
            self._script_generated = True


    @staticmethod
    def load(data: 'Any') -> 'Graph':
        try:
            graph = Graph()
            for node_data in data['graph']['inputs']:
                graph.input_nodes.append(Node.load(graph, NodeType.Input, node_data))
            for node_data in data['graph']['outputs']:
                graph.output_nodes.append(Node.load(graph, NodeType.Output, node_data))
            for node_data in data['graph']['hidden_nodes']:
                graph.hidden_nodes.append(Node.load(graph, NodeType.Hidden, node_data))
            for edge in data['graph']['edges']:
                head = graph.find_node(edge['head'])
                tail = graph.find_node(edge['tail'])
                graph.edges.append(Edge(head, tail))
            graph.configs = data.get('configs', {})
            utils = {k: v if isinstance(v, str) else Function.load(v) \
                for k, v in data.get('utils', {}).items()}
            graph.utils = utils
            graph.training = data.get('training', {})
            return graph
        except Exception as e:
            # TODO: show the error here
            _logger.error(logging.exception('message'))
            _debug_dump_graph(data)
            # TODO: this raised error is not gotten by anyone
            raise ValueError('Bad graph')

    def dump(self) -> 'Any':
        data = {}
        data['inputs'] = [node.dump() for node in self.input_nodes]
        data['outputs'] = [node.dump() for node in self.output_nodes]
        data['hidden_nodes'] = [node.dump() for node in self.hidden_nodes]
        data['edges'] = []
        for edge in self.edges:
            data['edges'].append({'head': edge.head.name, 'tail': edge.tail.name})
        return {
            'graph': data,
            'configs': self.configs,
            'utils': {k: v if isinstance(v, str) else v.dump() for k, v in self.utils.items()},
            'training': self.training
        }

    def __eq__(self, other: 'Graph') -> bool:
        return self is other


class Node:
    def __init__(self,
            graph: 'Graph',
            node_type: 'NodeType',
            name: 'str',
            operation: 'Optional[Operation]' = None
    ) -> None:
        self.id: int = utils.uuid()
        self.graph: 'Graph' = graph
        self.node_type: 'NodeType' = node_type
        self.name: str = name
        self.operation: 'Optional[Operation]' = operation

    def set_operation(self, type: 'Union[Operation, str]', **parameters) -> None:
        if isinstance(type, Operation):
            self.operation = type
        else:
            self.operation = Operation.new(type, **parameters)

    def update_operation(self, type: 'Union[Operation, str]', **parameters) -> None:
        """
        if type is None, does not change type then
        """
        if isinstance(type, Operation):
            self.operation = type
        else:
            if self.operation is None or (type is not None and type != self.operation.type):
                self.operation = Operation.new(type, **parameters)
            else:
                self.operation.update_params(**parameters)

    def update_operation_super(self, param_choices: 'List') -> None:
        """
        directly update the node to nni mutable LayerChoice
        """
        op_choices = []
        for choice in param_choices:
            params = self.operation.params.copy()
            params.update(choice)
            params_str = ''
            for k, v in params.items():
                params_str += '{}={}, '.format(k, repr(v))
            op_choices.append('{}({})'.format(self.operation.type, params_str[:-2]))
        self.operation = Operation.new('LayerChoice', **{'op_candidates': op_choices})

    @staticmethod
    def load(graph, node_type: 'NodeType', data: 'Any') -> 'Node':
        node = Node(graph, node_type, data['name'])
        if node_type is NodeType.Hidden:
            node.operation = Operation.load(data['operation'])
        return node

    def dump(self) -> 'Any':
        data = {'name': self.name}
        if self.operation:
            data['operation'] = self.operation.dump()
        return data

    def __eq__(self, other: 'Node') -> bool:
        return self is other

    def __repr__(self):
        return 'node type: {}, name: {}, operations: {}'.format(self.node_type, self.name, self.operation)

class Edge:
    def __init__(self, head: Node, tail: Node) -> None:
        self.graph: Graph = head.graph
        self.head: Node = head
        self.tail: Node = tail

    def __eq__(self, other: 'Edge') -> bool:
        return self.head is other.head and self.tail is other.tail


class NodeType(Enum):
    Input = 'input_node'
    Output = 'output_node'
    Hidden = 'hidden_node'
