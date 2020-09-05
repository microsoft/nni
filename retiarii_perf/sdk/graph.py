from collections import defaultdict
from enum import Enum
import json
import logging

from . import codegen
from .operations import Operation
from . import utils

_logger = logging.getLogger(__name__)

def _debug_dump_graph(graph, filename = 'debug_error.json'):
    if isinstance(graph, Graph):
        graph = graph.dump()
    with open(filename, 'w') as dump_file:
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
        self.name: 'str' = ""

    ## Mutation Primitives Begin ##
    def add_node(self, type: 'Union[Operation, str]', **parameters) -> 'Node':
        name = 'node_{}'.format(utils.uuid())
        node = Node(self, NodeType.Hidden, name)
        node.set_operation(type, **parameters)
        self.hidden_nodes.append(node)
        return node

    def add_edge(self, head: 'Node', tail: 'Node') -> 'Edge':
        edge = Edge(head, tail)
        self.edges.append(edge)
        return edge

    def remove_edge(self, edge: 'Edge') -> None:
        self.edges.remove(edge)

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

    def get_predecessors(self, node: 'Node') -> 'List[Node]':
        return [edge.head for edge in self.edges if edge.tail is node]

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
            # the order of a sequence of send should be the same as recv
            heads.sort(key=lambda node: (0, '_'.join(node.split('_')[1:3])) if 'send' in node or 'recv' in node else (1, node))
            if not heads:
                _debug_dump_graph(self)
                raise ValueError('Cycle detected in graph {}'.format(self.id))
            head = heads[0]
            rest_nodes.remove(head)
            sorted_nodes.append(head)
            for successor in successors[head]:
                in_degree[successor] -= 1

        nodes = [self.find_node(name) for name in sorted_nodes]
        hidden_and_logical_nodes = [node for node in nodes if (node not in self.input_nodes) and (node not in self.output_nodes)]
        return hidden_and_logical_nodes

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
            open(output_file, 'w').write(script)
            self._script_generated = True


    @staticmethod
    def load(data: 'Any') -> 'Graph':
        try:
            graph = Graph()
            graph.name = data['name']
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
            'name': self.name,
            'graph': data,
            'configs': self.configs,
            'utils': {k: v if isinstance(v, str) else v.dump() for k, v in self.utils.items()},
            'training': self.training
        }

    def dump_utils(self) -> 'dict':
        return {k: v if isinstance(v, str) else v.dump() for k, v in self.utils.items()}
        
    def __eq__(self, other: 'Graph') -> bool:
        return self is other

    def find_multiple_nodes(self, name = None, hashval = None):
        all_nodes = self.input_nodes + self.hidden_nodes + self.output_nodes
        matched_nodes = []
        for node in all_nodes:
            if name != None and node.name != name:
                continue
            if hashval != None and node_hash(node) != hashval:
                continue
            matched_nodes.append(node)
        return matched_nodes

def node_hash(node : "Node") -> 'str':
    name = node.name
    if 'logical_g' in name:
        name = '_'.join(name.split('_')[2:])
    if node.node_type == NodeType.Input:
        data = {'name' : name}
        if len(node.graph.configs) > 0 and 'train_dataloader' in node.graph.configs[0]:
            train_util = node.graph.configs[0]['train_dataloader']
            data['train_util'] = node.graph.utils[train_util].dump()
        if node.attributes:
            data['attributes'] = node.attributes
        return hash(json.dumps(data))
    else:
        data = {'name' : name}
        if node.operation:
            data['operation'] = node.operation.dump()
        if node.attributes:
            data['attributes'] = node.attributes
        return hash(json.dumps(data))

class Node:
    def __init__(self,
            graph: 'Graph',
            node_type: 'NodeType',
            name: 'str',
            operation: 'Optional[Operation]' = None, 
            attributes: 'Dict[str]' = None
    ) -> None:
        self.id: int = utils.uuid()
        self.graph: 'Graph' = graph
        self.node_type: 'NodeType' = node_type
        self.name: str = name
        self.operation: 'Optional[Operation]' = operation
        if attributes == None:
            self.attributes: 'Dict[str]' = {}
        else:
            self.attributes: 'Dict[str]' = attributes

    def get_attribute(self, attr):
        if attr in self.attributes:
            return self.attributes[attr]
        else:
            return None

    def set_attribute(self, attr, val):
        self.attributes[attr] = val

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
            if type is not None and type != self.operation.type:
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
        if 'attributes' in data:
            node.attributes = dict(data['attributes'])
        if 'shape' in data:
            node.shape = list(data['shape'])
        if 'dtype' in data:
            node.shape = list(data['dtype'])
        return node

    def dump(self) -> 'Any':
        data = {'name': self.name}
        if self.operation:
            data['operation'] = self.operation.dump()
        if self.attributes:
            data['attributes'] = self.attributes
        return data

    def __eq__(self, other: 'Node') -> bool:
        return self is other


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
    Logical = 'logical_node'
