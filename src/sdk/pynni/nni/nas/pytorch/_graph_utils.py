import json
import re
from collections import defaultdict

import torch
from torch.utils.tensorboard._pytorch_graph import NodePyIO, NodeBase, GraphPy, CLASSTYPE_KIND, GETATTR_KIND, NodePyOP


def _graph(model, args, verbose=False):
    """
    Copy from ``torch.utils.tensorboard._pytorch_graph.graph``.
    Copied because we don't need to convert to proto.
    This works on PyTorch 1.4. We tried PyTorch 1.3, and it didn't work.

    Original docstring:

    This method processes a PyTorch model and produces a `GraphDef` proto
    that can be logged to TensorBoard.

    Args:
        model (PyTorch module): The model to be parsed.
        args (tuple): input tensor[s] for the model.
        verbose (bool): Whether to print out verbose information while processing.
    """

    def parse(graph, trace, args=None, omit_useless_nodes=True):
        """This method parses an optimized PyTorch model graph and produces
        a list of nodes and node stats for eventual conversion to TensorBoard
        protobuf format.

        Args:
            graph (PyTorch module): The model graph to be parsed.
            trace (PyTorch JIT TracedModule): The model trace to be parsed.
            args (tuple): input tensor[s] for the model.
            omit_useless_nodes (boolean): Whether to remove nodes from the graph.
        """
        n_inputs = len(args)

        scope = {}
        nodes_py = GraphPy()
        for node in graph.inputs():
            if omit_useless_nodes:
                if len(node.uses()) == 0:  # number of user of the node (= number of outputs/ fanout)
                    continue

            if node.type().kind() != CLASSTYPE_KIND:
                nodes_py.append(NodePyIO(node, 'input'))

        attr_to_scope = dict()
        for node in graph.nodes():
            if node.kind() == GETATTR_KIND:
                attr_name = node.s('name')
                parent = node.input().node()
                if parent.kind() == GETATTR_KIND:  # If the parent node is not the top-level "self" node
                    parent_attr_name = parent.s('name')
                    parent_scope = attr_to_scope[parent_attr_name]
                    attr_scope = parent_scope.split('/')[-1]
                    attr_to_scope[attr_name] = '{}/{}.{}'.format(parent_scope, attr_scope, attr_name)
                else:
                    attr_to_scope[attr_name] = '__module.{}'.format(attr_name)
                # We don't need classtype nodes; scope will provide this information
                if node.output().type().kind() != CLASSTYPE_KIND:
                    node_py = NodePyOP(node)
                    node_py.scopeName = attr_to_scope[attr_name]
                    nodes_py.append(node_py)
            else:
                nodes_py.append(NodePyOP(node))

        for i, node in enumerate(graph.outputs()):  # Create sink nodes for output ops
            node_py = NodePyIO(node, 'output')
            node_py.debugName = "output.{}".format(i + 1)
            node_py.inputs = [node.debugName()]
            nodes_py.append(node_py)

        def parse_traced_name(module_name):
            prefix = 'TracedModule['
            suffix = ']'
            if module_name.startswith(prefix) and module_name.endswith(suffix):
                module_name = module_name[len(prefix):-len(suffix)]
            return module_name

        alias_to_name = dict()
        base_name = parse_traced_name(trace._name)
        for name, module in trace.named_modules(prefix='__module'):
            mod_name = parse_traced_name(module._name)
            attr_name = name.split('.')[-1]
            alias_to_name[name] = '{}[{}]'.format(mod_name, attr_name)

        for node in nodes_py.nodes_op:
            module_aliases = node.scopeName.split('/')
            replacements = [
                alias_to_name[alias]
                if alias in alias_to_name
                else alias.split('.')[-1]
                for alias in module_aliases
            ]
            node.scopeName = base_name
            if any(replacements):
                node.scopeName += '/' + '/'.join(replacements)

        nodes_py.populate_namespace_from_OP_to_IO()
        return nodes_py

    with torch.onnx.set_training(model, False):
        try:
            trace = torch.jit.trace(model, args)
            graph = trace.graph
            torch._C._jit_pass_inline(graph)
        except RuntimeError as e:
            print(e)
            print('Error occurs, No graph saved')
            raise e

    if verbose:
        print(graph)
    list_of_nodes = parse(graph, trace, args)
    return list_of_nodes


class VisNode:
    def __init__(self, id, type, text, kind=None):
        assert type in ["input", "output", "op", "blob", "hub"]
        self.id = id
        self.text = text
        self.type = type
        self.kind = kind

    def to_json(self):
        return {"id": self.id, "text": self.text, "type": self.type, "kind": self.kind}


class VisGraph:
    def __init__(self, graph: GraphPy, module_name2key: dict):
        self.nodes = dict()
        self.edges = set()
        self.key2chain = dict()
        self.build(graph, module_name2key)

    def build(self, graph, module_name2key):
        self.default_display_treeset = {"input", "output"}
        for scope_name in graph.unique_name_to_scoped_name.values():
            # scope is an op, find corresponding father
            path = scope_name.split("/")
            if len(path) >= 1 and (
                    path[-2].startswith("LayerChoice[") or
                    path[-2].startswith("InputChoice[")):
                for i in range(1, len(path)):
                    self.default_display_treeset.add("/".join(path[:i]))

        self.oriid2visid = dict()
        for node_id, node in graph.nodes_io.items():
            id_in_graph = self.add_complex_node(node_id, node)
            self.oriid2visid[node_id] = id_in_graph
        for node_id, node in graph.nodes_io.items():
            for input_name in node.inputs:
                input_id = input_name.split("/")[-1]
                if input_id in self.oriid2visid:
                    self.add_edge(self.oriid2visid[input_id], self.oriid2visid[node_id])
        self.eliminate_sidechain_nodes()

        key2chain = dict()
        for node_id, node in graph.nodes_io.items():
            if ("op." + node_id) not in self.nodes:
                # filter out unused ListConstruct
                continue
            if node.scope:
                scope_type = self._extract_scope_type(node.scope)[-1]
                if (scope_type == "LayerChoice" or scope_type == "InputChoice") and node.kind == "prim::ListConstruct":
                    # backtrack until out of scope
                    module_name = self._extract_module_name(node.scope)
                    module_key = module_name2key[module_name]
                    chain_list = self._retrieve_chains(graph, node)
                    if module_key not in key2chain:
                        key2chain[module_key] = [set() for _ in range(len(node.inputs))]
                    for i, c in enumerate(chain_list):
                        key2chain[module_key][i] |= c
        self.key2chain = {k: [list(t) for t in v] for k, v in key2chain.items()}

    def add_single_node(self, node_id: str, node: NodeBase):
        if node_id not in self.nodes:
            if isinstance(node, NodePyIO):
                new_node = VisNode(node_id, node.input_or_output, node.input_or_output)
            elif isinstance(node, NodeBase):
                module_name = self._extract_module_name(node.debugName)
                if module_name:
                    module_name += "."
                new_node = VisNode(node_id, "blob", module_name + node.kind.split("::")[-1], kind=node.kind)
            else:
                raise NotImplementedError
            self.nodes[node_id] = new_node
        return self.nodes[node_id]

    def add_hub_node(self, node_id: str, text):
        if node_id not in self.nodes:
            self.nodes[node_id] = VisNode(node_id, "hub", text)
        return self.nodes[node_id]

    def add_edge(self, src_node, dst_node):
        if src_node == dst_node:
            if self.nodes[src_node].type == "hub":
                return None
        self.edges.add((src_node, dst_node))
        return (src_node, dst_node)

    def add_complex_node(self, node_id, node: NodeBase):
        scope_name, is_hub_node = self._reformat_node_name(node.debugName)
        if is_hub_node:
            node_id = "hub." + scope_name
            self.add_hub_node(node_id, self._extract_module_name(scope_name))
        else:
            node_id = "op." + node_id
            self.add_single_node(node_id, node)
        return node_id

    def adjacency_list(self, reverse=False):
        adj_list = defaultdict(list)
        for src, dst in self.edges:
            if reverse:
                adj_list[dst].append(src)
            else:
                adj_list[src].append(dst)
        return adj_list

    def eliminate_sidechain_nodes(self):
        visited_nodes = set()
        source_node = [u for u, d in self.nodes.items() if d.type == "input"]
        if not source_node:
            return
        source_node = source_node[0]
        visited_nodes.add(source_node)
        dfs_stack = [source_node]
        adj_list = self.adjacency_list()
        while dfs_stack:
            u = dfs_stack.pop()
            for v in adj_list[u]:
                if v not in visited_nodes:
                    visited_nodes.add(v)
                    dfs_stack.append(v)
        self.nodes = {k: d for k, d in self.nodes.items() if k in visited_nodes}
        self.edges = {(u, v) for u, v in self.edges if u in visited_nodes and v in visited_nodes}

    def to_json(self):
        return {
            "nodes": [u.to_json() for u in self.nodes.values()],
            "edges": [(u, v) for u, v in self.edges],
            "key2chain": self.key2chain
        }

    def _reformat_node_name(self, name):
        max_prefix = ""
        for d in self.default_display_treeset:
            if name.startswith(d):
                if len(d) > len(max_prefix):
                    max_prefix = d
        assert max_prefix, name
        # takes one more step
        next_slash = name[len(max_prefix) + 1:].find("/")
        if next_slash != -1:
            return name[:len(max_prefix) + 1 + next_slash], True
        return name, False

    @staticmethod
    def _escape_label(name):
        return json.dumps(name)

    @staticmethod
    def _extract_module_name(name):
        path = []
        for x in name.split("/"):
            m = re.match(r"(.*)\[(.*)\]", x)
            if m is not None:
                path.append(m.group(2))
        return ".".join(path)

    @staticmethod
    def _extract_scope_type(name):
        path = []
        for x in name.split("/"):
            m = re.match(r"(.*)\[(.*)\]", x)
            if m is not None:
                path.append(m.group(1))
            else:
                path.append(x)
        return path

    def _retrieve_chains(self, graph, node):
        # node is the ListConstruct node of mutable
        _adj_list = defaultdict(list)
        _rev_adj_list = defaultdict(list)
        src_scope = node.scope

        def _debug_name_to_id(debug_name):
            return debug_name.split("/")[-1]

        for node_id, node_data in graph.nodes_io.items():
            for input_name in node_data.inputs:
                input_name = _debug_name_to_id(input_name)
                if input_name in graph.nodes_io:
                    _adj_list[input_name].append(node_id)
                    _rev_adj_list[node_id].append(input_name)

        def _get_connected_edges(src_node, reverse=False):
            if reverse:
                adj_list = _rev_adj_list
            else:
                adj_list = _adj_list
            visited_edges = set()
            dfs_stack = [src_node]
            while dfs_stack:
                u = dfs_stack.pop()
                if not graph.nodes_io[u].scope or not graph.nodes_io[u].scope.startswith(src_scope):
                    continue
                for v in adj_list[u]:
                    if v not in visited_edges:
                        if reverse:
                            visited_edges.add((v, u))
                        else:
                            visited_edges.add((u, v))
                        dfs_stack.append(v)
            return visited_edges

        chains = [set() for _ in range(len(node.inputs))]
        for chain_id, input_name in enumerate(node.inputs):
            chain_list = {(_debug_name_to_id(input_name), _debug_name_to_id(node.debugName))}
            chain_list |= _get_connected_edges(_debug_name_to_id(input_name), reverse=True)
            chain_list |= _get_connected_edges(_debug_name_to_id(node.debugName))
            chain_list = {(self.oriid2visid[u], self.oriid2visid[v]) for u, v in chain_list
                          if u in self.oriid2visid and v in self.oriid2visid and
                          (self.oriid2visid[u], self.oriid2visid[v]) in self.edges}
            chains[chain_id] |= chain_list
        return chains


def get_vis_graph(model, inputs, mutator):
    module_name2key = {mutable.name: mutable.key for mutable in mutator.mutables.traverse(deduplicate=False)}
    g = _graph(model, inputs, verbose=False)
    vis_graph = VisGraph(g, module_name2key)
    vis_graph_json = vis_graph.to_json()
    return vis_graph_json
