import re
import torch
import logging
import torch.nn as nn
import torch.jit as jit
from torch.utils.tensorboard._pytorch_graph import CLASSTYPE_KIND, GETATTR_KIND

__all__ = ["PyNode", "Graph_Builder"]

TUPLE_UNPACK = 'prim::TupleUnpack'

logger = logging.getLogger('Graph_From_Trace')

class PyNode:
    def __init__(self, cnode, isValue=False):
        self.cnode = cnode
        self.isValue = isValue
        self.isTensor = False
        self.isOp = not self.isValue
        if self.isValue:
            if isinstance(self.cnode.type(), torch._C.TensorType):
                self.isTensor = True
                self.shape = self.cnode.type().sizes()
        if self.isOp:
            self.name = cnode.scopeName()
            # remove the __module prefix
            if self.name.startswith('__module.'):
                self.name = self.name[len('__module.'):]

    def __str__(self):
        if self.isTensor:
            name = 'Tensor: {}'.format(self.shape)
        elif self.isOp:
            name = self.cnode.kind()
            name = re.split('::', name)[1]
            scope = self.cnode.scopeName()
            scope = re.split('/', scope)
            if len(scope) > 0:
                name = scope[-1] + '\nType: ' + name
        else:
            name = str(self.cnode.type())
        return name

    def parents(self):
        if self.isOp:
            return list(self.cnode.inputs())
        else:
            return [self.cnode.node()]


class Graph_Builder:
    def __init__(self, model, data):
        """
            input:
                model: model to build the network architecture
                data:  input data for the model
            We build the network architecture  graph according the graph
            in the scriptmodule.
        """
        self.model = model
        self.data = data
        self.traced_model = jit.trace(model, data)
        self.forward_edge = {}
        self.graph = self.traced_model.graph
        # Start from pytorch 1.4.0, we need this function to get more
        # detail information
        torch._C._jit_pass_inline(self.graph)
        self.c2py = {}
        self.visited = set()
        self.build_graph()
        self.unpack_tuple()

    def unpack_tuple(self):
        """
        jit.trace also traces the tuple creation and unpack, which makes 
        the grapgh complex and difficult to understand. Therefore, we 
        """
        for node in self.graph.nodes():
            if node.kind() == TUPLE_UNPACK:
                in_tuple = list(node.inputs())[0]
                parent_node = in_tuple.node()
                in_tensors = list(parent_node.inputs())
                out_tensors = list(node.outputs())
                assert len(in_tensors) == len(out_tensors)
                for i in range(len(in_tensors)):
                    ori_edges = self.forward_edge[in_tensors[i]]
                    # remove the out edge to the Tuple_construct OP node
                    self.forward_edge[in_tensors[i]] = list(
                        filter(lambda x: x != parent_node, ori_edges))
                    # Directly connect to the output nodes of the out_tensors
                    self.forward_edge[in_tensors[i]].extend(
                        self.forward_edge[out_tensors[i]])

    def build_graph(self):
        for node in self.graph.nodes():
            self.c2py[node] = PyNode(node)
            for input in node.inputs():
                if input not in self.c2py:
                    self.c2py[input] = PyNode(input, True)
                if input in self.forward_edge:
                    self.forward_edge[input].append(node)
                else:
                    self.forward_edge[input] = [node]
            for output in node.outputs():
                if output not in self.c2py:
                    self.c2py[output] = PyNode(output, True)
                if node in self.forward_edge:
                    self.forward_edge[node].append(output)
                else:
                    self.forward_edge[node] = [output]

    def visual_traverse(self, curnode, graph, lastnode):
        """"
        Input:
            curnode: current visiting node(tensor/module)
            graph: the handle of the Dgraph
            lastnode: the last visited node
        """
        if curnode in self.visited:
            if lastnode is not None:
                graph.edge(str(id(lastnode)), str(id(curnode)))
            return
        self.visited.add(curnode)
        name = str(self.c2py[curnode])
        if self.c2py[curnode].isOp:
            graph.node(str(id(curnode)), name, shape='ellipse', color='orange')
        else:
            graph.node(str(id(curnode)), name, shape='box', color='lightblue')
        if lastnode is not None:
            graph.edge(str(id(lastnode)), str(id(curnode)))
        if curnode in self.forward_edge:
            for _next in self.forward_edge[curnode]:
                self.visual_traverse(_next, graph, curnode)

    def visualization(self, filename, format='jpg'):
        """
        visualize the network architecture automaticlly.
        Input:
            filename: the filename of the saved image file
            format: the output format
            debug: if enable the debug mode
        """
        import graphviz
        graph = graphviz.Digraph(format=format)
        self.visited.clear()
        for input in self.graph.inputs():
            if input.type().kind() == CLASSTYPE_KIND:
                continue
            self.visual_traverse(input, graph, None)
        graph.render(filename)
