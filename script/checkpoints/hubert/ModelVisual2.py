import graphviz
import logging
import torch
import torchvision
from nni.common.graph_utils import TorchModuleGraph

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class ModelVisual:
    def __init__(self, model, dummy_input):
        """
        visualize the target model. Note, the dummy_input
        should be the same side with model.
        """
        self.model = model
        self.dummy_input = dummy_input
        self.visited = set()
        self.graph = None
        self.unique_id = 0
        self.mapping = {}


    def traverse(self, modulegraph, curnode, lastnode, graph):
        _logger.info('Visiting %s , from %s', curnode, lastnode)
        if curnode in self.visited:
            if lastnode is not None:
                graph.edge(self.mapping[lastnode], self.mapping[curnode])
            return
        self.visited.add(curnode)
        self.mapping[curnode] = str(self.unique_id)
        self.unique_id += 1
        render_cfg = {'shape': 'ellipse', 'style': 'solid'}
        nodestring = modulegraph.name_to_node[curnode].name + \
            '\n'+modulegraph.name_to_node[curnode].op_type
        graph.node(self.mapping[curnode], nodestring, **render_cfg)
        if lastnode is not None:
            graph.edge(self.mapping[lastnode], self.mapping[curnode])
        nexts = modulegraph.find_successors(curnode)

        for _next in nexts:
            self.traverse(modulegraph, _next, curnode, graph)



    def visualize(self, filepath, unpack=False):
        self.unique_id = 0
        self.mapping.clear()
        self.visited.clear()
        mg = TorchModuleGraph(self.model, self.dummy_input)
        # unpack the tensor tuple/list if needed
        if unpack:
            mg.unpack_manually()
        self.graph = graphviz.Digraph(format='jpg')
        for name, nodeio in mg.nodes_py.nodes_io.items():
            if nodeio.input_or_output == 'input':
                # find the input of the whole graph
                nodes = mg.input_to_node[name]
                for node in nodes:
                    self.traverse(mg, node.name, None, self.graph)
        self.graph.render(filepath)




