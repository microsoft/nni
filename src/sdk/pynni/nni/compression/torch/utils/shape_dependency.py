# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import logging

from nni._graph_utils import TorchModuleGraph


CONV_TYPE = 'aten::_convolution'
ADD_TYPES = ['aten::add', 'aten::add_']
CAT_TYPE = 'aten::cat'
logger = logging.getLogger('Shape_Dependency')


class ChannelDependency:
    def __init__(self, model=None, dummy_input=None, traced_model=None):
        """
        This model analyze the channel dependencis between the conv
        layers in a model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be analyzed.
        data : torch.Tensor
            The example input data to trace the network architecture.
        traced_model : torch._C.Graph
            if we alreay has the traced graph of the target model, we donnot
            need to trace the model again.
        """
        # check if the input is legal
        if traced_model is None:
            # user should provide model & dummy_input to trace the model or a already traced model
            assert model is not None and dummy_input is not None
        self.graph = TorchModuleGraph(model, dummy_input, traced_model)
        self.dependency = dict()
        self.build_channel_dependency()

    def _get_parent_layers(self, node):
        """
        Find the nearest father conv layers for the target node.

        Parameters
        ---------
        node : torch._C.Node
            target node.

        Returns
        -------
        parent_layers: list
            nearest father conv/linear layers for the target worknode.
        """
        parent_layers = []
        queue = []
        queue.append(node)
        while queue:
            curnode = queue.pop(0)
            if curnode.op_type == 'Conv2d' or curnode.op_type == 'Linear':
                # find the first met conv
                parent_layers.append(curnode.name)
                continue
            parents = self.graph.find_predecessors(curnode.unique_name)
            parents = [self.graph.name_to_node[name] for name in parents]
            for parent in parents:
                queue.append(parent)
        return parent_layers

    def build_channel_dependency(self):
        """
        Build the channel dependency for the conv layers
        in the model.
        """
        for node in self.graph.nodes_py.nodes_op:
            parent_layers = []
            # find the node that contains aten::add
            # or aten::cat operations
            if node.op_type in ADD_TYPES:
                parent_layers = self._get_parent_layers(node)
            elif node.op_type == CAT_TYPE:
                # To determine if this cat operation will introduce channel
                # dependency, we need the specific input parameters of the cat
                # opertion. To get the input parameters of the cat opertion, we
                # need to traverse all the cpp_nodes included by this NodePyGroup,
                # because, TorchModuleGraph merges the important nodes and the adjacent
                # unimportant nodes (nodes started with prim::attr, for example) into a
                # NodepyGroup.
                cat_dim = None
                for cnode in node.node_cpps:
                    if cnode.kind() == CAT_TYPE:
                        cat_dim = list(cnode.inputs())[1].toIValue()
                        break
                if cat_dim != 1:
                    parent_layers = self._get_parent_layers(node)
            dependency_set = set(parent_layers)
            # merge the dependencies
            for parent in parent_layers:
                if parent in self.dependency:
                    dependency_set.update(self.dependency[parent])
            # save the dependencies
            for _node in dependency_set:
                self.dependency[_node] = dependency_set


    def export(self, filepath):
        """
        export the channel dependencies as a csv file.
        The layers at the same line have output channel
        dependencies with each other. For example,
        layer1.1.conv2, conv1, and layer1.0.conv2 have
        output channel dependencies with each other, which
        means the output channel(filters) numbers of these
        three layers should be same with each other, otherwise
        the model may has shape conflict.

        Output example:
        Dependency Set,Convolutional Layers
        Set 1,layer1.1.conv2,layer1.0.conv2,conv1
        Set 2,layer1.0.conv1
        Set 3,layer1.1.conv1
        """
        header = ['Dependency Set', 'Convolutional Layers']
        setid = 0
        visited = set()
        with open(filepath, 'w') as csvf:
            csv_w = csv.writer(csvf, delimiter=',')
            csv_w.writerow(header)
            for node in self.graph.nodes_py.nodes_op:
                if node.op_type != 'Conv2d' or node in visited:
                    continue
                setid += 1
                row = ['Set %d' % setid]
                if node.name not in self.dependency:
                    visited.add(node)
                    row.append(node.name)
                else:
                    for other in self.dependency[node.name]:
                        visited.add(self.graph.name_to_node[other])
                        row.append(other)
                csv_w.writerow(row)

    @property
    def dependency_sets(self):
        """
        Get the list of the dependency set.

        Returns
        -------
        dependency_sets : list
            list of the dependency sets. For example,
            [set(['conv1', 'conv2']), set(['conv3', 'conv4'])]

        """
        d_sets = []
        visited = set()
        for node in self.graph.nodes_py.nodes_op:
            if node.op_type != 'Conv2d' or node in visited:
                continue
            tmp_set = set()
            if node.name not in self.dependency:
                visited.add(node)
                tmp_set.add(node.name)
            else:
                for other in self.dependency[node.name]:
                    visited.add(self.graph.name_to_node[other])
                    tmp_set.add(other)
            d_sets.append(tmp_set)
        return d_sets
