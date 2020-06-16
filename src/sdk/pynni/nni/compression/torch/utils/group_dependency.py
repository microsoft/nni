# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import logging

from nni._graph_utils import TorchModuleGraph


CONV_TYPE = 'aten::_convolution'
ADD_TYPES = ['aten::add', 'aten::add_']
CAT_TYPE = 'aten::cat'
logger = logging.getLogger('Group_Dependency')


class GroupDependency:
    def __init__(self, model=None, dummy_input=None, traced_model=None):
        """
        This model analyze the group dependencis between the conv
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
        self.build_group_dependency()

    def _get_parent_convs(self, node):
        """
        Find the nearest father conv layers for the target node.

        Parameters
        ---------
        node : torch._C.Node
            target node.

        Returns
        -------
        parent_layers : list
            nearest father conv/linear layers for the target worknode.
        """
        parent_layers = []
        queue = []
        queue.append(node)
        while queue:
            curnode = queue.pop(0)
            if curnode.op_type == 'Conv2d':
                # find the first met conv
                parent_layers.append(curnode.name)
                continue
            parents = self.graph.find_predecessors(curnode.unique_name)
            parents = [self.graph.name_to_node[name] for name in parents]
            for parent in parents:
                queue.append(parent)
        return parent_layers

    def _get_conv_groups(self, node_group):
        """
        Get the number of groups for a convolutional layer.
        
        Parameters
        ----------
        node_group : NodePyGroup
            target node.
        
        Returns
        -------
        group : int
            the number of the groups of the target conv layer.
        """
        cpp_conv = list(filter(lambda x: x.kind()==CONV_TYPE, node_group.node_cpps))
        assert len(cpp_conv) == 1
        cpp_conv = cpp_conv[0]
        inputs = list(cpp_conv.inputs())
        # get the number of the group from the input parameters
        group = inputs[8].toIValue()
        return group
        
    def build_group_dependency(self):
        """
        Build the channel dependency for the conv layers
        in the model.

        Returns
        -------
        self.dependency : dict
            key: the name of conv layers, value: the minimum value that the number of
            filters should be divisible to.
        """
        for node in self.graph.nodes_py.nodes_op:
            if node.op_type == 'Conv2d':
                group = self._get_conv_groups(node)
                if node.name in self.dependency:
                    # the conv layer whose group is larger than 1 will require that
                    # it's number of output channel to be divisible by the number of group.
                    self.dependency[node.name] = max(self.dependency[node.name], group)
                else:
                    self.dependency[node.name] = group
                if group > 1:
                    # for the conv layer whose group is larger than 1, it will require the number
                    # of output channels of their parent conv layer to be divisible by group.
                    parent_convs = self._get_parent_convs(node)
                    for parent in parent_convs:
                        if parent in self.dependency:
                            self.dependency[parent] = max(self.dependency[parent], group)
                        else:
                            self.dependency[parent] = group
        return self.dependency

    def export(self, filepath):
        """
        export the group dependency to a csv file.
        output example:
        Conv layer, Shoule be divisible by
        """
        