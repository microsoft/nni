# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import torch
import logging

import nni._graph_utils as _graph_utils
from _graph_utils import TorchModuleGraph

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
        self.build_channel_dependency()
        self.dependency = {}


    def get_parent_convs(self, node):
        """
        Find the nearest father conv layers for the target node.

        Parameters
        ---------
        node : torch._C.Node
            target node.

        Returns
        -------
        parent_convs: list
            nearest father conv layers for the target worknode.
        """
        parent_convs = []
        queue = []
        queue.append(node)
        while queue:
            curnode = queue.pop(0)
            if node.op_type == 'Conv2d':
                # find the first met conv
                parent_convs.append(curnode.name)
                continue
            parents = self.graph.find_predecessors(curnode.unique_name)
            parents = [self.graph.name_to_node[name] for name in parents]
            for parent in parents:
                queue.append(parent)
        return parent_convs

    def build_channel_dependency(self):
        """
        Build the channel dependency for the conv layers
        in the model.
        """
        for node in self.graph.nodes_py.nodes_op:
            parent_convs = []
            # find the node that contains aten::add
            # or aten::cat operations
            if node.op_type in ADD_TYPES:
                parent_convs = self.get_parent_convs(node)
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
                    if cnode.kind == CAT_TYPE:
                        cat_dim = list(cnode.inputs())[1].toIValue()
                        break
                if cat_dim != 1:
                    parent_convs = self.get_parent_convs(node)
            dependency_set = set(parent_convs)
            # merge the dependencies
            for parent in parent_convs:
                if parent in self.dependency:
                    dependency_set.update(self.dependency[parent])
            # save the dependencies
            for _node in dependency_set:
                self.dependency[_node] = dependency_set


    def filter_prune_check(self, ratios):
        """
        According to the channel dependencies between the conv
        layers, check if the filter pruning ratio for the conv
        layers is legal.

        Parameters
        ---------
        ratios : dict
            the prune ratios for the layers. %ratios is a dict,
            in which the keys are the names of the target layer
            and the values are the prune ratio for the corresponding
            layers. For example:
            ratios = {'body.conv1': 0.5, 'body.conv2':0.5}
            Note: the name of the layers should looks like
            the names that model.named_modules() functions
            returns.

        Returns
        -------
        True/False
        """
        for node in self.graph.nodes_py.nodes_op:
            if node.op_type == 'Conv2d' and node.name in ratios:
                if node.name not in self.dependency:
                    # this layer has no dependency on other layers
                    # it's legal to set any prune ratio between 0 and 1
                    continue
                for other in self.dependency[node.name]:
                    if other not in ratios:
                        return False
                    elif ratios[other] != ratios[node.name]:
                        return False
        return True


    def export(self, filepath):
        """
        export the channel dependencies as a csv file.
        """
        header = ['Dependency Set', 'Convolutional Layers']
        setid = 0
        visited = set()
        with open(filepath, 'w') as csvf:
            csv_w = csv.writer(csvf, delimiter=',')
            csv_w.writerow(header)
            for node in self.graph.nodes_py.nodes_op:
                if node.op_type() != 'Conv2d' or node in visited:
                    continue
                setid += 1
                row = ['Set %d' % setid]
                if node.name not in self.dependency:
                    visited.add(node)
                    row.append(node.name)
                else:
                    for other in self.dependency[node]:
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
