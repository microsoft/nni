# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import logging


from .graph_from_trace import VisualGraph


CONV_TYPE = 'aten::_convolution'
ADD_TYPES = ['aten::add', 'aten::add_']
CAT_TYPE = 'aten::cat'
logger = logging.getLogger('Shape_Dependency')


class ChannelDependency:
    def __init__(self, model=None, data=None, graph=None):
        """
        This model analyze the channel dependencis between the conv
        layers in a model.

        Parameters
        ----------
            model:
                The model to be analyzed.
            data:
                The example input data to trace the network architecture.
            graph:
                if we alreay has the traced graph of the target model, we donnot
                need to trace the model again.
        """
        self.graph_builder = VisualGraph(model, data, graph)
        self.cnodes = list(self.graph_builder.graph.nodes())
        self.graph = self.graph_builder.graph
        self.forward_edge = self.graph_builder.forward_edge
        self.c2py = self.graph_builder.c2py
        self.dependency = {}
        self.build_channel_dependency()


    def get_parent_convs(self, node):
        """
        Find the nearest father conv layers for the target node.

        Parameters
        ---------
            node:
                target node.

        Returns
        -------
            parent_convs:
                nearest father conv layers for the target worknode.
        """
        parent_convs = []
        queue = []
        queue.append(node)
        while queue:
            curnode = queue.pop(0)
            if curnode in self.c2py and self.c2py[curnode].isOp \
                    and curnode.kind() == CONV_TYPE:
                # find the first met conv
                parent_convs.append(curnode)
                continue
            parents = self.c2py[curnode].parents()
            for parent in parents:
                if parent in self.c2py and (self.c2py[parent].isOp or 'Tensor' in str(parent.type())):
                    # filter the scalar parameters of the functions
                    # only consider the Tensors/ List(Tensor)
                    queue.append(parent)
        return parent_convs

    def build_channel_dependency(self):
        """
            Build the channel dependency for the conv layers
            in the model.
        """
        for node in self.cnodes:
            parent_convs = []
            if node.kind() in ADD_TYPES:
                parent_convs = self.get_parent_convs(node)
            if node.kind() == CAT_TYPE:
                cat_dim = list(node.inputs())[1].toIValue()
                # N * C * H * W
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
            ratios:
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

        for node in self.cnodes:
            if node.kind() == CONV_TYPE and self.c2py[node].name in ratios:
                if node not in self.dependency:
                    # this layer has no dependency on other layers
                    # it's legal to set any prune ratio between 0 and 1
                    continue
                for other in self.dependency[node]:
                    if self.c2py[other].name not in ratios:
                        return False
                    elif ratios[self.c2py[node].name] != ratios[self.c2py[other].name]:
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
            for node in self.cnodes:
                if node.kind() != CONV_TYPE or node in visited:
                    continue
                setid += 1
                row = ['Set %d' % setid]
                if node not in self.dependency:
                    visited.add(node)
                    row.append(self.c2py[node].name)
                else:
                    for other in self.dependency[node]:
                        visited.add(other)
                        row.append(self.c2py[other].name)
                csv_w.writerow(row)

    @property
    def dependency_sets(self):
        """
        Get the list of the dependency set.

        Returns
        -------
            dependency_sets:
                list of the dependency sets. For example,
                [set(['conv1', 'conv2']), set(['conv3', 'conv4'])]

        """
        d_sets = []
        visited = set()
        for node in self.cnodes:
            if node.kind() != CONV_TYPE or node in visited:
                continue
            tmp_set = set()
            if node not in self.dependency:
                visited.add(node)
                tmp_set.add(self.c2py[node].name)
            else:
                for other in self.dependency[node]:
                    visited.add(other)
                    tmp_set.add(self.c2py[other].name)
            d_sets.append(tmp_set)
        return d_sets
