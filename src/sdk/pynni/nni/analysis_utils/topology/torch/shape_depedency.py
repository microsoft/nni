import torch
import queue
import logging
import torch.nn as nn

from .graph_from_trace import *


CONV_TYPE = 'aten::_convolution'

logger = logging.getLogger('Shape_Depedency')


class Prune_Check:
    def __init__(self, model, data):
        """
            Input: 
                model: model to be pruned
                data: example input data to trace the network architecture
        """
        self.graph_builder = Graph_Builder(model, data)
        self.cnodes = list(self.graph_builder.graph.nodes())
        self.graph = self.graph_builder.graph
        self.forward_edge = self.graph_builder.forward_edge
        self.c2py = self.graph_builder.c2py
        self.depedency = {}
        self.build_channel_depedency()
        self.visited = set()

    def get_parent_convs(self, node):
        parent_convs = []
        queue = []
        queue.append(node)
        while len(queue) > 0:
            curnode = queue.pop(0)
            print(curnode)
            print()
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

    def build_channel_depedency(self):
        for node in self.cnodes:
            parent_convs = []
            if 'add' in node.kind():
                parent_convs = self.get_parent_convs(node)
            if 'cat' in node.kind():
                cat_dim = list(node.inputs())[1].toIValue()
                # N * C * H * W
                if cat_dim != 1:
                    parent_convs = self.get_parent_convs(node)
            depedency_set = set(parent_convs)
            # merge the depedencies
            for node in parent_convs:
                if node in self.depedency:
                    depedency_set.update(self.depedency[node])
            # save the depedencies
            for node in depedency_set:
                self.depedency[node] = depedency_set

    def filter_prune_check(self, ratios):
        """
        input:
            ratios: the prune ratios for the layers
            ratios is the dict, in which the keys are 
            the names of the target layer and the values
            are the prune ratio for the corresponding layers
            For example:
            ratios = {'body.conv1': 0.5, 'body.conv2':0.5}
            Note: the name of the layers should looks like 
            the names that model.named_modules() functions 
            returns.
        """

        for node in self.cnodes:
            if node.kind() == CONV_TYPE and self.c2py[node].name in ratios:
                if node not in self.depedency:
                    # this layer has no depedency on other layers
                    # it's legal to set any prune ratio between 0 and 1
                    continue
                for other in self.depedency[node]:
                    if self.c2py[other].name not in ratios:
                        return False
                    elif ratios[self.c2py[node].name] != ratios[self.c2py[other].name]:
                        return False
        return True
