# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import logging
import torch
import numpy as np

from .attr import get_nested_attr


__all__ = ['ChannelDependency', 'GroupDependency', 'ReshapeDependency',
           'InputChannelDependency', 'AttentionWeightDependency']


CONV_TYPE = 'aten::_convolution'
ADD_MUL_LOGICAL_TYPES = [
    'aten::add', 'aten::add_', 'aten::sub', 'aten::sub_', 'aten::subtract', 'aten::subtract_',
    'aten::mul', 'aten::mul_', 'aten::div', 'aten::div_', 'aten::multiply', 'aten::multiply_', 'aten::divide', 'aten::divide_',
    'aten::addcmul', 'aten::addcmul_',
    'aten::addcdiv', 'aten::addcdiv_',
    'aten::logical_xor', 'aten::logical_xor_',
    'aten::logical_and', 'aten::logical_and_',
    'aten::logical_or', 'aten::logical_or_',
]
CAT_TYPE = 'aten::cat'
logger = logging.getLogger('Shape_Dependency')
RESHAPE_OPS = [CAT_TYPE, 'aten::view',
               'aten::reshape', 'aten::flatten', 'aten::mean', 'aten::expand_as', 'aten::pixel_shuffle']


def lcm_list(L):
    lcm = 1
    for i in L:
        lcm = np.lcm(lcm, i)
    return lcm


def gcd_list(L):
    gcd = L[0]
    for i in L:
        gcd = np.gcd(gcd, i)
    return gcd


class Dependency:
    def __init__(self, model=None, dummy_input=None, traced_model=None):
        """
        Build the graph for the model.
        """
        from nni.common.graph_utils import TorchModuleGraph

        # check if the input is legal
        if traced_model is None:
            # user should provide model & dummy_input to trace
            # the model or a already traced model
            assert model is not None and dummy_input is not None
        self.graph: TorchModuleGraph = TorchModuleGraph(model, dummy_input, traced_model)
        self.model = model
        self.dependency = dict()
        self.build_dependency()

    def build_dependency(self):
        raise NotImplementedError

    def export(self, filepath):
        raise NotImplementedError


def reshape_break_channel_dependency(op_node):
    """
    The reshape operations such as (reshape, view, flatten) may break
    the channel dependency. We need to check the input parameters of
    these reshape operations to check if this reshape node will break
    the channel dependency. However, it's complicated to analyze the the input
    parameters for each reshape function and infer if it will break the channel
    dependency. So currently, we just check if the input channel and the output
    channel is the same, if so, then we can say the original reshape function
    doesn't want to change the number of the channels, which means the channel
    dependency is not broken. In contrast, the original reshap operation wants
    to change the number of channels, so it breaks the channel dependency.

    Parameters
    ----------
    opnode: NodePyOP
        A Op node of the graph.
    Returns
    -------
    bool
        If this operation will break the channel dependency.
    """
    in_shape = op_node.auxiliary['in_shape']
    out_shape = op_node.auxiliary['out_shape']
    # FIXME: e.g., in_shape will be None if the input comes from a buffer, should be fixed in next release
    if not in_shape or not out_shape:
        return True
    if len(in_shape) <= 1 or len(out_shape) <= 1:
        return True
    in_channel = in_shape[1]
    out_channel = out_shape[1]
    return in_channel != out_channel


class ChannelDependency(Dependency):
    """
    This model analyze the channel dependencies between the conv
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
    prune_type: str
        This parameter indicates the channel pruning type: 1) `Filter`
        prune the filter of the convolution layer to prune the corresponding
        channels 2) `Batchnorm`: prune the channel in the batchnorm layer
    """

    def __init__(self, model, dummy_input, traced_model=None, prune_type='Filter'):
        self.prune_type = prune_type
        self.target_types = []
        if self.prune_type == 'Filter':
            self.target_types.extend(['Conv2d', 'Linear', 'ConvTranspose2d', 'Embedding'])
        elif self.prune_type == 'Batchnorm':
            self.target_types.append('BatchNorm2d')

        from typing import Dict, Set
        self.dependency: Dict[str, Set[str]]

        super(ChannelDependency, self).__init__(
            model, dummy_input, traced_model)

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
        visited_set = set()
        while queue:
            curnode = queue.pop(0)
            if curnode in visited_set:
                continue
            visited_set.add(curnode)
            if curnode.op_type in self.target_types:
                # find the first met conv
                parent_layers.append(curnode.name)
                continue
            elif curnode.op_type in RESHAPE_OPS:
                if reshape_break_channel_dependency(curnode):
                    continue
            parents = self.graph.find_predecessors(curnode.unique_name)
            parents = [self.graph.name_to_node[name] for name in parents]
            for parent in parents:
                if parent in visited_set:
                    continue
                queue.append(parent)

        return parent_layers

    def build_dependency(self):
        """
        Build the channel dependency for the conv layers
        in the model.
        """
        # unpack the tuple/list manually before analyze the
        # channel dependency
        self.graph.unpack_manually()
        for node in self.graph.nodes_py.nodes_op:
            parent_layers = []
            # find the node that contains aten::add
            # or aten::cat operations
            if node.op_type in ADD_MUL_LOGICAL_TYPES:
                # refer issue 4540 for more details. Multiplication actually
                # will not introduce the channel dependency, cause the misaligned
                # channels can propagate to each other. However, when one of the input
                # tensor is from skip connection(residual), the channel propagation
                # may be failed(the input is also used by another layer and cannot be
                # pruned), in this case, we need to fix the conflict maunally.
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

            # additional denpendency for (group number == output channel number) depth-wise conv
            if node.op_type in ['Conv2d', 'ConvTranspose2d', "GroupNorm"]:
                if node.op_type in ['Conv2d', 'ConvTranspose2d']:
                    condition = self._conv_condition(node)
                elif node.op_type == "GroupNorm":
                    condition = self._group_norm_condition(node)
                if condition:
                    parent_layers.append(node.name)
                    parent_layers.extend(self._depthwise_get_parent(node))

            dependency_set = set(parent_layers)
            # merge the dependencies
            for parent in parent_layers:
                if parent in self.dependency:
                    dependency_set.update(self.dependency[parent])
            # save the dependencies
            for _node in dependency_set:
                self.dependency[_node] = dependency_set

    def _conv_condition(self, node_group):
        node_name = node_group.name
        leaf_module = get_nested_attr(self.model, node_name)
        assert isinstance(
            leaf_module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))
        group = leaf_module.groups
        n_filter = leaf_module.out_channels
        return n_filter == group and group != 1

    def _group_norm_condition(self, node_group) -> int:
        node_name = node_group.name
        leaf_module = get_nested_attr(self.model, node_name)
        assert isinstance(leaf_module, (torch.nn.GroupNorm))
        return leaf_module.num_groups == leaf_module.num_channels

    def _depthwise_get_parent(self, node):
        parent_layers = []
        # the input node is a Conv node
        predeessors = self.graph.find_predecessors(node.unique_name)
        predeessors = [self.graph.name_to_node[x] for x in predeessors]
        queue = predeessors
        while queue:
            curnode = queue.pop(0)
            if curnode.op_type == 'Conv2d' or curnode.op_type == 'ConvTranspose2d' or curnode.op_type == 'Linear':
                # find the first met conv
                parent_layers.append(curnode.name)
                continue
            parents = self.graph.find_predecessors(curnode.unique_name)
            parents = [self.graph.name_to_node[name] for name in parents]
            for parent in parents:
                queue.append(parent)
        return parent_layers

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
        header = ['Dependency Set', 'Layers']
        setid = 0
        visited = set()
        with open(filepath, 'w') as csvf:
            csv_w = csv.writer(csvf, delimiter=',')
            csv_w.writerow(header)
            for node in self.graph.nodes_py.nodes_op:
                if node.op_type not in self.target_types or node in visited:
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
            if node.op_type not in self.target_types or node in visited:
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


class InputChannelDependency(ChannelDependency):
    """
    Some pruners may prune the input channel of the convolutional
    layers. While pruning the input channel of the convolutional layers,
    the layers that share the same input tensor should prune the same
    channels, and we say these layers that share the same input tensor/channel
    has the input channel dependency. If we only prune the input channel of one
    layer in the dependency set, there will be a shape conflict for the other
    layers in the same dependency set, which may trigger a runtime error.
    Here we judge whether the application will truncate the dependency by analyzing
    whether the number of channels before and after the operation has changed.
    If not, the input channel dependency will be passed to the following nodes.
    """

    def __init__(self, model, dummy_input, traced_model=None):
        """
        This model analyze the input channel dependencies between the conv
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
        super(InputChannelDependency, self).__init__(
            model, dummy_input, traced_model)

    def _get_following_convs(self, tensor):
        queue = []
        key_layers = []
        queue.extend(self.graph.input_to_node[tensor])
        while queue:
            curnode = queue.pop(0)
            if curnode.op_type == 'Conv2d' or curnode.op_type == 'Linear' or curnode.op_type == 'ConvTranspose2d':
                # find the first met conv
                key_layers.append(curnode.name)
                continue
            elif curnode.op_type in RESHAPE_OPS:
                # check if the reshape operation will break the channel dependency
                if reshape_break_channel_dependency(curnode):
                    # reshape operations also breaks the dependency relationship
                    continue
            successors = self.graph.find_successors(curnode.unique_name)
            successors = [self.graph.name_to_node[name] for name in successors]
            for layer in successors:
                queue.append(layer)
        return key_layers

    def build_dependency(self):
        """
        Build the input channel dependencies.
        The `InputChannelDependency` indicates the layers that have
        dependencies when pruning the input channel of the conv layers.
        In contrast, `ChannelDependency` indicates the dependent layers
        when pruning the output channles of conv layers (for example, L1FilterPruner).
        """
        # unpack the tuple or list manually
        self.graph.unpack_manually()
        for tensor in self.graph.input_to_node:
            # start from this tensor, find all the conv layers that
            # take this tensor as input. Similar to the `ChannelDependency`
            # the conv layer will truncate the dependencies
            layers = self._get_following_convs(tensor)
            dependency_set = set(layers)
            for layer in layers:
                if layer in self.dependency:
                    dependency_set.update(self.dependency[layer])
            for layer in dependency_set:
                self.dependency[layer] = dependency_set


class GroupDependency(Dependency):
    """
    This model analyze the group dependencis between the conv
    layers in a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be analyzed.
    dummy_input : torch.Tensor
        The example input data to trace the network architecture.
    traced_model : torch._C.Graph
        if we alreay has the traced graph of the target model, we donnot
        need to trace the model again.
    """

    def __init__(self, model, dummy_input, traced_model=None):
        self.min_groups = {}
        super(GroupDependency, self).__init__(model, dummy_input, traced_model)

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
            nearest father conv layers for the target node. Due to the group
            dependency only exists between the conv layers, so we only find
            the parent conv layers.
        """
        parent_layers = []
        # the input node is a Conv node
        predeessors = self.graph.find_predecessors(node.unique_name)
        predeessors = [self.graph.name_to_node[x] for x in predeessors]
        queue = predeessors
        while queue:
            curnode = queue.pop(0)
            if curnode.op_type == 'Conv2d' or curnode.op_type == 'ConvTranspose2d' or curnode.op_type == 'Linear':
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
        node_name = node_group.name
        leaf_module = get_nested_attr(self.model, node_name)
        assert isinstance(
            leaf_module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))
        group = leaf_module.groups
        n_filter = leaf_module.out_channels
        if n_filter == group:
            # depthwise conv will not introduce extra group dependency
            return 1
        return group

    def _get_group_norm_condition(self, node_group) -> int:
        """
        Get the number of groups for a group norm layer.

        Parameters
        ----------
        node_group : NodePyGroup
            target node.
        Returns
        -------
        condition: int
            the number that layer's num channel
            require to be divisible to
        """
        node_name = node_group.name
        leaf_module = get_nested_attr(self.model, node_name)
        assert isinstance(leaf_module, (torch.nn.GroupNorm))

        return leaf_module.num_groups


    def build_dependency(self):
        """
        Build the channel dependency for the conv layers
        in the model. This function return the group number
        of each conv layers. Note that, here, the group count
        of conv layers may be larger than their originl groups.
        This is because that the input channel will also be grouped
        for the group conv layers. To make this clear, assume we
        have two group conv layers: conv1(group=2), conv2(group=4).
        conv2 takes the output features of conv1 as input.
        Then we have to the filters of conv1 can still be
        divided into 4 groups after filter pruning, because
        the input channels of conv2 should be divided into
        4 groups.

        Returns
        -------
        self.dependency : dict
            key: the name of conv layers, value: the minimum value that the number of
            filters should be divisible to.
        """
        self.groups = {}
        for node in self.graph.nodes_py.nodes_op:
            if node.op_type in ['Conv2d', 'ConvTranspose2d', "GroupNorm"]:
                if node.op_type in ['Conv2d', 'ConvTranspose2d']:
                    group = self._get_conv_groups(node)
                elif node.op_type == "GroupNorm":
                    group = self._get_group_norm_condition(node)
                if node.name in self.groups:
                    # the conv layer whose group is larger than 1 will require that
                    # it's number of output channel to be divisible by the number of group.
                    self.groups[node.name].append(group)
                else:
                    self.groups[node.name] = [group]
                if group > 1:
                    # for the conv layer whose group is larger than 1, it will require the number
                    # of output channels of their parent conv layer to be divisible by group.
                    parent_convs = self._get_parent_convs(node)
                    for parent in parent_convs:
                        if parent in self.groups:
                            self.groups[parent].append(group)
                        else:
                            self.groups[parent] = [group]

        for name in self.groups:
            self.dependency[name] = lcm_list(self.groups[name])
            if min(self.groups[name]) == gcd_list(self.groups[name]):
                self.min_groups[name] = min(self.groups[name])
            else:
                self.min_groups[name] = 1

        return self.dependency

    def export(self, filepath):
        """
        export the group dependency to a csv file.
        Each line describes a convolution layer, the
        first part of each line is the Pytorch module
        name of the conv layer. The second part of each
        line is the group count of the filters in this layer.
        Note that, the group count may be larger than this
        layers original group number.
        output example:
        Conv layer, Groups
        Conv1, 1
        Conv2, 2
        Conv3, 4
        """
        header = ['Conv Layer Name', 'Group']
        with open(filepath, 'w') as csvf:
            csv_w = csv.writer(csvf, delimiter=',')
            csv_w.writerow(header)
            for name in self.dependency:
                group = self.dependency[name]
                csv_w.writerow([name, group])

    @property
    def dependency_sets(self):
        return self.dependency


class ReshapeDependency(Dependency):
    def __init__(self, model=None, dummy_input=None, traced_model=None):
        """
        Some model may have the view/reshape functions, such functions may have fixed parameters
        and cannot be replaced at all. Therefore, these functions may have some constraints on
        their input shapes. In this class, we find the direct input conv/linear layers of these
        reshape functions. If you get the shape conflict when run the forward inference on the
        speeduped model, please try remove these layers from the pruner config list and try again.

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
        super(ReshapeDependency, self).__init__(
            model, dummy_input, traced_model)

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
            if curnode.op_type == 'Conv2d' or curnode.op_type == 'Linear' or curnode.op_type == 'ConvTranspose2d':
                # find the first met conv
                parent_layers.append(curnode.name)
                continue
            parents = self.graph.find_predecessors(curnode.unique_name)
            parents = [self.graph.name_to_node[name] for name in parents]
            for parent in parents:
                queue.append(parent)
        return parent_layers

    def build_dependency(self):
        """
        Build the channel dependency for the conv layers
        in the model.
        """
        # unpack the tuple/list manually before analyze the
        # channel dependency
        self.graph.unpack_manually()
        for node in self.graph.nodes_py.nodes_op:
            parent_layers = []
            # find the node that contains aten::add
            # or aten::cat operations
            if node.op_type in ['aten::view', 'aten::reshape']:
                logger.info('Detect reshape-like functions: %s', node.op_type)
                parent_layers = self._get_parent_layers(node)
                print('Parent layers', parent_layers)
                self.dependency[node.unique_name] = parent_layers

    def export(self, filepath):
        """
        export the reshape dependencies as a csv file.

        Output example:
        Reshape OP, Dependent Layers
        model.view.1,layer1.1.conv2,layer1.0.conv2,conv1
        model.mean.1,layer1.0.conv1
        model.reshape.1,layer1.1.conv1
        """
        header = ['Reshape OP', 'Dependent Layers']
        with open(filepath, 'w') as csvf:
            csv_w = csv.writer(csvf, delimiter=',')
            csv_w.writerow(header)
            for reshape_op in self.dependency:
                row = [reshape_op].extend(self.dependency[reshape_op])
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
        for reshape_node in self.dependency:
            d_sets.extend(self.dependency[reshape_node])
        d_sets = list(set(d_sets))
        return d_sets


class AttentionWeightDependency(Dependency):
    def __init__(self, model=None, dummy_input=None, traced_model=None):
        """
        Groups the linear layers belonging to the same attention layer in a model.
        Currently, we only capture weights in attention layers with forward computations written
        as four Linear layers (projections for Q, K, V, and output) and two matmul operations.
        The method implemented here can work for Huggingface transformers but may not correctly
        capture transformers written in other fashions (e.g., torch.nn.Transformer).

        Parameters
        ----------
        model : torch.nn.Module
            The model to be analyzed.
        dummy_input : torch.Tensor
            The example input data to trace the network architecture.
        traced_model : torch._C.Graph
            if we already have the traced graph of the target model, we do not
            need to trace the model again.
        """
        super(AttentionWeightDependency, self).__init__(
            model, dummy_input, traced_model)

    def _get_parent_layers(self, node):
        """
        Find the nearest parent linear layers for the target node.

        Parameters
        ---------
        node : torch._C.Node
            target node.

        Returns
        -------
        parent_layers: list
            nearest parent linear layers for the target worknode.
        """
        parent_layers = []
        queue = []
        queue.append(node)
        while queue:
            curnode = queue.pop(0)
            if curnode.op_type == 'Linear':
                if curnode.name not in parent_layers:
                    parent_layers.append(curnode.name)
                continue
            if curnode.op_type == 'LayerNorm':
                continue
            parents = self.graph.find_predecessors(curnode.unique_name)
            parents = [self.graph.name_to_node[name] for name in parents]
            for parent in parents:
                queue.append(parent)
        return parent_layers

    def _get_children_layers(self, node):
        """
        Find the nearest children linear layers for the target node.

        Parameters
        ---------
        node : torch._C.Node
            target node.

        Returns
        -------
        children_layers: list
            nearest children linear layers for the target worknode.
        """
        children_layers = []
        queue = []
        queue.append(node)
        while queue:
            curnode = queue.pop(0)
            if curnode.op_type == 'Linear':
                if curnode.name not in children_layers:
                    children_layers.append(curnode.name)
                continue
            if curnode.op_type == 'LayerNorm':
                continue
            children = self.graph.find_successors(curnode.unique_name)
            children = [self.graph.name_to_node[name] for name in children]
            for child in children:
                queue.append(child)
        return children_layers

    def build_dependency(self):
        """
        For every matmul operation, find the immediate parent and children Linear operations.
        If we get three parents and one children, add these four weights as a dependecy group.
        """
        self.graph.unpack_manually()
        for node in self.graph.nodes_py.nodes_op:
            layers = []
            if node.op_type == 'aten::matmul':
                parent_layers = self._get_parent_layers(node)
                children_layers = self._get_children_layers(node)
                if len(parent_layers) == 3 and len(children_layers) == 1:
                    layers.extend(parent_layers)
                    layers.extend(children_layers)

            self.dependency[node.name] = layers

    @property
    def dependency_sets(self):
        """
        Get the list of the dependency set.

        Returns
        -------
        dependency_sets : list
            list of the dependency sets.
            Each dependency set is a 4-element list of module names, with the first three elements being the projection
            matrices for Q, K, V (in any order), and the last element being the dense matrix.
        """
        d_sets = []
        for node in self.graph.nodes_py.nodes_op:
            if node.op_type != 'aten::matmul' or node.name not in self.dependency or len(self.dependency[node.name]) != 4:
                continue
            d_sets.append(self.dependency[node.name])

        return d_sets

    def export(self, filepath):
        """
        Export the group dependency to a csv file. Each line describes an attention layer.

        Output example:
        Attention layer matmul op, Group
        """
        header = ['Attention layer matmul op', 'Group']
        with open(filepath, 'w') as csvf:
            csv_w = csv.writer(csvf, delimiter=',')
            csv_w.writerow(header)
            for name in self.dependency:
                group = self.dependency[name]
                if len(group) > 0:
                    csv_w.writerow([name, group])
