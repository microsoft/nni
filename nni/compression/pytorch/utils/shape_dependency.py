# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import logging

__all__ = ['ChannelDependency', 'GroupDependency',
           'CatPaddingDependency', 'InputChannelDependency']

CONV_TYPE = 'aten::_convolution'
ADD_TYPES = ['aten::add', 'aten::add_']
CAT_TYPE = 'aten::cat'
logger = logging.getLogger('Shape_Dependency')
RESHAPE_OPS = [CAT_TYPE, 'aten::view',
               'aten::reshape', 'aten::flatten', 'aten::mean']


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
        self.graph = TorchModuleGraph(model, dummy_input, traced_model)
        self.dependency = dict()
        self.build_dependency()

    def build_dependency(self):
        raise NotImplementedError

    def export(self, filepath):
        raise NotImplementedError


class ChannelDependency(Dependency):
    def __init__(self, model=None, dummy_input=None, traced_model=None):
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
        """
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
        header = ['Dependency Set', 'Layers']
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
    in_channel = in_shape[1]
    out_channel = out_shape[1]
    return in_channel != out_channel


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

    def __init__(self, model, dummy_input=None, traced_model=None):
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


class CatPaddingDependency(ChannelDependency):
    def __init__(self, model=None, dummy_input=None, traced_model=None):
        super(CatPaddingDependency, self).__init__(
            model, dummy_input, traced_model)

    def build_dependency(self):
        """
        Build the cat padding dependencies.
        If the output features of several layers are stitched together
        by cat operation, then these layers have cat padding dependencies.
        This is because when inferring the cat mask, we need all the input
        masks for the cat operation. At this time we need to know the source
        of all input vectors of a cat operation.
        """
        for node in self.graph.nodes_py.nodes_op:
            parent_layers = []
            if node.op_type == CAT_TYPE:
                parent_layers = self._get_parent_layers(node)
                dependency_set = set(parent_layers)
                # merge the dependencies
                for parent in parent_layers:
                    if parent in self.dependency:
                        dependency_set.update(self.dependency[parent])
                # save the dependencies
                for _node in dependency_set:
                    self.dependency[_node] = dependency_set

    @property
    def dependency_sets(self):
        d_sets = []
        visited = set()
        for nodename in self.dependency:
            if nodename in visited:
                continue
            d_sets.append(self.dependency[nodename])
        return d_sets

    def export(self, filepath):
        """
        Export the dependencies into a file.
        In the output file, each line contains a set of layers
        whose output features are stitched together by the cat
        operation.

        output example:
        Dependency Set, Layers
        set1, Conv1, Conv2
        set2, Conv3, Conv4
        """
        header = ['Dependency Set', 'Layers']
        setid = 0
        with open(filepath, 'w') as csvf:
            csv_w = csv.writer(csvf, delimiter=',')
            csv_w.writerow(header)
            for layers in self.dependency_sets:
                setid += 1
                row = ['Set %d' % setid]
                row.extend(list(layers))
                csv_w.writerow(row)


class GroupDependency(Dependency):
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
            if curnode.op_type == 'Conv2d' or curnode.op_type == 'ConvTranspose2d':
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
        cpp_conv = list(filter(lambda x: x.kind() ==
                               CONV_TYPE, node_group.node_cpps))
        assert len(cpp_conv) == 1
        cpp_conv = cpp_conv[0]
        inputs = list(cpp_conv.inputs())
        # get the number of the group from the input parameters
        group = inputs[8].toIValue()
        return group

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
        the input channels of conv2 shoule be divided into
        4 groups.

        Returns
        -------
        self.dependency : dict
            key: the name of conv layers, value: the minimum value that the number of
            filters should be divisible to.
        """
        for node in self.graph.nodes_py.nodes_op:
            if node.op_type == 'Conv2d' or node.op_type == 'ConvTranspose2d':
                group = self._get_conv_groups(node)

                if node.name in self.dependency:
                    # the conv layer whose group is larger than 1 will require that
                    # it's number of output channel to be divisible by the number of group.
                    self.dependency[node.name] = max(
                        self.dependency[node.name], group)
                else:
                    self.dependency[node.name] = group
                if group > 1:
                    # for the conv layer whose group is larger than 1, it will require the number
                    # of output channels of their parent conv layer to be divisible by group.
                    parent_convs = self._get_parent_convs(node)
                    for parent in parent_convs:
                        if parent in self.dependency:
                            self.dependency[parent] = max(
                                self.dependency[parent], group)
                        else:
                            self.dependency[parent] = group
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
