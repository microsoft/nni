# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import operator
from typing import (Dict, List)

import numpy as np
import torch
import torch.fx
from torch.fx.passes.shape_prop import TensorMetadata

__all__ = ['build_channel_dependency', 'build_group_dependency']

# see https://pytorch.org/docs/stable/torch.html#pointwise-ops
CALL_FUNCTION_ADD_MUL = [
    torch.add, torch.sub, torch.subtract, torch.mul, torch.div, torch.multiply, torch.divide,
    torch.addcmul, torch.addcdiv, torch.logical_xor, torch.logical_and, torch.logical_or,
    operator.add, operator.iadd, operator.sub, operator.isub, operator.mul, operator.imul,
    operator.truediv, operator.floordiv, operator.or_, operator.and_, operator.xor,
]
CALL_METHOD_ADD_MUL = [
    'add', 'add_', 'sub', 'sub_', 'subtract', 'subtract_', 'mul', 'mul_', 
    'div', 'div_', 'multiply', 'multiply_', 'divide', 'divide_',
    'addcmul', 'addcmul_', 'addcdiv', 'addcdiv_', 'logical_xor', 'logical_xor_',
    'logical_and', 'logical_and_', 'logical_or', 'logical_or_',
]

# see https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops
CALL_FUNCTION_CAT = [torch.cat, torch.concat, torch.concatenate, torch.column_stack, torch.dstack,
                     torch.hstack, torch.row_stack, torch.stack]
CALL_FUNCTION_RESHAPE = [
    *CALL_FUNCTION_CAT,
    torch.chunk, torch.diagonal, torch.flatten, torch.flip, torch.fliplr, torch.flipud, torch.moveaxis, torch.movedim, torch.narrow,
    torch.reshape, torch.split, torch.split_with_sizes, torch.squeeze, torch.unsqueeze, torch.transpose, torch.t, torch.permute,
    torch.repeat_interleave,
]
CALL_METHOD_RESHAPE = [
    'chunk', 'diagonal', 'flatten', 'flip', 'fliplr', 'flipud', 'moveaxis', 'movedim', 'narrow',
    'reshape', 'split', 'split_with_sizes', 'squeeze', 'unsqueeze', 'transpose', 't', 'permute',
    'view', 'view_as', 'expand', 'expand_as', 'expand_dims', 'repeat', 'repeat_interleave',
]
logger = logging.getLogger('Shape Dependency')


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


def get_parent_layers(node: torch.fx.Node, module: torch.fx.GraphModule, target_types: List[torch.nn.Module]):
    """
    Find the nearest father layers of `target_types` in module for the target node.

    Parameters
    ---------
    node : torch.fx.Node
        The target node.
    module : torch.fx.GraphModule
        The model to be analyzed.
    target_types : list
        The target types of the father layers.

    Returns
    -------
    parent_layers: list
        The nearest father conv/linear layers for the target worknode.
    """

    parent_layers = []
    for parent in node.all_input_nodes:
        if parent.op == 'call_module':
            target_module = module.get_submodule(parent.target)
            if isinstance(target_module, tuple(target_types)):
                parent_layers.append(parent)
                continue
        elif reshape_break_channel_dependency(parent):
            continue
        parent_layers.extend(get_parent_layers(parent, module, target_types))
    return parent_layers


def get_child_layers(node: torch.fx.Node, module: torch.fx.GraphModule, target_types: List[torch.nn.Module]):
    """
    Find the nearest child layers of `target_types` in module for the target node.

    Parameters
    ---------
    node : torch.fx.Node
        The target node.
    module : torch.fx.GraphModule
        The model to be analyzed.
    target_types : list
        The target types of the child layers.

    Returns
    -------
    child_layers: list
        The nearest child conv/linear layers for the target worknode.
    """

    child_layers = []
    for child in node.users:
        if child.op == 'call_module':
            target_module = module.get_submodule(child.target)
            if isinstance(target_module, tuple(target_types)):
                child_layers.append(child)
                continue
        elif reshape_break_channel_dependency(child):
            continue
        child_layers.extend(get_child_layers(child, module, target_types))
    return child_layers


def reshape_break_channel_dependency(node: torch.fx.Node):
    """
    The reshape operations such as (reshape, view, flatten) may break
    the channel dependency. We need to check the input parameters of
    these reshape operations to check if this reshape node will break
    the channel dependency. However, it's complicated to analyze the the input
    parameters for each reshape function and infer if it will break the channel
    dependency. So currently, we just check if the input channel and the output
    channel is the same, if so, then we can say the original reshape function
    doesn't want to change the number of the channels, which means the channel
    dependency is not broken. In contrast, the original reshape operation wants
    to change the number of channels, so it breaks the channel dependency.

    Parameters
    ----------
    node : torch.fx.Node

    Returns
    -------
    bool
        If this operation will break the channel dependency.
    """

    if node.target not in CALL_FUNCTION_RESHAPE and node.target not in CALL_METHOD_RESHAPE:
        return False

    def get_parent(node: torch.fx.Node):
        par_arg = node.all_input_nodes[0]

        while isinstance(par_arg, tuple):
            par_arg = par_arg[0]
        return par_arg

    def get_shape(node):
        meta = node.meta.get('tensor_meta', None)
        if isinstance(meta, TensorMetadata):
            return meta[0]
        elif isinstance(meta, tuple):
            return meta[0][0]
        return None

    in_shape = get_shape(get_parent(node))
    out_shape = get_shape(node)
    if not in_shape or not out_shape or len(in_shape) <= 1 or len(out_shape) <= 1:
        return True
    in_channel = in_shape[1]
    out_channel = out_shape[1]
    return in_channel != out_channel

def dependency_to_list(dependency: Dict):
    """
    Convert the dependency dict to a list.

    Parameters
    ----------
    dependency : Dict
        The individual dependency mapping for the target graph.

    Returns
    -------
    List
        A list of dependencies for the target graph.
    """
    visited = set()
    dependency_list = []
    for key, value in dependency.items():
        if key not in visited:
            tmp = set([key]) | value
            visited.update(tmp)
            if len(tmp) > 1:
                dependency_list.append(tmp)
    return dependency_list


def build_weight_sharing_dependency(graph_module: torch.fx.GraphModule):
    """
    This model analyze the weight sharing dependencies between the conv
    layers in a model. (e.g. different node refer to same module)

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The target graph and module.
    
    Returns
    -------
    dependency : List
        The weight sharing dependency for the target graph.
    """

    dependency = {}
    target_types = [torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.Embedding]
    for node in graph_module.graph.nodes:
        if node.op == 'call_module':
            target_module = graph_module.get_submodule(node.target)
            if isinstance(target_module, tuple(target_types)):
                dependency[node.target] = dependency.get(node.target, []) + [node]
    return [dep for dep in dependency.values() if len(dep) > 1]


def build_channel_dependency(graph_module: torch.fx.GraphModule, prune_type='Filter', prune_axis=1):
    """
    This model analyze the channel dependencies between the conv
    layers in a model.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The target graph and module.
    prune_type : str
        The channel pruning type. `Filter` prune the filter of the conv
        layer to prune the corresponding channels. `BatchNorm` prune the
        batchnorm layer to prune the corresponding channels.
    channel_axis : int
        The pruned axis of the conv layer. 1 for output channel, 0 for input channel.
    
    Returns
    -------
    dependency : List
        The channel dependency for the target graph.
    """

    if prune_type not in ['Filter', 'BatchNorm']:
        raise ValueError('Unsupported prune type: {}'.format(prune_type))

    if prune_axis not in [0, 1]:
        raise ValueError('Unsupported prune axis: {}. Support 0 for input channel and 1 for output channel.'.format(prune_axis))

    graph = graph_module.graph
    target_types = [torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d, torch.nn.Embedding]
    if prune_type == 'BatchNorm':
        target_types.append(torch.nn.BatchNorm2d)

    dependency = dict()
    for node in graph.nodes:
        d_set = set()
        node: torch.fx.Node

        # input channel dependency
        if prune_axis == 0:
            # Some pruners may prune the input channel of the convolutional
            # layers. While pruning the input channel of the convolutional layers,
            # the layers that share the same input tensor should prune the same
            # channels, and we say these layers that share the same input tensor/channel
            # has the input channel dependency. If we only prune the input channel of one
            # layer in the dependency set, there will be a shape conflict for the other
            # layers in the same dependency set, which may trigger a runtime error.
            # Here we judge whether the application will truncate the dependency by analyzing
            # whether the number of channels before and after the operation has changed.
            # If not, the input channel dependency will be passed to the following nodes.

            d_set = set(get_child_layers(node, graph_module, [torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d]))

        # output channel dependency
        else:
            # Output channel dependency indicates the dependent layers when pruning the
            # output channles of conv layers (for example, L1FilterPruner).

            if node.op == 'call_module':
                submodule = graph_module.get_submodule(node.target)
                # additional denpendency for (group number == output channel number) depth-wise conv:
                if (isinstance(submodule, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) and submodule.groups == submodule.out_channels) \
                    or (isinstance(submodule, torch.nn.GroupNorm) and submodule.num_groups == submodule.num_channels):
                        d_set = set([node] + get_parent_layers(node, graph_module, [torch.nn.Conv2d, torch.nn.Linear,
                                                                                    torch.nn.ConvTranspose2d]))

            elif node.op == 'call_function':
                if node.target in CALL_FUNCTION_ADD_MUL:
                        # refer issue 4540 for more details. Multiplication actually
                        # will not introduce the channel dependency, cause the misaligned
                        # channels can propagate to each other. However, when one of the input
                        # tensor is from skip connection(residual), the channel propagation
                        # may be failed(the input is also used by another layer and cannot be
                        # pruned), in this case, we need to fix the conflict maunally.
                        d_set = set(get_parent_layers(node, graph_module, target_types))

                elif node.target in CALL_FUNCTION_CAT:
                    # To determine if this cat operation will introduce channel
                    # dependency, we need the specific input parameters of the cat
                    # operation.
                    cat_dim = node.kwargs.get('dim', None)
                    if cat_dim != 1:
                        d_set = set(get_parent_layers(node, graph_module, target_types))

            elif node.op == 'call_method':
                if node.target in CALL_METHOD_ADD_MUL:
                    d_set = set(get_parent_layers(node, graph_module, target_types))

        # merge dependencies
        for parent in tuple(d_set):
            if parent in dependency:
                d_set |= dependency[parent]

        for parent in d_set:
            dependency[parent] = d_set

    return dependency_to_list(dependency)


def build_group_dependency(graph_module: torch.fx.GraphModule):
    """
    This model analyze the group dependencis between the conv
    layers in a model.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The target graph and module.

    Returns
    -------
    dependency : List
        The group dependency for the target graph.
    """
    graph = graph_module.graph
    dependency = dict()
    groups = dict()

    for node in graph.nodes:
        node: torch.fx.Node
        d_set = set()

        # Build the channel dependency for the conv layers in the model.
        # This function return the group number of each conv layers. Note
        # that, here, the group count of conv layers may be larger than
        # their originl groups. This is because that the input channel
        # will also be grouped for the group conv layers. To make this
        # clear, assume we have two group conv layers: conv1(group=2),
        # conv2(group=4). conv2 takes the output features of conv1 as input.
        # Then we have to the filters of conv1 can still be divided into
        # 4 groups after filter pruning, because the input channels of
        # conv2 should be divided into 4 groups.

        if node.op == 'call_module':
            num_groups = 0
            submodule = graph_module.get_submodule(node.target)
            if isinstance(submodule, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                # depthwise conv will not introduce extra group dependency
                num_groups = submodule.groups if submodule.groups != submodule.out_channels else 1
            elif isinstance(submodule, torch.nn.GroupNorm):
                num_groups = submodule.num_groups
            else:
                continue

            groups[node] = groups.get(node, []) + [num_groups]

            if num_groups > 1:
                # for the conv layer whose group is larger than 1, it will require the number
                # of output channels of their parent conv layer to be divisible by group.
                d_set = get_parent_layers(node, graph_module, [torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear])
                for parent in d_set:
                    groups[parent] = groups.get(parent, []) + [num_groups]

    for node in groups:
        dependency[node] = lcm_list(groups[node])
        if min(groups[node]) == gcd_list(groups[node]):
            groups[node] = min(groups[node])
        else:
            groups[node] = 1
    return dependency, groups


def build_reshape_dependency(graph_module: torch.fx.GraphModule):
    """
    Some model may have the view/reshape functions, such functions may have fixed parameters
    and cannot be replaced at all. Therefore, these functions may have some constraints on
    their input shapes. In this class, we find the direct input conv/linear layers of these
    reshape functions. If you get the shape conflict when run the forward inference on the
    speeduped model, please try remove these layers from the pruner config list and try again.
    """

    graph = graph_module.graph
    dependency = dict()

    for node in graph.nodes:
        pass