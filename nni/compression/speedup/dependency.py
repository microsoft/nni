# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import operator
from copy import deepcopy
from typing import (Any, Dict, List, Set)
from uuid import uuid4

import numpy as np
import torch
import torch.fx
from torch.fx.passes.shape_prop import TensorMetadata

from nni.compression.base.config import trans_legacy_config_list, select_modules_by_config

__all__ = [
    'build_channel_dependency',
    'build_group_dependency',
    'build_reshape_dependency',
    'build_weight_sharing_dependency',
    'auto_set_denpendency_group_ids',
]

# see https://pytorch.org/docs/stable/torch.html#pointwise-ops
CALL_FUNCTION_REDUCE = [
    torch.add, torch.sub, torch.subtract, torch.mul, torch.div, torch.multiply, torch.divide,
    torch.addcmul, torch.addcdiv, torch.logical_xor, torch.logical_and, torch.logical_or,
    operator.add, operator.iadd, operator.sub, operator.isub, operator.mul, operator.imul,
    operator.truediv, operator.floordiv, operator.or_, operator.and_, operator.xor,
]
CALL_METHOD_REDUCE = [
    'add', 'add_', 'sub', 'sub_', 'subtract', 'subtract_', 'mul', 'mul_',
    'div', 'div_', 'multiply', 'multiply_', 'divide', 'divide_',
    'addcmul', 'addcmul_', 'addcdiv', 'addcdiv_', 'logical_xor', 'logical_xor_',
    'logical_and', 'logical_and_', 'logical_or', 'logical_or_',
]

# see https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops
CALL_FUNCTION_CONCAT = [torch.cat, torch.concat, torch.concatenate, torch.column_stack, torch.dstack,
                     torch.hstack, torch.row_stack, torch.stack]
CALL_FUNCTION_RESHAPE = [
    *CALL_FUNCTION_CONCAT,
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


def channel_dependency_breakpoint(node: torch.fx.Node):
    """
    Check if this operation will break the channel dependency.

    Parameters
    ----------
    node : torch.fx.Node

    Returns
    -------
    bool
        If this operation will break the channel dependency.
    """
    if len(node.all_input_nodes) == 0:
        return True

    def find_parent(node: torch.fx.Node):
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

    in_shape = get_shape(find_parent(node))
    out_shape = get_shape(node)
    if not in_shape or not out_shape or len(in_shape) <= 1 or len(out_shape) <= 1:
        return True
    in_channel = in_shape[1]
    out_channel = out_shape[1]
    # TODO: add more rules here
    return in_channel != out_channel


def find_adjacent_layers(node: torch.fx.Node,
                         module: torch.fx.GraphModule,
                         target_types: List[torch.nn.Module],
                         direction: str = 'parent') -> List[torch.fx.Node]:
    """
    Find the nearest layers of `target_types` in module for the target node given search direction.

    Parameters
    ---------
    node : torch.fx.Node
        The target node.
    module : torch.fx.GraphModule
        The model to be analyzed.
    target_types : list
        The target types of the father layers.
    direction : str
        The search direction, 'parent' or 'child'.

    Returns
    -------
    parent_layers: list
        The nearest father conv/linear layers for the target worknode.
    """

    layers = []
    if direction == 'parent':
        nodes = node.all_input_nodes
    elif direction == 'child':
        nodes = node.users
    else:
        raise ValueError("search direction should be 'parent' or 'child'")

    for node in nodes:
        if node.op == 'call_module':
            target_module = module.get_submodule(node.target)
            if isinstance(target_module, tuple(target_types)):
                layers.append(node)
                continue
        elif channel_dependency_breakpoint(node):
            continue
        layers.extend(find_adjacent_layers(node, module, target_types, direction))
    return layers


def auto_set_denpendency_group_ids(graph_module: torch.fx.GraphModule,
                                   config_list: List[Dict[str, Any]],
                                   prune_type: str = 'Filter',
                                   prune_axis: int = 0) -> List[Dict[str, Any]]:
    """
    Auto find the output dependency between all 'Conv2d', 'Linear', 'ConvTranspose2d', 'Embedding' modules,
    then set the ``dependency_group_id`` in config list.

    Note that a new dependency group id will be set as a shortcut in one config,
    it will replace the old configured one in that config.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The traced model to be analyzed.
    config_list : list
        The compression config list.

    Returns
    config_list : list
        The new config list with dependency group id.
    """

    channel_dependency = build_channel_dependency(graph_module, prune_type, prune_axis)
    module2uid = {}
    for d_set in channel_dependency:
        uid = uuid4().hex
        module2uid.update({node.target: uid for node in d_set})

    group_dependency = build_group_dependency(graph_module)
    group_dependency = {node.target: group_max for node, (group_max, group_min) in group_dependency.items()}

    config_list = trans_legacy_config_list(config_list)
    new_config_list = []
    for config in config_list:
        modules, public_config, _ = select_modules_by_config(graph_module, config)
        for target in modules.keys():
            sub_config = deepcopy(public_config)
            if target in module2uid:
                sub_config['dependency_group_id'] = module2uid[target]
            if target in group_dependency:
                sub_config['internal_metric_block'] = int(group_dependency[target])
            new_config_list.append({
                'op_names': [target],
                **sub_config
            })

    return new_config_list


def convert_dependency_to_set(dependency: Dict[Any, Set[Any]]) -> List[Set[Any]]:
    """
    Convert the dependency dict to sets of dependent nodes.

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
    d_sets = []
    for key, value in dependency.items():
        if key not in visited:
            tmp = set([key]) | value
            visited.update(tmp)
            if len(tmp) > 1:
                d_sets.append(tmp)
    return d_sets


def build_weight_sharing_dependency(graph_module: torch.fx.GraphModule) -> List[List[str]]:
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


def build_channel_dependency(graph_module: torch.fx.GraphModule,
                             prune_type: str = 'Filter',
                             prune_axis: int = 0) -> List[Set[torch.fx.Node]]:
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
    prune_axis : int
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
        if prune_axis == 1:
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

            d_set = set(find_adjacent_layers(node, graph_module, target_types, 'child'))

        # output channel dependency
        else:
            # Output channel dependency indicates the dependent layers when pruning the
            # output channles of conv layers (for example, L1FilterPruner).

            if node.op == 'call_module':
                submodule = graph_module.get_submodule(node.target)
                # additional denpendency for (group number == output channel number) depth-wise conv:
                if (isinstance(submodule, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) and submodule.groups == submodule.out_channels) \
                    or (isinstance(submodule, torch.nn.GroupNorm) and submodule.num_groups == submodule.num_channels):
                        d_set = set([node] + find_adjacent_layers(node, graph_module, target_types, 'parent'))

            elif node.op == 'call_function':
                if node.target in CALL_FUNCTION_REDUCE:
                    # refer issue 4540 for more details. Multiplication actually
                    # will not introduce the channel dependency, cause the misaligned
                    # channels can propagate to each other. However, when one of the input
                    # tensor is from skip connection(residual), the channel propagation
                    # may be failed(the input is also used by another layer and cannot be
                    # pruned), in this case, we need to fix the conflict maunally.
                    d_set = set(find_adjacent_layers(node, graph_module, target_types, 'parent'))

                elif node.target in CALL_FUNCTION_CONCAT:
                    # To determine if this cat operation will introduce channel
                    # dependency, we need the specific input parameters of the cat
                    # operation.
                    cat_dim = node.kwargs.get('dim', None) if node.kwargs.get('dim', None) is not None else node.args[1]
                    if cat_dim != 1:
                        d_set = set(find_adjacent_layers(node, graph_module, target_types, 'parent'))

            elif node.op == 'call_method':
                if node.target in CALL_METHOD_REDUCE:
                    d_set = set(find_adjacent_layers(node, graph_module, target_types, 'parent'))

        # merge dependencies
        for parent in tuple(d_set):
            if parent in dependency:
                d_set |= dependency[parent]

        for parent in d_set:
            dependency[parent] = d_set

    return convert_dependency_to_set(dependency)


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
    dependency : Dict
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
                d_set = find_adjacent_layers(node, graph_module, [torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear], 'parent')
                for parent in d_set:
                    groups[parent] = groups.get(parent, []) + [num_groups]

    for node in groups:
        group_max = lcm_list(groups[node])
        group_min = gcd_list(groups[node]) if min(groups[node]) == gcd_list(groups[node]) else 1

        if group_max == 1 and group_min == 1:
            continue

        dependency[node] = (group_max, group_min)
    return dependency


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
        parent_layers = []
        # find the node that contains the reshape-like function
        if node.op == 'call_function' and node.target in [torch.narrow, torch.reshape]:
            logger.info(f'Detect reshape-like functions: {node.target}, args = {node.args}, kwargs = {node.kwargs}')
            parent_layers = find_adjacent_layers(node, graph_module, [torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear], 'parent')
            dependency[node] = parent_layers
        elif node.op == 'call_method' and node.target in ['narrow', 'reshape', 'view']:
            logger.info(f'Detect reshape-like methods: {node.target}, args = {node.args}, kwargs = {node.kwargs}')
            parent_layers = find_adjacent_layers(node, graph_module, [torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear], 'parent')
            dependency[node] = parent_layers
    return dependency
