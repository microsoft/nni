# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
import torch
import torch.nn as nn
from .error_code import EmptyLayerError, ShapeMisMatchError, InputsNumberError, OutputTypeError, UnBalancedGroupError

_logger = logging.getLogger(__name__)

replace_module = {
    'BatchNorm2d': lambda module, masks: replace_batchnorm2d(module, masks),
    'BatchNorm1d': lambda module, masks: replace_batchnorm1d(module, masks),
    'InstanceNorm2d': lambda module, masks: replace_instancenorm2d(module, masks),
    'Conv2d': lambda module, masks: replace_conv2d(module, masks),
    'Conv1d': lambda module, masks: replace_conv1d(module, masks),
    'Linear': lambda module, masks: replace_linear(module, masks),
    'MaxPool2d': lambda module, masks: no_replace(module, masks),
    'AvgPool2d': lambda module, masks: no_replace(module, masks),
    'AdaptiveAvgPool2d': lambda module, masks: no_replace(module, masks),
    'ZeroPad2d': lambda module, masks: no_replace(module, masks),
    'ReLU': lambda module, masks: no_replace(module, masks),
    'ReLU6': lambda module, masks: no_replace(module, masks),
    'LeakyReLU': lambda module, masks: no_replace(module, masks),
    'ELU': lambda module, masks: no_replace(module, masks),
    'Hardtanh': lambda module, masks: no_replace(module, masks),
    'Hardsigmoid': lambda module, masks: no_replace(module, masks),
    'LogSigmoid': lambda module, masks: no_replace(module, masks),
    'PReLU': lambda module, masks: replace_prelu(module, masks),
    'RReLU': lambda module, masks: no_replace(module, masks),
    'SELU': lambda module, masks: no_replace(module, masks),
    'CELU': lambda module, masks: no_replace(module, masks),
    'GELU': lambda module, masks: no_replace(module, masks),
    'GELUActivation': lambda module, masks: no_replace(module, masks),
    'Sigmoid': lambda module, masks: no_replace(module, masks),
    'SiLU': lambda module, masks: no_replace(module, masks),
    'Mish': lambda module, masks: no_replace(module, masks),
    'Tanh': lambda module, masks: no_replace(module, masks),
    'Softplus': lambda module, masks: no_replace(module, masks),
    'Softshrink': lambda module, masks: no_replace(module, masks),
    'Softmax': lambda module, masks: no_replace(module, masks),
    'Tanhshrink': lambda module, masks: no_replace(module, masks),
    'Dropout': lambda module, masks: no_replace(module, masks),
    'Dropout2d': lambda module, masks: no_replace(module, masks),
    'Dropout3d': lambda module, masks: no_replace(module, masks),
    'Upsample': lambda module, masks: no_replace(module, masks),
    'LayerNorm': lambda module, masks: replace_layernorm(module, masks),
    'ConvTranspose2d': lambda module, masks: replace_convtranspose2d(module, masks),
    'Embedding': lambda module, masks: replace_embedding(module, masks),
    'PixelShuffle': lambda module, masks: replace_pixelshuffle(module, masks),
    'Flatten': lambda module, masks: no_replace(module, masks),
    'GroupNorm': lambda module, masks: replace_groupnorm(module, masks),
    'Hardswish': lambda module, masks: no_replace(module, masks),
}


def convert_to_coarse_mask(t_mask, dim):
    """
    Convert the mask tensor to the coarse-grained mask tensor.

    Parameters
    ---------
    t_mask: torch.Tensor
        The tensor only have 1s and 0s, 0 indicates this value is masked
        and 1 indicates the corresponding value is not masked.
    dim: int
        Try to reduce the mask tensor on this dimension.

    Returns
    -------
    indexes: torch.Tensor
        The indexes of the sparsity that can be structurally removed.
    remained_indexes: torch.Tensor
        The indexes of values that need to be remained.
    """
    assert isinstance(t_mask, torch.Tensor)
    shape = list(t_mask.size())
    n_dims = len(shape)
    dim_list = list(range(n_dims))
    # try to reduce the mask from the dim-th dimension
    dim = dim if dim >= 0 else n_dims + dim
    dim_list.remove(dim)

    t_merged = torch.sum(t_mask, dim_list)
    assert t_merged.size(0) == shape[dim]
    all_pruned = t_merged == 0
    need_remain = t_merged != 0
    # return the indexes of the sparsity that can be removed
    indexes = torch.nonzero(all_pruned, as_tuple=True)[0]
    remained_indexes = torch.nonzero(need_remain, as_tuple=True)[0]
    return indexes, remained_indexes


def convert_dense_shape(mask):
    """
    Get the dense shape of the tensor after removing the sparsity
    values.

    Parameters
    ----------
    mask: torch.Tensor
        The mask tensor.

    Returns
    -------
    dense_shape: tuple
        The dense shape after removing the sparsity values.
    """
    assert isinstance(mask, torch.Tensor)
    n_dim = len(mask.size())
    dense_shape = []
    for dim in range(n_dim):
        _, remained = convert_to_coarse_mask(mask, dim)
        dense_shape.append(remained.size(0))
    return tuple(dense_shape)


def no_replace(module, masks):
    """
    No need to replace
    """
    _logger.debug("no need to replace")
    return module


def replace_prelu(prelu, masks):
    """
    Parameters
    ----------
    module : torch.nn.PReLU
        The prelu module to be replace
    masks : tuple of masks
        The input/output/weight masks of the target module

    Returns
    -------
    torch.nn.PReLU
        The new prelu module
    """
    in_masks, output_mask, weight_mask = masks
    if len(in_masks) != 1:
        raise InputsNumberError()
    if not isinstance(output_mask, torch.Tensor):
        raise OutputTypeError(type(output_mask), torch.Tensor)

    in_mask = in_masks[0]
    weight_mask = weight_mask['weight']
    if weight_mask.size(0) == 1:
        return prelu
    pruned_in, remained_in = convert_to_coarse_mask(in_mask, 1)
    pruned_out, remained_out = convert_to_coarse_mask(output_mask, 1)
    n_remained_in = weight_mask.size(0) - pruned_in.size(0)
    n_remained_out = weight_mask.size(0) - pruned_out.size(0)
    remained_in, remained_out = remained_in.to(
        prelu.weight.device), remained_out.to(prelu.weight.device)
    if n_remained_in != n_remained_out:
        raise ShapeMisMatchError()

    if n_remained_in == 0:
        return torch.nn.Identity()
    new_prelu = torch.nn.PReLU(n_remained_in)
    new_prelu.weight.data = torch.index_select(
        prelu.weight.data, 0, remained_in)
    return new_prelu


def replace_linear(linear, masks):
    """
    This function will replace the original linear according to
    the infered masks. This function support the fine-grained and
    coarse-grained sparsity. In the fine-grained scenario, this function
    will remove the whole column/row that happen to be totally covered by
    the masks.

    Parameters
    ----------
    linear : torch.nn.Linear
        The linear module to be replace
    masks : Tuple of the input masks, output masks and weight masks
        Tuple of the masks, for example
        ([input_m1, input_m2], [output_m], {'weight':weight_m})

    Returns
    -------
    torch.nn.Linear
        The new linear module
    """
    in_masks, output_mask, weight_mask = masks
    assert isinstance(linear, nn.Linear)
    if len(in_masks) != 1:
        raise InputsNumberError()
    if not isinstance(output_mask, torch.Tensor):
        raise OutputTypeError(type(output_mask), torch.Tensor)

    in_mask = in_masks[0]

    weight_mask = weight_mask['weight']
    # N C K
    pruned_in, remained_in = convert_to_coarse_mask(in_mask, -1)
    pruned_out, remained_out = convert_to_coarse_mask(output_mask, -1)
    n_remained_in = weight_mask.size(1) - pruned_in.size(0)
    n_remained_out = weight_mask.size(0) - pruned_out.size(0)
    remained_in, remained_out = remained_in.to(
        linear.weight.device), remained_out.to(linear.weight.device)
    _logger.info("replace linear with new in_features: %d, out_features: %d",
                 n_remained_in, n_remained_out)
    need_bias = False
    if linear.bias is not None:
        need_bias = True
    new_linear = torch.nn.Linear(in_features=n_remained_in,
                                 out_features=n_remained_out,
                                 bias=need_bias)
    new_linear.to(linear.weight.device)
    # Copy the remained weight from the original module
    with torch.no_grad():
        tmp_weight_data = torch.index_select(
            linear.weight.data, 0, remained_out)
        new_linear.weight.data = torch.index_select(
            tmp_weight_data, 1, remained_in)

        if linear.bias is not None:
            new_linear.bias.data = torch.index_select(
                linear.bias.data, 0, remained_out)

    return new_linear


def replace_batchnorm1d(norm, masks):
    """
    Parameters
    ----------
    norm : torch.nn.BatchNorm1d
        The batchnorm module to be replace
    masks : Tuple of the input masks, output masks and weight masks
        Tuple of the masks, for example
        ([input_m1, input_m2], [output_m], {'weight':weight_m})

    Returns
    -------
    torch.nn.BatchNorm1d
        The new batchnorm module
    """
    in_masks, output_mask, _ = masks
    assert isinstance(norm, nn.BatchNorm1d)
    in_mask = in_masks[0]

    # N, C, H, W
    _, remained_in = convert_to_coarse_mask(in_mask, 1)
    _, remained_out = convert_to_coarse_mask(output_mask, 1)
    if remained_in.size(0) != remained_out.size(0):
        raise ShapeMisMatchError()

    num_features = remained_in.size(0)
    _logger.info("replace batchnorm1d with num_features: %d", num_features)
    new_norm = torch.nn.BatchNorm1d(num_features=num_features,
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)
    # assign weights
    if norm.affine:
        new_norm.weight.data = torch.index_select(norm.weight.data, 0, remained_in)
        new_norm.bias.data = torch.index_select(norm.bias.data, 0, remained_in)

    new_norm.running_mean.data = torch.index_select(
        norm.running_mean.data, 0, remained_in)
    new_norm.running_var.data = torch.index_select(
        norm.running_var.data, 0, remained_in)
    return new_norm


def replace_batchnorm2d(norm, masks):
    """
    Parameters
    ----------
    norm : torch.nn.BatchNorm2d
        The batchnorm module to be replace
    masks : Tuple of the input masks, output masks and weight masks
        Tuple of the masks, for example
        ([input_m1, input_m2], [output_m], {'weight':weight_m})

    Returns
    -------
    torch.nn.BatchNorm2d
        The new batchnorm module
    """
    in_masks, output_mask, _ = masks
    assert isinstance(norm, nn.BatchNorm2d)
    in_mask = in_masks[0]

    # N, C, H, W
    _, remained_in = convert_to_coarse_mask(in_mask, 1)
    _, remained_out = convert_to_coarse_mask(output_mask, 1)
    if remained_in.size(0) != remained_out.size(0):
        raise ShapeMisMatchError()

    num_features = remained_in.size(0)
    _logger.info("replace batchnorm2d with num_features: %d", num_features)
    new_norm = torch.nn.BatchNorm2d(num_features=num_features,
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)
    # assign weights
    if norm.affine:
        new_norm.weight.data = torch.index_select(norm.weight.data, 0, remained_in)
        new_norm.bias.data = torch.index_select(norm.bias.data, 0, remained_in)

    new_norm.running_mean.data = torch.index_select(
        norm.running_mean.data, 0, remained_in)
    new_norm.running_var.data = torch.index_select(
        norm.running_var.data, 0, remained_in)
    return new_norm


def replace_groupnorm(norm: nn.GroupNorm, masks):
    """
    Parameters
    ----------
    norm : torch.nn.GroupNorm
        The group norm module to be replace
    masks : Tuple of the input masks, output masks and weight masks
        Tuple of the masks, for example
        ([input_m1, input_m2], [output_m], {'weight':weight_m})

    Returns
    -------
    torch.nn.GroupNorm
        The new group norm module
    """
    in_masks, output_mask, _ = masks
    assert isinstance(norm, nn.GroupNorm)
    in_mask = in_masks[0]

    # N, C, H, W
    _, remained_in = convert_to_coarse_mask(in_mask, 1)
    _, remained_out = convert_to_coarse_mask(output_mask, 1)

    assert len(remained_in.size()) == 1
    if remained_in.size(0) != remained_out.size(0):
        raise ShapeMisMatchError()

    ori_channel_step = norm.num_channels // norm.num_groups
    for groupid in range(norm.num_groups):
        in_start = groupid * ori_channel_step
        in_end = in_start + ori_channel_step

        new_channel_step = torch.logical_and(
            in_start <= remained_in,
            remained_in < in_end,
        ).sum().item()

        # this group fully pruned
        if new_channel_step == 0:
            continue

        break


    new_groups = 0

    # Validate
    for groupid in range(norm.num_groups):
        in_start = groupid * ori_channel_step
        in_end = in_start + ori_channel_step
        num_item = torch.logical_and(
            in_start <= remained_in,
            remained_in < in_end,
        ).sum().item()
        if num_item == 0:
            continue

        # check if the number of remained channel of each group are the same
        if num_item != new_channel_step:
            raise UnBalancedGroupError()

        new_groups += 1

    new_num_channels = remained_in.size()[0]
    new_module = nn.GroupNorm(
        new_groups,
        new_num_channels,
        eps=norm.eps,
        affine=norm.affine,
    )
    if new_module.affine:
        new_module.weight.data = torch.index_select(
            norm.weight.data,
            0,
            remained_in,
        )
        new_module.bias.data = torch.index_select(
            norm.bias.data,
            0,
            remained_in,
        )
    return new_module


def replace_instancenorm2d(norm, masks):
    """
    Parameters
    ----------
    norm : torch.nn.InstanceNorm2d
        The instancenorm module to be replace
    masks : Tuple of the input masks, output masks and weight masks
        Tuple of the masks, for example
        ([input_m1, input_m2], [output_m], {'weight':weight_m})

    Returns
    -------
    torch.nn.InstanceNorm2d
        The new instancenorm module
    """
    in_masks, output_mask, _ = masks
    assert isinstance(norm, nn.InstanceNorm2d)
    in_mask = in_masks[0]

    # N, C, H, W
    _, remained_in = convert_to_coarse_mask(in_mask, 1)
    _, remained_out = convert_to_coarse_mask(output_mask, 1)
    if remained_in.size(0) != remained_out.size(0):
        raise ShapeMisMatchError()

    num_features = remained_in.size(0)
    _logger.info("replace instancenorm2d with num_features: %d", num_features)
    new_norm = torch.nn.InstanceNorm2d(num_features=num_features,
                                       eps=norm.eps,
                                       momentum=norm.momentum,
                                       affine=norm.affine,
                                       track_running_stats=norm.track_running_stats)
    # assign weights
    if norm.affine:
        new_norm.weight.data = torch.index_select(norm.weight.data, 0, remained_in)
        new_norm.bias.data = torch.index_select(norm.bias.data, 0, remained_in)

    if norm.track_running_stats:
        new_norm.running_mean.data = torch.index_select(
            norm.running_mean.data, 0, remained_in)
        new_norm.running_var.data = torch.index_select(
            norm.running_var.data, 0, remained_in)
    return new_norm


def replace_conv2d(conv, masks):
    """
    Replace the original conv with a new one according to the infered
    masks, the function support the fine-grained sparsity and coarse-grained
    sparsity. In the fine-grained scenario, this replace function will replace
    the filters that happen to be totally coverd by the fine-grained sparsity.

    Parameters
    ----------
    conv : torch.nn.Conv2d
        The conv2d module to be replaced
    masks : Tuple of the input masks, output masks and weight masks
        Tuple of the masks, for example
        ([input_m1, input_m2], [output_m], {'weight':weight_m})

    Returns
    -------
    torch.nn.Conv2d
        The new conv2d module
    """
    in_masks, output_mask, weight_masks = masks
    assert isinstance(conv, nn.Conv2d)
    # the conv layer should only have one input tensor
    if len(in_masks) != 1:
        raise InputsNumberError()

    in_mask = in_masks[0]

    weight_mask = weight_masks['weight']
    pruned_in, remained_in = convert_to_coarse_mask(in_mask, 1)
    pruned_out, remained_out = convert_to_coarse_mask(output_mask, 1)

    n_remained_in = weight_mask.size(1) * conv.groups - pruned_in.size(0)
    n_remained_out = weight_mask.size(0) - pruned_out.size(0)

    if n_remained_in != remained_in.size(0) or n_remained_out != remained_out.size(0):
        raise ShapeMisMatchError()

    k_size1, k_size2 = conv.kernel_size
    # Note: We should resolve the group dependency of the conv layers before
    # run into here.
    # check if the mask tensor meets the group dependency and calculate the
    # new number of the groups after pruning
    # the original step size of the input channel for each group
    ori_inchannel_step = int(conv.in_channels/conv.groups)
    # the original step size of the output channel for each group
    ori_outchannel_step = int(conv.out_channels/conv.groups)
    # calculate the new_in_channel_step and new_outchannel_step first
    new_inchannel_step = new_outchannel_step = None
    for groupid in range(conv.groups):
        in_start = groupid * ori_inchannel_step
        in_end = in_start + ori_inchannel_step
        out_start = groupid * ori_outchannel_step
        out_end = out_start + ori_outchannel_step

        new_inchannel_step: int = torch.logical_and(
            in_start <= remained_in,
            remained_in < in_end
        ).sum().item()
        new_outchannel_step: int = torch.logical_and(
            out_start <= remained_out,
            remained_out < out_end
        ).sum().item()
        # remap the global index to the group index
        if new_inchannel_step == 0:
            # if the whole group are pruned
            continue
        else:
            break
    tmp_weight = torch.ones(
        n_remained_out, new_inchannel_step, k_size1, k_size2)
    tmp_weight = tmp_weight.to(conv.weight.device)
    if new_inchannel_step == 0 or new_outchannel_step == 0:
        raise EmptyLayerError()
    if n_remained_in % new_inchannel_step != 0 or n_remained_out % new_outchannel_step != 0:
        raise UnBalancedGroupError()

    new_groups = 0
    for groupid in range(conv.groups):
        in_start = groupid * ori_inchannel_step
        in_end = in_start + ori_inchannel_step
        out_start = groupid * ori_outchannel_step
        out_end = out_start + ori_outchannel_step

        current_input_mask = torch.logical_and(in_start <= remained_in, remained_in < in_end)
        current_input_index = remained_in[current_input_mask]

        current_output_mask = torch.logical_and(out_start <= remained_out, remained_out < out_end)
        current_output_index = remained_out[current_output_mask]

        # remap the global index to the group index
        current_input_index = current_input_index - in_start
        if len(current_input_index) == 0:
            # if the whole group are pruned
            assert len(current_output_index) == 0
            continue
        # check if the number of remained channel of each group are the same
        if len(current_input_index) != new_inchannel_step or len(current_output_index) != new_outchannel_step:
            raise UnBalancedGroupError()

        # copy the weight into tmp_weight
        new_out_start = new_outchannel_step * new_groups
        new_out_end = new_out_start + new_outchannel_step
        tmp_weight[new_out_start:new_out_end] = torch.index_select(
            conv.weight[current_output_index], 1, torch.as_tensor(current_input_index, dtype=torch.long).to(conv.weight.device))
        new_groups += 1

    _logger.info("replace conv2d with in_channels: %d, out_channels: %d",
                  n_remained_in, n_remained_out)

    # need_bias is a flag that indicates that if a conv layer need
    # bias, if the original conv doesn't have a bias and there is
    # no constant need to be folded into the bias, the need_bias is False.
    need_bias = conv.bias is not None
    new_conv = torch.nn.Conv2d(in_channels=n_remained_in,
                               out_channels=n_remained_out,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=new_groups,
                               bias=need_bias,
                               padding_mode=conv.padding_mode)

    new_conv.to(conv.weight.device)
    new_conv.weight.copy_(tmp_weight)

    # copy the bias data
    if conv.bias is not None:
        new_conv.bias.data.copy_(torch.index_select(
            conv.bias.data, 0, remained_out))

    return new_conv


def replace_conv1d(conv, masks):
    """
    Replace the original conv with a new one according to the infered
    masks, the function support the fine-grained sparsity and coarse-grained
    sparsity. In the fine-grained scenario, this replace function will replace
    the filters that happen to be totally coverd by the fine-grained sparsity.

    Parameters
    ----------
    conv : torch.nn.Conv1d
        The conv1d module to be replaced
    masks : Tuple of the input masks, output masks and weight masks
        Tuple of the masks, for example
        ([input_m1, input_m2], [output_m], {'weight':weight_m})

    Returns
    -------
    torch.nn.Conv1d
        The new conv1d module
    """
    in_masks, output_mask, weight_masks = masks
    assert isinstance(conv, torch.nn.Conv1d)
    if len(in_masks) != 1:
        raise InputsNumberError()

    in_mask = in_masks[0]
    weight_mask = weight_masks['weight']
    pruned_in, remained_in = convert_to_coarse_mask(in_mask, 1)
    pruned_out, remained_out = convert_to_coarse_mask(output_mask, 1)

    n_remained_in = weight_mask.size(1) * conv.groups - pruned_in.size(0)
    n_remained_out = weight_mask.size(0) - pruned_out.size(0)

    if n_remained_in != remained_in.size(0) or n_remained_out != remained_out.size(0):
        raise ShapeMisMatchError()

    k_size = conv.kernel_size[0]
    # Note: We should resolve the group dependency of the conv layers before
    # run into here.
    # check if the mask tensor meets the group dependency and calculate the
    # new number of the groups after pruning
    # the original step size of the input channel for each group
    ori_inchannel_step = int(conv.in_channels / conv.groups)
    # the original step size of the output channel for each group
    ori_outchannel_step = int(conv.out_channels / conv.groups)
    # calculate the new_in_channel_step and new_outchannel_step first
    new_inchannel_step = new_outchannel_step = None
    for groupid in range(conv.groups):
        in_start = groupid * ori_inchannel_step
        in_end = in_start + ori_inchannel_step
        out_start = groupid * ori_outchannel_step
        out_end = out_start + ori_outchannel_step

        new_inchannel_step: int = torch.logical_and(
            in_start <= remained_in,
            remained_in < in_end
        ).sum().item()
        new_outchannel_step: int = torch.logical_and(
            out_start <= remained_out,
            remained_out < out_end
        ).sum().item()
        # remap the global index to the group index
        if new_inchannel_step == 0:
            # if the whole group are pruned
            continue
        else:
            break
    tmp_weight = torch.ones(
        n_remained_out, new_inchannel_step, k_size)
    tmp_weight = tmp_weight.to(conv.weight.device)
    if new_inchannel_step == 0 or new_outchannel_step == 0:
        raise EmptyLayerError()
    if n_remained_in % new_inchannel_step != 0 or n_remained_out % new_outchannel_step != 0:
        raise UnBalancedGroupError()

    new_groups = 0
    for groupid in range(conv.groups):
        in_start = groupid * ori_inchannel_step
        in_end = in_start + ori_inchannel_step
        out_start = groupid * ori_outchannel_step
        out_end = out_start + ori_outchannel_step

        current_input_mask = torch.logical_and(in_start <= remained_in, remained_in < in_end)
        current_input_index = remained_in[current_input_mask]

        current_output_mask = torch.logical_and(out_start <= remained_out, remained_out < out_end)
        current_output_index = remained_out[current_output_mask]

        # remap the global index to the group index
        current_input_index = current_input_index - in_start
        if len(current_input_index) == 0:
            # if the whole group are pruned
            assert len(current_output_index) == 0
            continue
        # check if the number of remained channel of each group are the same
        if len(current_input_index) != new_inchannel_step or len(current_output_index) != new_outchannel_step:
            raise UnBalancedGroupError()

        # copy the weight into tmp_weight
        new_out_start = new_outchannel_step * new_groups
        new_out_end = new_out_start + new_outchannel_step
        tmp_weight[new_out_start:new_out_end] = torch.index_select(
            conv.weight[current_output_index], 1, torch.as_tensor(current_input_index, dtype=torch.long).to(conv.weight.device))
        new_groups += 1

    _logger.info("replace conv1d with in_channels: %d, out_channels: %d",
                  n_remained_in, n_remained_out)

    # need_bias is a flag that indicates that if a conv layer need
    # bias, if the original conv doesn't have a bias and there is
    # no constant need to be folded into the bias, the need_bias is False.
    need_bias = conv.bias is not None
    new_conv = torch.nn.Conv1d(in_channels=n_remained_in,
                               out_channels=n_remained_out,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=new_groups,
                               bias=need_bias,
                               padding_mode=conv.padding_mode)

    new_conv.to(conv.weight.device)
    new_conv.weight.copy_(tmp_weight)

    # copy the bias data
    if conv.bias is not None:
        new_conv.bias.data.copy_(torch.index_select(
            conv.bias.data, 0, remained_out))

    return new_conv


def replace_convtranspose2d(convtrans, masks):
    """
    We need anothor replace function for
    convtranspose2d, because the layout of
    the weight is different from traditional
    conv layers. The layout of the weight is [N_in, N_out, ksize_1, ksize_2]
    Parameters
    ----------
    convtrans : torch.nn.ConvTranspose2d
        The conv2d module to be replaced
    masks : Tuple of the input masks, output masks and weight masks
        Tuple of the masks, for example
        ([input_m1, input_m2], [output_m], {'weight':weight_m})
    Returns
    -------
    torch.nn.ConvTranspose2d
        The new conv2d module
    """
    in_masks, output_mask, weight_masks = masks
    assert isinstance(convtrans, torch.nn.ConvTranspose2d)
    if len(in_masks) != 1:
        raise InputsNumberError()
    in_mask = in_masks[0]

    weight_mask = weight_masks['weight']
    pruned_in, remained_in = convert_to_coarse_mask(in_mask, 1)
    pruned_out, remained_out = convert_to_coarse_mask(output_mask, 1)
    # ConvTranspose2d has the weight shape of [N_in, N_out/groups, k1, k2]
    n_remained_in = weight_mask.size(0) - pruned_in.size(0)
    n_remained_out = weight_mask.size(
        1) * convtrans.groups - pruned_out.size(0)
    if n_remained_in != remained_in.size(0) or n_remained_out != remained_out.size(0):
        raise ShapeMisMatchError()

    k_size1, k_size2 = convtrans.kernel_size
    # Note: we should resolve the group dependency of the convtrans layers before
    # run into this function
    ori_inchannel_step = int(convtrans.in_channels/convtrans.groups)
    ori_outchannel_step = int(convtrans.out_channels/convtrans.groups)
    new_inchannel_step = new_outchannel_step = None
    for groupid in range(convtrans.groups):
        in_start = groupid * ori_inchannel_step
        in_end = in_start + ori_inchannel_step
        out_start = groupid * ori_outchannel_step
        out_end = out_start + ori_outchannel_step
        current_input_index = list(
            filter(lambda x: in_start <= x and x < in_end, remained_in.tolist()))
        current_output_index = list(
            filter(lambda x: out_start <= x and x < out_end, remained_out.tolist()))
        if len(current_input_index) == 0:
            # if the whole group are pruned
            continue
        else:
            new_inchannel_step = len(current_input_index)
            new_outchannel_step = len(current_output_index)
            break
    tmp_weight = torch.ones(
        n_remained_in, new_outchannel_step, k_size1, k_size2)
    tmp_weight = tmp_weight.to(convtrans.weight.device)

    if new_inchannel_step == 0 or new_outchannel_step == 0:
        raise EmptyLayerError()
    if n_remained_in % new_inchannel_step != 0 or n_remained_out % new_outchannel_step != 0:
        raise UnBalancedGroupError()

    new_groups = 0
    for groupid in range(convtrans.groups):
        # copy the weights of this group
        in_start = groupid * ori_inchannel_step
        in_end = in_start + ori_inchannel_step
        out_start = groupid * ori_outchannel_step
        out_end = out_start + ori_outchannel_step
        current_input_index = list(
            filter(lambda x: in_start <= x and x < in_end, remained_in.tolist()))
        current_output_index = list(
            filter(lambda x: out_start <= x and x < out_end, remained_out.tolist()))
        # remap the global index to the group index
        # in the convtranspose layer, the groups are on
        # the output channel dimension
        current_output_index = [x-out_start for x in current_output_index]
        if len(current_input_index) == 0:
            # if the whole group are pruned
            assert len(current_output_index) == 0
            continue
        # check if the number of remained channel of each group are the same
        if len(current_input_index) != new_inchannel_step or len(current_output_index) != new_outchannel_step:
            raise UnBalancedGroupError()

        # copy the weight into tmp_weight
        new_in_start = new_inchannel_step * new_groups
        new_in_end = new_in_start + new_inchannel_step
        tmp_weight[new_in_start:new_in_end] = torch.index_select(
            convtrans.weight[current_input_index], 1, torch.as_tensor(current_output_index, dtype=torch.long).to(convtrans.weight.device))
        new_groups += 1

    _logger.debug('Replace convtranspose2d with in_channels:%d out_channels:%d',
                  n_remained_in, n_remained_out)
    new_convtrans = torch.nn.ConvTranspose2d(in_channels=n_remained_in,
                                             out_channels=n_remained_out,
                                             kernel_size=convtrans.kernel_size,
                                             stride=convtrans.stride,
                                             padding=convtrans.padding,
                                             output_padding=convtrans.output_padding,
                                             dilation=convtrans.dilation,
                                             groups=new_groups,
                                             bias=convtrans.bias is not None,
                                             padding_mode=convtrans.padding_mode)
    new_convtrans.to(convtrans.weight.device)
    new_convtrans.weight.copy_(tmp_weight)
    if convtrans.bias is not None:
        if output_mask is not None:
            new_convtrans.bias.data[:] = torch.index_select(
                convtrans.bias.data, 0, remained_out)
        else:
            new_convtrans.bias.data.copy_(convtrans.bias.data)
    return new_convtrans


def replace_layernorm(layernorm, masks):
    in_masks, _, _ = masks
    assert isinstance(layernorm, nn.LayerNorm)
    if len(in_masks) != 1:
        raise InputsNumberError()
    in_mask = in_masks[0]

    old_normalized_shape = layernorm.normalized_shape
    new_normalized_shape = []
    remained_list = []
    for i in range(-len(old_normalized_shape), 0):
        pruned, remained = convert_to_coarse_mask(in_mask, i)
        new_normalized_shape.append(old_normalized_shape[i] - pruned.size()[i])
        remained_list.append(remained)

    new_layernorm = nn.LayerNorm(tuple(new_normalized_shape), layernorm.eps, layernorm.elementwise_affine)
    _logger.info(f"replace LayerNorm with new normalized_shape: {tuple(new_normalized_shape)}")

    if new_layernorm.elementwise_affine:
        new_layernorm.to(layernorm.weight.device)
        # NOTE: should we keep the weight & bias?
        with torch.no_grad():
            tmp_weight_data = layernorm.weight.data
            tmp_bias_data = layernorm.bias.data
            for i, remained in enumerate(remained_list):
                tmp_weight_data = torch.index_select(tmp_weight_data, i, remained)
                tmp_bias_data = torch.index_select(tmp_bias_data, i, remained)
            new_layernorm.weight.data = tmp_weight_data
            new_layernorm.bias.data = tmp_bias_data
    return new_layernorm


def replace_embedding(embedding, masks):
    """
    Replace the embedding layer according the infered masks.
    We replace the embedding layer according the weight masks,
    """
    # currently we donnot support replace the embedding layer
    # because we donnot have the corressponding pruner
    in_masks, output_mask, weight_mask = masks
    assert isinstance(embedding, nn.Embedding)
    if len(in_masks) != 1:
        raise InputsNumberError()
    if not isinstance(output_mask, torch.Tensor):
        raise OutputTypeError(type(output_mask), torch.Tensor)

    weight_mask = weight_mask['weight']

    # never prune num_embeddings
    n_remained_in = weight_mask.shape[0]
    pruned_out, remained_out = convert_to_coarse_mask(output_mask, -1)
    n_remained_out = weight_mask.size(1) - pruned_out.size(0)
    # this is a workaround when the input always be a tensor contains same value, often happened in transformer
    if n_remained_out == 0:
        _logger.warning("Embedding out masks remained out size is 0, using weight masks remained out instead")
        pruned_out, remained_out = convert_to_coarse_mask(weight_mask, -1)
        n_remained_out = weight_mask.size(1) - pruned_out.size(0)
    remained_out = remained_out.to(embedding.weight.device)
    _logger.info("replace embedding with new in_features: %d, out_features: %d",
                 n_remained_in, n_remained_out)

    new_embedding = torch.nn.Embedding(n_remained_in, n_remained_out)
    new_embedding.padding_idx = embedding.padding_idx
    new_embedding.max_norm = embedding.max_norm
    new_embedding.norm_type = embedding.norm_type
    new_embedding.scale_grad_by_freq = embedding.scale_grad_by_freq
    new_embedding.sparse = embedding.sparse
    new_embedding.to(embedding.weight.device)

    # Copy the remained weight from the original module
    with torch.no_grad():
        new_embedding.weight.data = torch.index_select(
            embedding.weight.data, 1, remained_out)

    return new_embedding


def replace_pixelshuffle(pixelshuffle, masks):
    """
    This is a nearly `no_replace` function.

    We can not replace pixelshuffle easily right now, pixelshuffle is a kind of location mapping.
    It will map tensor with shape (r^2 * C, H, W) to (C, r * H, r* W). So we have a dependency here,
    the preserved input channel number should be a multiple of C, and the multiple can be squared to positive integer.
    This dependence is similar to the group dependency in ConvXD, but more restrictive,
    i.e., each `r^2 input channels` group can not be free to preserve any number of channels, must be a number in [1, 4, 9, 16, ... , r^2].
    """
    in_masks, output_mask, _ = masks
    assert isinstance(pixelshuffle, torch.nn.PixelShuffle)
    if len(in_masks) != 1:
        raise InputsNumberError()
    in_mask = in_masks[0]

    # FIXME: This should be a correct replacement logic, but since we can't correctly generate qualified masks,
    # most of the time this is a no_replace.
    _, remained_in = convert_to_coarse_mask(in_mask, 1)
    _, remained_out = convert_to_coarse_mask(output_mask, 1)
    in_channel_num, out_channel_num = remained_in.size(0), remained_out.size(0)
    upscale_factor = math.floor(math.sqrt(in_channel_num / out_channel_num))

    if in_channel_num != out_channel_num * (upscale_factor * upscale_factor):
        err_msg = "Your speedup model may encounter shape mismatch error during inference. "
        err_msg += f"PixelShuffle preserved input channel number is {in_channel_num}, "
        err_msg += f"preserved output channel number is {out_channel_num}, "
        err_msg += "unable to find a suitable upscale_factor, keep it as it is, please replace this module manually, "
        err_msg += "or adjust the module sparsity ratio before this module to ensure that a suitable upscale_factor can be found."
        # Don't raise an error because the user maybe know how to manually replace this function.
        _logger.error(err_msg)
        # NOTE: no_replace, use the orignal upscale_factor if we can not find a suitable upscale_factor.
        upscale_factor = pixelshuffle.upscale_factor

    if upscale_factor != pixelshuffle.upscale_factor:
        war_msg = f"Change PixelShuffle upscale_factor from {pixelshuffle.upscale_factor} to {upscale_factor}, "
        war_msg += "subsequent computation semantics may have changed."
        _logger.warning(war_msg)

    new_pixelshuffle = torch.nn.PixelShuffle(upscale_factor)
    return new_pixelshuffle
