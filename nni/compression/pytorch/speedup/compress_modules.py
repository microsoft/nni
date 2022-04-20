# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
import torch.nn as nn
from .error_code import EmptyLayerError, ShapeMisMatchError, InputsNumberError, OutputTypeError, UnBalancedGroupError

_logger = logging.getLogger(__name__)

replace_module = {
    'BatchNorm2d': lambda module, masks: replace_batchnorm2d(module, masks),
    'BatchNorm1d': lambda module, masks: replace_batchnorm1d(module, masks),
    'Conv2d': lambda module, masks: replace_conv2d(module, masks),
    'Linear': lambda module, masks: replace_linear(module, masks),
    'MaxPool2d': lambda module, masks: no_replace(module, masks),
    'AvgPool2d': lambda module, masks: no_replace(module, masks),
    'AdaptiveAvgPool2d': lambda module, masks: no_replace(module, masks),
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
    'ConvTranspose2d': lambda module, masks: replace_convtranspose2d(module, masks)
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
    dim_list.remove(dim)

    t_merged = torch.sum(t_mask, dim_list)
    assert t_merged.size(0) == shape[dim]
    all_pruned = t_merged == 0
    need_remain = t_merged != 0
    # return the indexes of the sparsity that can be removed
    indexes = torch.nonzero(all_pruned, as_tuple=True)[0]
    remained_indexes = torch.nonzero(need_remain, as_tuple=True)[0]
    return indexes, remained_indexes


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
    pruned_in, remained_in = convert_to_coarse_mask(in_mask, 1)
    pruned_out, remained_out = convert_to_coarse_mask(output_mask, 1)
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
        current_input_index = list(
            filter(lambda x: in_start <= x and x < in_end, remained_in.tolist()))
        current_output_index = list(
            filter(lambda x: out_start <= x and x < out_end, remained_out.tolist()))
        # remap the global index to the group index
        if len(current_input_index) == 0:
            # if the whole group are pruned
            continue
        else:

            new_inchannel_step = len(current_input_index)
            new_outchannel_step = len(current_output_index)
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
        current_input_index = list(
            filter(lambda x: in_start <= x and x < in_end, remained_in.tolist()))
        current_output_index = list(
            filter(lambda x: out_start <= x and x < out_end, remained_out.tolist()))
        # remap the global index to the group index
        current_input_index = [x-in_start for x in current_input_index]
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

    _logger.debug("replace conv2d with in_channels: %d, out_channels: %d",
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
    dim_n = len(in_mask.size())
    new_shape = []
    for i in range(1, dim_n):
        sum_dims = list(range(0, dim_n))
        sum_dims.remove(i)
        reduced = torch.sum(in_mask, sum_dims)
        n_remained = torch.sum(reduced > 0)
        new_shape.append(n_remained)

    return nn.LayerNorm(tuple(new_shape), layernorm.eps, layernorm.elementwise_affine)
