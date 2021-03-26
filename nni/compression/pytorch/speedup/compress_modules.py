# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)

replace_module = {
    'BatchNorm2d': lambda module, auto_infer: replace_batchnorm2d(module, auto_infer),
    'Conv2d': lambda module, auto_infer: replace_conv2d(module, auto_infer),
    'Linear': lambda module, auto_infer: replace_linear(module, auto_infer),
    'MaxPool2d': lambda module, auto_infer: no_replace(module, auto_infer),
    'AvgPool2d': lambda module, auto_infer: no_replace(module, auto_infer),
    'AdaptiveAvgPool2d': lambda module, auto_infer: no_replace(module, auto_infer),
    'ReLU': lambda module, auto_infer: no_replace(module, auto_infer),
    'ReLU6': lambda module, auto_infer: no_replace(module, auto_infer),
    'Dropout': lambda module, auto_infer: no_replace(module, auto_infer),
    'Dropout2d': lambda module, auto_infer: no_replace(module, auto_infer),
    'Dropout3d': lambda module, auto_infer: no_replace(module, auto_infer),
    'LayerNorm': lambda module, auto_infer: replace_layernorm(module, auto_infer),
    'ConvTranspose2d': lambda module, auto_infer: replace_convtranspose2d(module, auto_infer)
}

# NEED_FOLD_BIAS = True


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
    # try to reduce the mask from the 0-th dimension
    dim_list.remove(dim)

    t_merged = torch.sum(t_mask, dim_list)
    assert t_merged.size(0) == shape[dim]
    all_pruned = t_merged == 0
    need_remain = t_merged != 0
    # return the indexes of the sparsity that can be removed
    indexes = torch.nonzero(all_pruned, as_tuple=True)[0]
    remained_indexes = torch.nonzero(need_remain, as_tuple=True)[0]
    return indexes, remained_indexes


def no_replace(module, auto_infer):
    """
    No need to replace
    """
    _logger.debug("no need to replace")
    return module


def replace_linear(linear, auto_infer):
    """
    Parameters
    ----------
    linear : torch.nn.Linear
        The linear module to be replace
    auto_infer : AutoMaskInference
        The auto mask inference object that contains the input,
        parameter and output masks.

    Returns
    -------
    torch.nn.Linear
        The new linear module
    """
    NEED_FOLD_BIAS = auto_infer.fold_bias
    assert isinstance(linear, nn.Linear)
    assert len(auto_infer.in_masks) == 1
    assert isinstance(auto_infer.output_mask, torch.Tensor)
    in_mask = auto_infer.in_masks[0]
    # only need the first batch of the constant
    in_constant = auto_infer.in_constants[0][:1]
    output_mask = auto_infer.output_mask
    weight_mask = auto_infer.weight_mask['weight']
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
    if linear.bias is not None or (NEED_FOLD_BIAS and torch.sum(in_constant > 0)):
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

    if NEED_FOLD_BIAS and torch.sum(in_constant) > 0:
        # we need zero the bias in the original linear before we calculate the
        # the folded bias constant
        if linear.bias is not None:
            linear.bias[:] = 0
        out = linear(in_constant)
        bias_constant = torch.index_select(out[0], 0, remained_out)
        if new_linear.bias is not None:
            new_linear.bias.data += bias_constant
    # print(auto_infer.in_constants[0].size())
    # print(torch.sum(auto_infer.in_constants[0],[0]))
    # print(in_constant)
    # print(bias_constant)
    # exit(-1)
    return new_linear


def replace_batchnorm2d(norm, auto_infer):
    """
    Parameters
    ----------
    norm : torch.nn.BatchNorm2d
        The batchnorm module to be replace
    auto_infer : AutoMaskInference
        The auto mask inference object that contains the input,
        parameter and output masks.

    Returns
    -------
    torch.nn.BatchNorm2d
        The new batchnorm module
    """
    assert isinstance(norm, nn.BatchNorm2d)
    in_mask = auto_infer.in_masks[0]
    output_mask = auto_infer.output_mask
    # print('BN CONSTANT')
    # print(auto_infer.in_constants[0])
    # N, C, H, W
    _, remained_in = convert_to_coarse_mask(in_mask, 1)
    _, remained_out = convert_to_coarse_mask(output_mask, 1)
    assert remained_in.size(0) == remained_out.size(0)

    num_features = remained_in.size(0)
    _logger.info("replace batchnorm2d with num_features: %d", num_features)
    new_norm = torch.nn.BatchNorm2d(num_features=num_features,
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)
    # assign weights
    new_norm.weight.data = torch.index_select(norm.weight.data, 0, remained_in)
    new_norm.bias.data = torch.index_select(norm.bias.data, 0, remained_in)

    new_norm.running_mean.data = torch.index_select(
        norm.running_mean.data, 0, remained_in)
    new_norm.running_var.data = torch.index_select(
        norm.running_var.data, 0, remained_in)
    return new_norm


class BiasModule(nn.Module):
    def __init__(self, module, bias):
        super(BiasModule, self).__init__()
        self.ori_module = module
        self.register_buffer('speedup_bias', bias)

    def forward(self, x):
        return self.ori_module(x)[:] + self.speedup_bias


def replace_conv2d(conv, auto_infer):
    """
    Parameters
    ----------
    conv : torch.nn.Conv2d
        The conv2d module to be replaced
    auto_infer : AutoMaskInference
        The auto mask inference object that contains the mask of the input
        tensor, output tensor and parameters

    Returns
    -------
    torch.nn.Conv2d
        The new conv2d module
    """
    assert isinstance(conv, nn.Conv2d)
    # the conv layer should only have one input tensor
    assert len(auto_infer.in_masks) == 1
    NEED_FOLD_BIAS = auto_infer.fold_bias
    in_mask = auto_infer.in_masks[0]
    in_constant = auto_infer.in_constants[0]
    output_mask = auto_infer.output_mask
    weight_mask = auto_infer.weight_mask['weight']
    pruned_in, remained_in = convert_to_coarse_mask(in_mask, 1)
    pruned_out, remained_out = convert_to_coarse_mask(output_mask, 1)
    # print('%%%%%%%%%%%%%%%%%')
    # print(remained_out)
    # # print('Output mask')
    # print(output_mask)

    # if pruned_in.size(0) == 0 and pruned_out.size(0) == 0:
    #     # if this is not structurally pruned at all
    #     ori_bias = conv.bias
    #     if conv.bias is not None:
    #         conv.bias = torch.zeros_like(ori_bias)
    #     bias_constant = torch.index_select(
    #         conv(in_constant)[0], 0, remained_out)
    #     # print(bias_constant)
    #     # exit(-1)
    #     conv.bias=ori_bias
    #     return BiasModule(conv, bias_constant)
    #     # return conv

    n_remained_in = weight_mask.size(1) * conv.groups - pruned_in.size(0)
    n_remained_out = weight_mask.size(0) - pruned_out.size(0)

    # print(n_remained_out, remained_out.size(0))

    assert n_remained_in == remained_in.size(0)
    assert n_remained_out == remained_out.size(0)

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
            # print(current_input_index)
            # print(current_output_index)
            # exit()
            new_inchannel_step = len(current_input_index)
            new_outchannel_step = len(current_output_index)
            break
    tmp_weight = torch.ones(n_remained_out, new_inchannel_step, k_size1, k_size2)
    tmp_weight = tmp_weight.to(conv.weight.device)
    # print(n_remained_out, new_outchannel_step)
    # print(current_output_index)
    # print(remained_in)
    # print(remained_out)
    # print(conv)
    assert n_remained_in % new_inchannel_step == 0
    assert n_remained_out % new_outchannel_step == 0
    
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
        assert len(current_input_index) == new_inchannel_step
        assert len(current_output_index) == new_outchannel_step
        # copy the weight into tmp_weight
        new_out_start = new_outchannel_step * new_groups
        new_out_end = new_out_start + new_outchannel_step
        tmp_weight[new_out_start:new_out_end] = torch.index_select(
            conv.weight[current_output_index], 1, torch.tensor(current_input_index).to(conv.weight.device))
        new_groups += 1

    _logger.debug("replace conv2d with in_channels: %d, out_channels: %d",
                  n_remained_in, n_remained_out)
    # in_constant only need to consider the first batch
    in_constant = in_constant * (1-in_mask)
    in_constant = in_constant[:1]
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
    
    # if auto_infer.name == 'conv2':
    #     print('in conv2')
    #     pos = torch.abs(in_constant) > 0.0001
    #     print(in_constant[pos])
        # exit()

    if NEED_FOLD_BIAS and torch.sum(torch.abs(in_constant)) > 0:
        # Fold the input constants into the new_conv bias
        # For conv, we can only fold the input constant into
        # bias when all the constant in the same channel are the
        # same.
        # print('CONSTANT HERE!!')
        # print(in_constant)
        # pos= in_constant>0.000001
        # print(torch.sum(pos))
        # print(in_constant[pos])
        # print(torch.sum(in_constant))
        # exit(-1)
        # set the bias to zero and calculate the folded bias for new conv
        if conv.bias is not None:
            conv.bias.data[:] = 0
        bias_constant = torch.index_select(
            conv(in_constant)[0], 0, remained_out)
        # print(bias_constant)
        # exit(-1)
        return BiasModule(new_conv, bias_constant)
    else:
        # print('Fuck me!!!', auto_infer.name)
        return new_conv

def replace_convtranspose2d(module, auto_infer):
    # TODO add the replace function for the convtranspose2d module
    return module

def replace_layernorm(layernorm, auto_infer):
    assert isinstance(layernorm, nn.LayerNorm)
    assert len(auto_infer.in_masks) == 1
    in_mask = auto_infer.in_masks[0]
    dim_n = len(in_mask.size())
    new_shape = []
    for i in range(1, dim_n):
        sum_dims = list(range(0, dim_n))
        sum_dims.remove(i)
        reduced = torch.sum(in_mask, sum_dims)
        n_remained = torch.sum(reduced > 0)
        new_shape.append(n_remained)
    print('Original input shape')
    print(in_mask.size())
    print('new normalized shape')
    print(new_shape)
    print(dim_n)
    print(auto_infer.in_masks)
    print(auto_infer.name)
    # print()
    # exit()
    return nn.LayerNorm(tuple(new_shape), layernorm.eps, layernorm.elementwise_affine)