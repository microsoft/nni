# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import torch

import numpy as np

from torch import nn
from torch.autograd import Variable


def count_flops(model=None, in_shape=(3, 112, 112), multiply_adds=False):
    """Compute the flops of model."""
    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2["names"] = np.prod(input[0].shape)

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        bias_ops = 1 if self.bias is not None else 0

        num_weight_params_non_zero = (self.weight.data != 0).float().sum()
        num_weight_params_zero = (self.weight.data == 0).float().sum()
        num_weight_params = num_weight_params_non_zero + num_weight_params_zero
        if self.groups == 1:
            ops = num_weight_params * (2 if multiply_adds else 1)
        else:
            multiplys = num_weight_params / self.groups
            adds = multiplys - output_channels
            ops = (multiplys + adds) if multiply_adds else adds
        flops = (
            (ops + bias_ops * output_channels)
            * output_height
            * output_width
            * batch_size
        )
        list_conv.append(flops)

    list_deconv = []

    def deconv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        bias_ops = 1 if self.bias is not None else 0

        num_weight_params_non_zero = (self.weight.data != 0).float().sum()
        num_weight_params_zero = (self.weight.data == 0).float().sum()
        num_weight_params = num_weight_params_non_zero + num_weight_params_zero
        ops = num_weight_params * (2 if multiply_adds else 1)
        flops = (
            (ops + bias_ops * output_channels)
            * output_height
            * output_width
            * batch_size
        )
        list_deconv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement() if self.bias is not None else 0

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        flops = (
            (kernel_ops + bias_ops)
            * output_channels
            * output_height
            * output_width
            * batch_size
        )

        list_pooling.append(flops)

    list_upsample = []

    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        output_area = output_height * output_width
        flops = output_area * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.ConvTranspose2d):
                net.register_forward_hook(deconv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(
                net, torch.nn.AvgPool2d
            ):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    model.eval()
    with torch.no_grad():
        input_data = torch.rand(3, in_shape[0], in_shape[1], in_shape[2])
        input = Variable(input_data, requires_grad=False)
        out = model(input)
    total_flops = (
        sum(list_conv)
        + sum(list_deconv)
        + sum(list_linear)
        + sum(list_bn)
        + sum(list_relu)
        + sum(list_pooling)
        + sum(list_upsample)
    )
    # batchsize=3
    del input, out
    return total_flops / 3


def model_init(model, state_dict, replace=[]):
    """Initialize the model from state_dict."""
    prefix = "module."
    param_dict = dict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k[7:]
        param_dict[k] = v

    for k, (name, m) in enumerate(model.named_modules()):
        if replace:
            for layer_replace in replace:
                assert len(layer_replace) == 3, "The elements should be three."
                pre_scope, key, replace_key = layer_replace
                if pre_scope in name:
                    name = name.replace(key, replace_key)

        # Copy the state_dict to current model
        if (name + ".weight" in param_dict) or (
            name + ".running_mean" in param_dict
        ):
            if isinstance(m, nn.BatchNorm2d):
                shape = m.running_mean.shape
                if shape == param_dict[name + ".running_mean"].shape:
                    if m.weight is not None:
                        m.weight.data = param_dict[name + ".weight"]
                        m.bias.data = param_dict[name + ".bias"]
                    m.running_mean = param_dict[name + ".running_mean"]
                    m.running_var = param_dict[name + ".running_var"]

            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                shape = m.weight.data.shape
                if shape == param_dict[name + ".weight"].shape:
                    m.weight.data = param_dict[name + ".weight"]
                    if m.bias is not None:
                        m.bias.data = param_dict[name + ".bias"]

            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data = param_dict[name + ".weight"]
                if m.bias is not None:
                    m.bias.data = param_dict[name + ".bias"]
