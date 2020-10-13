# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

# [reference] https://github.com/ShichenLiu/CondenseNet/blob/master/utils.py


def get_num_gen(gen):
    return sum(1 for _ in gen)


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    import operator
    import functools

    return sum([functools.reduce(operator.mul, i.size(), 1) for i in model.parameters()])

count_ops = 0
count_params = 0

def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    # ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    # ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel() / x.size(0)
        delta_params = get_layer_param(layer)

    # ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    # ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = weight_ops + bias_ops
        delta_params = get_layer_param(layer)

    # ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    # unknown layer type
    else:
        delta_params = get_layer_param(layer)

    count_ops += delta_ops
    count_params += delta_params

    return


def measure_model(model, H, W, device):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = torch.zeros(2, 3, H, W).to(device)

    def should_measure(x):
        return is_leaf(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params
