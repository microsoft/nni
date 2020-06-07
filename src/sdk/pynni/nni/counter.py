import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm

import logging

_logger = logging.getLogger(__name__)

def count_params(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += torch.DoubleTensor([p.numel()])
    m.total_params[0] = total_params

def count_flops_params(model: nn.Module, input_size=None, verbose=False):
    assert input_size is not None
    handlers = {}
    
    def add_hooks(m: nn.Module):
        m.register_buffer('total_flops', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))

        m_type = type(m)

        fn = None
        if m_type in counter_map:
            fn = counter_map[m_type]
            handlers[m] = (m.register_forward_hook(fn), m.register_forward_hook(count_params))


    device = next(model.parameters()).device
    inputs = torch.randn(input_size).to(device)
    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(inputs)

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        total_flops, total_params = 0, 0
        for m in module.children():

            if m in handlers and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_flops, m_params = m.total_flops.item(), m.total_params.item()
            else:
                m_flops, m_params = dfs_count(m, prefix=prefix + "\t")

            total_flops += m_flops
            total_params += m_params
            
        return total_flops, total_params

    total_flops, total_params = dfs_count(model)

    model.train(prev_training_status)
    for m, (flops_handler, params_handler) in handlers.items():
        flops_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_flops")
        m._buffers.pop("total_params")

    return total_flops, total_params

def count_convNd(m, x, y):

    conv_flops = torch.zeros(m.weight.size()[2:]).numel()
    bias_flops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_flops = y.nelement() * (m.in_channels // m.groups * conv_flops + bias_flops)

    m.total_flops += torch.DoubleTensor([int(total_flops)])

def count_linear(m, x, y):
    total_mul = m.in_features
    num_elements = y.numel()
    total_flops = total_mul * num_elements

    m.total_flops += torch.DoubleTensor([int(total_flops)])

def count_bn(m, x, y):
    total_flops = y.numel()

    if not m.affine:
        total_flops *= 2

    m.total_flops += torch.DoubleTensor([int(total_flops)])

def count_relu(m, x, y):
    total_flops = y.numel()
    
    m.total_flops += torch.DoubleTensor([int(total_flops)])

def count_pool(m, x, y):
    total_flops = y.numel()

    m.total_flops += torch.DoubleTensor([int(total_flops)])

def count_adap_avgpool(m, x, y):
    kernel = torch.DoubleTensor([*(x[0].shape[2:])]) // torch.DoubleTensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_flops = kernel_ops * num_elements

    m.total_flops += torch.DoubleTensor([int(total_flops)])

def zero_ops(m, x, y):
    m.total_flops += torch.DoubleTensor([int(0)])

counter_map = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,
    nn.Linear: count_linear,
    nn.ReLU: count_relu,
    nn.PReLU: count_relu,
    nn.ELU: count_relu,
    nn.LeakyReLU: count_relu,
    nn.ReLU6: count_relu,
    nn.AvgPool1d: count_pool,
    nn.AvgPool2d: count_pool,
    nn.AvgPool3d: count_pool,
    nn.MaxPool1d: count_pool,
    nn.MaxPool2d: count_pool,
    nn.MaxPool3d: count_pool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.ZeroPad2d: zero_ops
}
