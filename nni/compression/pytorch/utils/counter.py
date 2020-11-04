# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import DefaultDict
from collections import defaultdict
from prettytable import PrettyTable
import torch
import torch.nn as nn
from nni.compression.pytorch.compressor import PrunerModuleWrapper
import functools

def _get_params(m):
    return sum([p.numel() for p in m.parameters()])

def count_convNd(m, x, y, add_results):
    cin = m.in_channels
    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    output_size = torch.zeros(y.size()[2:]).numel()
    cout = y.size()[1]

    if hasattr(m, "weight_mask"):
        cout = m.weight_mask.sum() // (cin * kernel_ops)

    total_ops = cout * output_size * kernel_ops * cin // m.groups  # cout x oW x oH

    results = {
        "flops": total_ops,
        "params": _get_params(m),
        "weight_size": tuple(m.weight.size()),
        "input_size": tuple(x[0].size()),
        "output_size": tuple(y.size()),
        "module_type": type(m).__name__
    }

    add_results(m._name, **results)

def count_linear(m, x, y, add_results):
    out_features = m.out_features
    if hasattr(m, "weight_mask"):
        out_features = m.weight_mask.sum() // m.in_features
    total_ops = out_features * m.in_features

    results = {
        "flops": total_ops,
        "params": _get_params(m),
        "weight_size": tuple(m.weight.size()),
        "input_size": tuple(x[0].size()),
        "output_size": tuple(y.size()),
        "module_type": type(m).__name__
    }

    add_results(m._name, **results)

def count_bn(m, x, y, add_results):
    results = {
        "flops": 2 * x[0].numel(),
        "params": _get_params(m),
        "weight_size": tuple(m.weight.size()),
        "input_size": tuple(x[0].size()),
        "output_size": tuple(y.size()),
        "module_type": type(m).__name__
    }

    add_results(m._name, **results)


def count_relu(m, x, y, add_results):
    results = {
        "flops": x[0].numel(),
        "params": 0,
        "weight_size": tuple(m.weight.size()),
        "input_size": tuple(x[0].size()),
        "output_size": tuple(y.size()),
        "module_type": type(m).__name__
    }

    add_results(m._name, **results)

def count_avgpool(m, x, y, add_results):
    print("avgpool", y.numel())
    results = {
        "flops": y.numel(),
        "params": 0,
        "weight_size": 0,
        "input_size": tuple(x[0].size()),
        "output_size": tuple(y.size()),
        "module_type": type(m).__name__
    }

    add_results(m._name, **results)


def count_adap_avgpool(m, x, y, add_results):
    kernel = torch.DoubleTensor([*(x[0].shape[2:])]) // torch.DoubleTensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()

    results = {
        "flops": kernel_ops * num_elements,
        "params": 0,
        "weight_size": 0,
        "input_size": tuple(x[0].size()),
        "output_size": tuple(y.size()),
        "module_type": type(m).__name__
    }

    add_results(m._name, **results)


def count_upsample(m, x, y, add_results):
    if m.mode not in ("linear", "bilinear", "bicubic"): 
        return

    if m.mode == "linear":
        total_ops = y.nelement() * 5  # 2 muls + 3 add
    elif m.mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = y.nelement() * 11  # 6 muls + 5 adds
    elif m.mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] x [4x4] x [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = y.nelement() * (ops_solve_A + ops_solve_p)
    elif m.mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = y.nelement() * (13 * 2 + 5)

    results = {
        "flops": total_ops,
        "params": 0,
        "weight_size": tuple(m.weight.size()),
        "input_size": tuple(x[0].size()),
        "output_size": tuple(y.size()),
        "module_type": type(m).__name__
    }

    add_results(m._name, **results)

def count_naive(m, _, __, add_results):
    pass

DEFAULT_OPS = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.Linear: count_linear
}

OTHER_OPS = {
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,
    
    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,
    
    nn.LeakyReLU: count_relu,
    
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    
    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample
}

mode_cls = {
    'default': DEFAULT_OPS,
    'full': {k: v for d in [DEFAULT_OPS, OTHER_OPS] for k, v in d.items()}
}

def format_results(modules):
    table = PrettyTable()
    has_multi_use = True if len(list(filter(lambda x: len(x) > 1, [module['flops'] for module in modules.values()]))) else False

    headers = [
        'Index', 
        'Name',
        'Module',
        'Weight Size',
        'FLOPs',
        '#Params',
    ]
    if has_multi_use:
        headers.append('#Calls')

    table.field_names = headers
    
    total_ops, total_params = 0, 0
    for i, (name, module) in enumerate(modules.items()):
        if len(module['flops']) == 0: continue
        # print(name, module)
        row_values = [
            i,
            name,  # Name
            module['module_type'][0],
            str(module['weight_size'][0]),
            sum(module['flops']),
            module['params'][0],
        ]
        
        if has_multi_use:
            row_values.append(len(module['flops']))

        total_ops += row_values[4]
        total_params += row_values[5]
        table.add_row(row_values)
        
    return table


def count_flops_params(model: nn.Module, input, custom_ops=None, verbose=True, mode='default'):
    """
    Count FLOPs and Params of the given model.
    This function would identify the mask on the module
    and take the pruned shape into consideration.
    Note that, for sturctured pruning, we only identify
    the remained filters according to its mask, which
    not taking the pruned input channels into consideration,
    so the calculated FLOPs will be larger than real number.

    Parameters
    ---------
    model : nn.Module
        target model.
    input: tuple or tensor
        the input shape of data or a tensor as input data
    custom_ops: dict
        a mapping of (module: custom operation)
        the custom operation will overwrite the default operation.
        for reference, please see ``custom_mask_ops``.
    verbose: bool
        If False, mute detail information about modules. Default is True.
    mode:
        the mode of how to collect information. If the mode is set to `default`, only the information of convolution and linear will be collected. If the mode is set to `full`, other operations will also be collected. 

    Returns
    -------
    flops: float
        total flops of the model
    params:
        total params of the model
    results: dict
        detail information of modules
    """

    assert isinstance(input, tuple) or isinstance(input, torch.Tensor)
    assert mode in ['default', 'full']

    original_device = next(model.parameters()).device
    training = model.training

    if torch.is_tensor(input[0]):
        x = (t.to(original_device) for t in input)
    else:
        x = (torch.zeros(input).to(original_device), )

    handler_collection = []
    results = dict()
    if custom_ops is None:
        custom_ops = {}
        
    register_ops = mode_cls[mode]
    register_ops.update(custom_ops)

    # set leaf module name
    prev_m = None
    for name, m in model.named_modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        setattr(m, "_name", name)
        results[name] = defaultdict(list)

        if isinstance(prev_m, PrunerModuleWrapper):
            weight_mask = prev_m.weight_mask
            m.weight_mask = weight_mask

        prev_m = m

    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return

        m_type = type(m_)
        fn = register_ops.get(m_type, count_naive)

        def _add_results(*args, **kwargs):
            name = args[0]
            for k, v in kwargs.items():
                if k in ['params', 'module_type', 'weight_size']:
                    results[name][k] = [v]
                else:
                    results[name][k].append(v)

        if fn is not None:
            _handler = m_.register_forward_hook(functools.partial(fn, add_results=_add_results))
            handler_collection.append(_handler)

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*x)


    # restore origin status
    for name, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue

        delattr(m, "_name")

        if not len(results[name]["flops"]) > 0:
            del results[name]
            if not verbose:
                print(f'Module {name} of type {type(m)} is not used.')
            continue

        if hasattr(m, "weight_mask"):
            delattr(m, "weight_mask")
        
    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    # get detail information
    table = format_results(results)
    total_ops = sum([sum(v["flops"]) for v in results.values()])
    total_params = sum([v["params"][0] for v in results.values()])
    
    if verbose:
        print(table)
        print(f'FLOPs total: {total_ops}')
        print(f'#Params total: {total_params}')

    return total_ops, total_params, results


def test_flops_params():
    class Model1(nn.Module):
        def __init__(self):
            super(Model1, self).__init__()
            self.conv = nn.Conv2d(3, 5, 1, 1)

        def forward(self, x):
            return self.conv(x)

    class Model2(nn.Module):
        def __init__(self):
            super(Model2, self).__init__()
            self.conv = nn.Conv2d(3, 5, 1, 1)
            self.conv2 = nn.Conv2d(5, 5, 1, 1)

        def forward(self, x):
            x = self.conv(x)
            for _ in range(5):
                x = self.conv2(x)
            return x
    
    flops, params, results = count_flops_params(Model1(), (1, 3, 2, 2), mode='full')
    assert (flops, params)  == (60, 20)

    flops, params, results = count_flops_params(Model2(), (1, 3, 2, 2))
    assert (flops, params)  == (560, 50)

    from torchvision.models import resnet18, resnet50
    flops, params, results = count_flops_params(resnet50(), (1, 3, 224, 224))
    assert (flops, params) == (4089184256, 25503912)

    x = torch.randn(1, 3, 224, 224)
    flops, params, results = count_flops_params(resnet18(), (x, ))
    assert (flops, params) == (1814073344, 11679912)
    print(flops, params)


if __name__ == '__main__':
    test_flops_params()
