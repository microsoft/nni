# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from nni.compression.torch.compressor import PrunerModuleWrapper

def _get_params(m):
    return sum([p.numel() for p in m.parameters()])

def count_convNd(m, _, y):
    cin = m.in_channels
    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    output_size = torch.zeros(y.size()[2:]).numel()
    cout = y.size()[1]

    if hasattr(m, "weight_mask"):
        cout = m.weight_mask.sum() // (cin * kernel_ops)

    total_ops = cout * output_size * kernel_ops * cin // m.groups  # cout x oW x oH
    m.total_ops = total_ops
    m.total_params = _get_params(m)
    m.num_calls += 1


def count_linear(m, _, __):
    out_features = m.out_features
    if hasattr(m, "weight_mask"):
        out_features = m.weight_mask.sum() // m.in_features

    m.total_ops = out_features * m.in_features 
    m.total_params = _get_params(m)
    m.num_calls += 1

def count_naive(m, _, __):
    m.num_calls += 1

register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.Linear: count_linear,
}

def format_results(modules):
    name_column_width = max([len(module['name']) for module in modules]) + 4
    has_multi_use = True if len(list(filter(lambda x: x > 1, [module['num_calls'] for module in modules]))) else False
    module_type_column_width = max([len(module['module_type']) for module in modules]) + 4
    headers = [
        'Index', 
        'Name',
        'Module',
        'FLOPs',
        '#Params',
    ]
    if has_multi_use:
        headers.append('#Calls')

    DEFAULT_COLUMN_WIDTH = 12
    SPACING_SIZE = 2
    row_format_lst = [""]
    header_sep_lst = [""]
    line_length_lst = [-SPACING_SIZE]

    def add_column(padding, text_dir='>'):
        row_format_lst[0] += '{: ' + text_dir + str(padding) + '}' + (' ' * SPACING_SIZE)
        header_sep_lst[0] += '-' * padding + (' ' * SPACING_SIZE)
        line_length_lst[0] += padding + SPACING_SIZE

    result = []
    def append(s):
        result.append(s)
        result.append('\n')  # Yes, newline after the end as well
        
    add_column(DEFAULT_COLUMN_WIDTH) # index column
    add_column(name_column_width)
    add_column(module_type_column_width)
    for _ in headers[3:]:
        add_column(DEFAULT_COLUMN_WIDTH)
    row_format = row_format_lst[0]
    header_sep = header_sep_lst[0]
    add_column = None  # type: ignore
    append(header_sep)
    append(row_format.format(*headers))
    
    total_ops, total_params = 0, 0
    for i, module in enumerate(modules):
        row_values = [
            i,
            module['name'],  # Name
            module['module_type'],
            module['flops'],
            module['params']
        ]
        
        if has_multi_use:
            row_values.append(module['num_calls'])
            
        total_ops += module['num_calls'] * module['flops']
        total_params += module['num_calls'] * module['params']

        append(row_format.format(*row_values))

    append(header_sep)
    result.append(f'FLOPs total: {total_ops} \n')
    result.append(f'#Params total: {total_params} \n')

    return ''.join(result)

def count_flops_params(model: nn.Module, input_size, custom_ops=None, verbose=True):
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
    input_size: list, tuple
        the input shape of data
    custom_ops: dict
        a mapping of (module: custom operation)
        the custom operation will overwrite the default operation.
        for reference, please see ``custom_mask_ops``.

    Returns
    -------
    flops: float
        total flops of the model
    params:
        total params of the model
    """
    handler_collection = []

    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return

        m_.total_ops = m_.total_params = 0
        m_.num_calls = 0

        m_type = type(m_)
        

        fn = register_hooks.get(m_type, count_naive)

        if fn is not None:
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    def remove_buffer(m_):
        if len(list(m_.children())) > 0:
            return

        del m_.total_ops, m_.total_params, m_.num_calls
        
    original_device = next(model.parameters()).device
    training = model.training

    model.eval()
    model.apply(add_hooks)

    assert isinstance(input_size, tuple)
    if torch.is_tensor(input_size[0]):
        x = (t.to(original_device) for t in input_size)
    else:
        x = (torch.zeros(input_size).to(original_device), )

    with torch.no_grad():
        model(*x)

    total_ops = 0
    total_params = 0
    results = []

    prev_m = None
    for name, m in model.named_modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        if not m.num_calls:
            if not verbose:
                print(f'Module {name} of type {type(m)} is not used.')
            continue
        
        if isinstance(prev_m, PrunerModuleWrapper):
            weight_mask = prev_m.weight_mask
            m.weight_mask = weight_mask

        total_ops += m.total_ops
        total_params += m.total_params

        if m.total_ops > 0 or m.total_params > 0:
            results.append({
                "name": name,
                "module_type": type(m).__name__,
                "flops": m.total_ops,
                "params": m.total_params,
                "num_calls": m.num_calls
                }
            )
        prev_m = m

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()
    model.apply(remove_buffer)
    results = format_results(results)
    # print(results)

    return total_ops, total_params, results

def count_convNd_mask(m, x, y):
    """
    The forward hook to count FLOPs and Parameters of convolution operation.
    Parameters
    ----------
    m : torch.nn.Module
        convolution module to calculate the FLOPs and Parameters
    x : torch.Tensor
        input data
    y : torch.Tensor
        output data
    """
    output_channel = y.size()[1]
    output_size = torch.zeros(y.size()[2:]).numel()
    kernel_size = torch.zeros(m.weight.size()[2:]).numel()

    bias_flops = 1 if m.bias is not None else 0

    if m.weight_mask is not None:
        output_channel = m.weight_mask.sum() // (m.in_channels * kernel_size)

    total_ops = output_channel * output_size * (m.in_channels // m.groups * kernel_size + bias_flops)

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_linear_mask(m, x, y):
    """
    The forward hook to count FLOPs and Parameters of linear transformation.
    Parameters
    ----------
    m : torch.nn.Module
        linear to calculate the FLOPs and Parameters
    x : torch.Tensor
        input data
    y : torch.Tensor
        output data
    """
    output_channel = y.numel()

    bias_flops = 1 if m.bias is not None else 0

    if m.weight_mask is not None:
        output_channel = m.weight_mask.sum() // m.in_features

    total_ops = output_channel * (m.in_features + bias_flops)

    m.total_ops += torch.DoubleTensor([int(total_ops)])

    
def test_profiler():
    
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(3, 5, 1, 1)

        def forward(self, x):
            return self.conv(x)

    assert count_flops_params(Model(), (1, 3, 2, 2)) == (60, 20)

    class Model2(nn.Module):
        def __init__(self):
            super(Model2, self).__init__()
            self.conv = nn.Conv2d(3, 5, 1, 1)
            self.conv2 = nn.Conv2d(5, 5, 1, 1)
        def forward(self, x):
            x = self.conv(x)
            for i in range(3):
                x = self.conv2(x)
            return x

    from torchvision.models import resnet18, resnet50
    
    assert count_flops_params(resnet50(), (1, 3, 224, 224)) == (4089184256, 25503912)
    
if __name__ == '__main__':
    test_profiler()
