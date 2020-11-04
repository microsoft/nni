# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from nni.compression.pytorch.compressor import PrunerModuleWrapper

def _get_params(m):
    return sum([p.numel() for p in m.parameters()])

def count_convNd(m, x, y):
    cin = m.in_channels
    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    output_size = torch.zeros(y.size()[2:]).numel()
    cout = y.size()[1]

    if hasattr(m, "weight_mask"):
        cout = m.weight_mask.sum() // (cin * kernel_ops)

    total_ops = cout * output_size * kernel_ops * cin // m.groups  # cout x oW x oH
    m.total_ops.append(total_ops)
    m.total_params.append(_get_params(m))
    m.input_sizes.append(tuple(x[0].size()))
    m.output_sizes.append(tuple(y.size()))
    m.weight_size = tuple(m.weight.size())
    m.num_calls += 1

def count_linear(m, x, y):
    out_features = m.out_features
    if hasattr(m, "weight_mask"):
        out_features = m.weight_mask.sum() // m.in_features

    m.total_ops.append(out_features * m.in_features)
    m.total_params.append(_get_params(m))
    m.input_sizes.append(tuple(x[0].size()))
    m.output_sizes.append(tuple(y.size()))
    m.weight_size = tuple(m.weight.size())
    m.num_calls += 1

def count_bn(m, x, y):
    m.total_ops.append(2 * x[0].numel())
    m.total_params.append(_get_params(m))
    m.input_sizes.append(tuple(x[0].size()))
    m.output_sizes.append(tuple(y.size()))

def count_relu(m, x, y):
    m.total_ops.append(x[0].numel())
    m.input_sizes.append(tuple(x[0].size()))
    m.output_sizes.append(tuple(y.size()))

def count_avgpool(m, x, y):
    m.total_ops.append(y.numel())
    m.input_sizes.append(tuple(x[0].size()))
    m.output_sizes.append(tuple(y.size()))


def count_adap_avgpool(m, x, y):
    kernel = torch.DoubleTensor([*(x[0].shape[2:])]) // torch.DoubleTensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    m.total_ops.append(kernel_ops * num_elements)
    m.input_sizes.append(tuple(x[0].size()))
    m.output_sizes.append(tuple(y.size()))


def count_upsample(m, x, y):
    if m.mode not in ("linear", "bilinear", "bicubic"): 
        return

    m.input_sizes.append(tuple(x[0].size()))
    m.output_sizes.append(tuple(y.size()))

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

    m.total_ops.append(total_ops)

def count_naive(m, _, __):
    m.num_calls = 0


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
    'full': DEFAULT_OPS.update(OTHER_OPS) or DEFAULT_OPS
}

def format_results(modules, show_data_size = False):

    max_width = []
    # get column width
    for label in ["name", "module_type", "weight_size"]:
        max_width.append(max([len(module[label] if label != "weight_size" else str(module[label])) for module in modules]) + 2)

    has_multi_use = True if len(list(filter(lambda x: x > 1, [module['num_calls'] for module in modules]))) else False

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

    DEFAULT_COLUMN_WIDTH = 10
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
        result.append('\n') 
        
    add_column(DEFAULT_COLUMN_WIDTH) # index column

    for width in max_width:
        add_column(width)
        
    for _ in range(2):
        add_column(DEFAULT_COLUMN_WIDTH) # flops and params column

    if has_multi_use:
        add_column(DEFAULT_COLUMN_WIDTH)

    row_format = row_format_lst[0]
    header_sep = header_sep_lst[0]
    add_column = None  
    append(header_sep)
    append(row_format.format(*headers))
    
    total_ops, total_params = 0, 0
    for i, module in enumerate(modules):
        row_values = [
            i,
            module['name'],  # Name
            module['module_type'],
            str(module['weight_size']),
            sum(module['flops']),
            sum(module['params']),
        ]
        
        if has_multi_use:
            row_values.append(module['num_calls'])

        total_ops += sum(module['flops'])
        total_params += sum(module['params'])

        append(row_format.format(*row_values))

    append(header_sep)
    result.append(f'FLOPs total: {total_ops} \n')
    result.append(f'#Params total: {total_params} \n')

    return ''.join(result)

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
        the input shape of data or the input data
    custom_ops: dict
        a mapping of (module: custom operation)
        the custom operation will overwrite the default operation.
        for reference, please see ``custom_mask_ops``.
    verbose: bool
        print the detail information about modules

    Returns
    -------
    flops: float
        total flops of the model
    params:
        total params of the model
    results: dict
        detail information of modules. 
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
    if custom_ops is None:
        custom_ops = {}
        
    register_ops = mode_cls[mode]
    register_ops.update(custom_ops)

    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return

        m_.total_ops = list()
        m_.total_params = list()
        m_.num_calls = m_.weight_size = 0
        m_.input_sizes = list()
        m_.output_sizes = list()

        m_type = type(m_)
        

        fn = register_ops.get(m_type, count_naive)

        if fn is not None:
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    def remove_buffer(m_):
        if len(list(m_.children())) > 0:
            return

        del m_.total_ops, m_.total_params, m_.num_calls, m_.input_sizes, m_.output_sizes, m_.weight_size
        

    model.eval()
    model.apply(add_hooks)

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

        total_ops += sum(m.total_ops)
        total_params += sum(m.total_params)

        if m.num_calls > 0:
            results.append({
                "name": name,
                "module_type": type(m).__name__,
                "flops": m.total_ops,
                "params": m.total_params,
                "num_calls": m.num_calls,
                "weight_size": m.weight_size,
                "input_sizes": m.input_sizes,
                "output_sizes": m.output_sizes
                }
            )
        prev_m = m

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    model.apply(remove_buffer)
    fomated_results = format_results(results)

    if verbose:
        print(fomated_results)

    return total_ops, total_params, results
