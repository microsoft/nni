# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
from collections import Counter
from prettytable import PrettyTable

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from nni.compression.pytorch.compressor import PrunerModuleWrapper


__all__ = ['count_flops_params']


def _get_params(m):
    return sum([p.numel() for p in m.parameters()])


class ModelProfiler:

    def __init__(self, custom_ops=None, mode='default'):
        """
        ModelProfiler is used to share state to hooks.

        Parameters
        ----------
        custom_ops: dict
            a mapping of (module -> torch.nn.Module : custom operation)
            the custom operation is a callback funtion to calculate
            the module flops, parameters and the weight shape, it will overwrite the default operation.
            for reference, please see ``self.ops``.
        mode:
            the mode of how to collect information. If the mode is set to `default`,
            only the information of convolution, linear and rnn modules will be collected.
            If the mode is set to `full`, other operations will also be collected.
        """
        self.ops = {
            nn.Conv1d: self._count_convNd,
            nn.Conv2d: self._count_convNd,
            nn.Conv3d: self._count_convNd,
            nn.ConvTranspose1d: self._count_convNd,
            nn.ConvTranspose2d: self._count_convNd,
            nn.ConvTranspose3d: self._count_convNd,
            nn.Linear: self._count_linear,
            nn.RNNCell: self._count_rnn_cell,
            nn.GRUCell: self._count_gru_cell,
            nn.LSTMCell: self._count_lstm_cell,
            nn.RNN: self._count_rnn,
            nn.GRU: self._count_gru,
            nn.LSTM: self._count_lstm
        }
        self._count_bias = False
        if mode == 'full':
            self.ops.update({
                nn.BatchNorm1d: self._count_bn,
                nn.BatchNorm2d: self._count_bn,
                nn.BatchNorm3d: self._count_bn,
                nn.LeakyReLU: self._count_relu,
                nn.AvgPool1d: self._count_avgpool,
                nn.AvgPool2d: self._count_avgpool,
                nn.AvgPool3d: self._count_avgpool,
                nn.AdaptiveAvgPool1d: self._count_adap_avgpool,
                nn.AdaptiveAvgPool2d: self._count_adap_avgpool,
                nn.AdaptiveAvgPool3d: self._count_adap_avgpool,
                nn.Upsample: self._count_upsample,
                nn.UpsamplingBilinear2d: self._count_upsample,
                nn.UpsamplingNearest2d: self._count_upsample
            })
            self._count_bias = True

        if custom_ops is not None:
            self.ops.update(custom_ops)

        self.mode = mode
        self.results = []

    def _push_result(self, result):
        self.results.append(result)

    def _get_result(self, m, flops):
        # assume weight is called `weight`, otherwise it's not applicable
        # if user customize the operation, the callback function should
        # return the dict result, inluding calculated flops, params and weight_shape.

        result = {
            'flops': flops,
            'params': _get_params(m),
            'weight_shape': tuple(m.weight.size()) if hasattr(m, 'weight') else 0,
        }
        return result

    def _count_convNd(self, m, x, y):
        cin = m.in_channels
        kernel_ops = torch.zeros(m.weight.size()[2:]).numel()
        output_size = torch.zeros(y.size()[2:]).numel()
        cout = y.size()[1]

        if hasattr(m, 'weight_mask'):
            cout = m.weight_mask.sum() // (cin * kernel_ops)

        total_ops = cout * output_size * kernel_ops * cin // m.groups  # cout x oW x oH

        if self._count_bias:
            bias_flops = 1 if m.bias is not None else 0
            total_ops += cout * output_size * bias_flops

        return self._get_result(m, total_ops)

    def _count_linear(self, m, x, y):
        out_features = m.out_features
        if hasattr(m, 'weight_mask'):
            out_features = m.weight_mask.sum() // m.in_features
        total_ops = out_features * m.in_features

        if self._count_bias:
            bias_flops = 1 if m.bias is not None else 0
            total_ops += out_features * bias_flops

        return self._get_result(m, total_ops)

    def _count_bn(self, m, x, y):
        total_ops = 2 * x[0].numel()
        return self._get_result(m, total_ops)

    def _count_relu(self, m, x, y):
        total_ops = x[0].numel()
        return self._get_result(m, total_ops)

    def _count_avgpool(self, m, x, y):
        total_ops = y.numel()
        return self._get_result(m, total_ops)

    def _count_adap_avgpool(self, m, x, y):
        kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
        total_add = int(torch.prod(kernel))
        total_div = 1
        kernel_ops = total_add + total_div
        num_elements = y.numel()
        total_ops = kernel_ops * num_elements

        return self._get_result(m, total_ops)

    def _count_upsample(self, m, x, y):
        if m.mode == 'linear':
            total_ops = y.nelement() * 5  # 2 muls + 3 add
        elif m.mode == 'bilinear':
            # https://en.wikipedia.org/wiki/Bilinear_interpolation
            total_ops = y.nelement() * 11  # 6 muls + 5 adds
        elif m.mode == 'bicubic':
            # https://en.wikipedia.org/wiki/Bicubic_interpolation
            # Product matrix [4x4] x [4x4] x [4x4]
            ops_solve_A = 224  # 128 muls + 96 adds
            ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
            total_ops = y.nelement() * (ops_solve_A + ops_solve_p)
        elif m.mode == 'trilinear':
            # https://en.wikipedia.org/wiki/Trilinear_interpolation
            # can viewed as 2 bilinear + 1 linear
            total_ops = y.nelement() * (13 * 2 + 5)
        else:
            total_ops = 0

        return self._get_result(m, total_ops)

    def _count_cell_flops(self, input_size, hidden_size, cell_type):
        # h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})
        total_ops = hidden_size * (input_size + hidden_size) + hidden_size

        if self._count_bias:
            total_ops += hidden_size * 2

        if cell_type == 'rnn':
            return total_ops

        if cell_type == 'gru':
            # r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
            # z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
            # n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
            total_ops *= 3

            # r hadamard : r * (~)
            total_ops += hidden_size

            # h' = (1 - z) * n + z * h
            # hadamard hadamard add
            total_ops += hidden_size * 3

        elif cell_type == 'lstm':
            # i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
            # f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
            # o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
            # g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
            total_ops *= 4

            # c' = f * c + i * g
            # hadamard hadamard add
            total_ops += hidden_size * 3

            # h' = o * \tanh(c')
            total_ops += hidden_size

        return total_ops


    def _count_rnn_cell(self, m, x, y):
        total_ops = self._count_cell_flops(m.input_size, m.hidden_size, 'rnn')
        batch_size = x[0].size(0)
        total_ops *= batch_size

        return self._get_result(m, total_ops)

    def _count_gru_cell(self, m, x, y):
        total_ops = self._count_cell_flops(m.input_size, m.hidden_size, 'gru')
        batch_size = x[0].size(0)
        total_ops *= batch_size

        return self._get_result(m, total_ops)

    def _count_lstm_cell(self, m, x, y):
        total_ops = self._count_cell_flops(m.input_size, m.hidden_size, 'lstm')
        batch_size = x[0].size(0)
        total_ops *= batch_size

        return self._get_result(m, total_ops)

    def _get_bsize_nsteps(self, m, x):
        if isinstance(x[0], PackedSequence):
            batch_size = torch.max(x[0].batch_sizes)
            num_steps = x[0].batch_sizes.size(0)
        else:
            if m.batch_first:
                batch_size = x[0].size(0)
                num_steps = x[0].size(1)
            else:
                batch_size = x[0].size(1)
                num_steps = x[0].size(0)

        return batch_size, num_steps

    def _count_rnn_module(self, m, x, y, module_name):
        input_size = m.input_size
        hidden_size = m.hidden_size
        num_layers = m.num_layers

        batch_size, num_steps = self._get_bsize_nsteps(m, x)
        total_ops = self._count_cell_flops(input_size, hidden_size, module_name)

        for _ in range(num_layers - 1):
            if m.bidirectional:
                cell_flops = self._count_cell_flops(hidden_size * 2, hidden_size, module_name) * 2
            else:
                cell_flops = self._count_cell_flops(hidden_size, hidden_size,module_name)
            total_ops += cell_flops

        total_ops *= num_steps
        total_ops *= batch_size
        return total_ops

    def _count_rnn(self, m, x, y):
        total_ops = self._count_rnn_module(m, x, y, 'rnn')

        return self._get_result(m, total_ops)

    def _count_gru(self, m, x, y):
        total_ops = self._count_rnn_module(m, x, y, 'gru')

        return self._get_result(m, total_ops)

    def _count_lstm(self, m, x, y):
        total_ops = self._count_rnn_module(m, x, y, 'lstm')

        return self._get_result(m, total_ops)


    def count_module(self, m, x, y, name):
        # assume x is tuple of single tensor
        result = self.ops[type(m)](m, x, y)
        output_size = y[0].size() if isinstance(y, tuple) else y.size()

        total_result = {
            'name': name,
            'input_size': tuple(x[0].size()),
            'output_size': tuple(output_size),
            'module_type': type(m).__name__,
            **result
        }

        self._push_result(total_result)

    def sum_flops(self):
        return sum([s['flops'] for s in self.results])

    def sum_params(self):
        return sum({s['name']: s['params'] for s in self.results}.values())

    def format_results(self):
        table = PrettyTable()
        name_counter = Counter([s['name'] for s in self.results])
        has_multi_use = any(map(lambda v: v > 1, name_counter.values()))
        name_counter = Counter()  # clear the counter to count from 0

        headers = [
            'Index',
            'Name',
            'Type',
            'Weight Shape',
            'FLOPs',
            '#Params',
        ]
        if has_multi_use:
            headers.append('#Call')

        table.field_names = headers
        for i, result in enumerate(self.results):
            row_values = [
                i,
                result['name'],
                result['module_type'],
                str(result['weight_shape']),
                result['flops'],
                result['params'],
            ]
            name_counter[result['name']] += 1
            if has_multi_use:
                row_values.append(name_counter[result['name']])
            table.add_row(row_values)
        return table


def count_flops_params(model, x, custom_ops=None, verbose=True, mode='default'):
    """
    Count FLOPs and Params of the given model. This function would
    identify the mask on the module and take the pruned shape into consideration.
    Note that, for sturctured pruning, we only identify the remained filters
    according to its mask, and do not take the pruned input channels into consideration,
    so the calculated FLOPs  will be larger than real number.

    Parameters
    ---------
    model : nn.Module
        Target model.
    x : tuple or tensor
        The input shape of data (a tuple), a tensor or a tuple of tensor as input data.
    custom_ops : dict
        A mapping of (module -> torch.nn.Module : custom operation)
        the custom operation is a callback funtion to calculate
        the module flops and parameters, it will overwrite the default operation.
        for reference, please see ``ops`` in ``ModelProfiler``.
    verbose : bool
        If False, mute detail information about modules. Default is True.
    mode : str
        the mode of how to collect information. If the mode is set to ``default``,
        only the information of convolution and linear will be collected.
        If the mode is set to ``full``, other operations will also be collected.

    Returns
    -------
    tuple of int, int and dict
        Representing total FLOPs, total parameters, and a detailed list of results respectively.
        The list of results are a list of dict, each of which contains (name, module_type, weight_shape,
        flops, params, input_size, output_size) as its keys.
    """

    assert isinstance(x, tuple) or isinstance(x, torch.Tensor)
    assert mode in ['default', 'full']

    original_device = next(model.parameters()).device
    training = model.training

    if isinstance(x, tuple) and all(isinstance(t, int) for t in x):
        x = (torch.zeros(x).to(original_device), )
    elif torch.is_tensor(x):
        x = (x.to(original_device), )
    else:
        x = (t.to(original_device) for t in x)

    handler_collection = []
    profiler = ModelProfiler(custom_ops, mode)

    prev_m = None
    for name, m in model.named_modules():
        # dealing with weight mask here
        if isinstance(prev_m, PrunerModuleWrapper):
            # weight mask is set to weight mask of its parent (wrapper)
            weight_mask = prev_m.weight_mask
            m.weight_mask = weight_mask
        prev_m = m

        if type(m) in profiler.ops:
            # if a leaf node
            _handler = m.register_forward_hook(functools.partial(profiler.count_module, name=name))
            handler_collection.append(_handler)

    model.eval()

    with torch.no_grad():
        model(*x)

    # restore origin status
    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    if verbose:
        # get detail information
        print(profiler.format_results())
        print(f'FLOPs total: {profiler.sum_flops()}')
        print(f'#Params total: {profiler.sum_params()}')

    return profiler.sum_flops(), profiler.sum_params(), profiler.results