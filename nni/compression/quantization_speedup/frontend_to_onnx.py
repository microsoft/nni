# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import onnx
import onnx.numpy_helper
from ..utils import set_nested_attr

"""
The main function of this page is to convert pytorch model to onnx model.
Convertion from pytorch model to onnx model is primary so that a critical
problem is caused that Layer name of pytorch model fail to convert to onnx
layer name directly. To solve it, we wrap pytorch model in new wrapper which
multiply bits number and input before computation of each op. Only in this
way can onnx model get bits number of corresponded layer.
"""

class LayernameModuleWrapper(torch.nn.Module):
    def __init__(self, module, module_bits) -> None:
        """
        Parameters
        ----------
        module : torch.nn.Module
            Layer module of pytorch model
        module_bits : int
            Bits width setting for module
        """
        super().__init__()
        self.module = module
        self.module_bits = module_bits

    def forward(self, inputs):
        inputs = inputs*self.module_bits
        inputs = self.module(inputs)
        return inputs

def unwrapper(model_onnx, index2name, config):
    """
    Fill onnx config and remove wrapper node in onnx

    Parameters
    ----------
    model_onnx : onnx model
        Onnx model which is converted from pytorch model
    index2name : dict
        Dictionary of layer index and name
    config : dict
        Config recording name of layers and calibration parameters

    Returns
    -------
    onnx model
        Onnx model which is converted from pytorch model
    dict
        The configuration of onnx model layers and calibration parameters
    """
    # Support Gemm, Conv, Relu, Clip(Relu6) and Maxpool
    support_op = ['Gemm', 'Conv', 'Relu', 'Clip', 'MaxP']
    idx = 0
    onnx_config = {}
    while idx < len(model_onnx.graph.node):
        nd = model_onnx.graph.node[idx]
        if nd.name[0:4] in support_op and  idx > 1:
            # Grad constant node and multiply node
            const_nd = model_onnx.graph.node[idx-2]
            mul_nd = model_onnx.graph.node[idx-1]
            # Get index number which is transferred by constant node
            index = int(onnx.numpy_helper.to_array(const_nd.attribute[0].t))
            if index != -1:
                name = index2name[index]
                onnx_config[nd.name] = config[name]
            nd.input[0] = mul_nd.input[0]
            # Remove constant node and multiply node
            model_onnx.graph.node.remove(const_nd)
            model_onnx.graph.node.remove(mul_nd)
            idx = idx-2
        idx = idx+1
    return model_onnx, onnx_config

def torch_to_onnx(model, config, input_shape, model_path, input_names, output_names):
    """
    Convert torch model to onnx model and get layer bits config of onnx model.

    Parameters
    ----------
    model : pytorch model
        The model to speedup by quantization
    config : dict
        Config recording bits number and name of layers
    input_shape : tuple
        The input shape of model, shall pass it to torch.onnx.export
    model_path : str
        The path user want to store onnx model which is converted from pytorch model
    input_names : list
        Input name of onnx model providing for torch.onnx.export to generate onnx model
    output_name : list
        Output name of onnx model providing for torch.onnx.export to generate onnx model

    Returns
    -------
    onnx model
        Onnx model which is converted from pytorch model
    dict
        The configuration of onnx model layers and calibration parameters
    """
    # Support Gemm, Conv, Relu, Clip(Relu6) and MaxPool
    support_op = [torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.MaxPool2d]
    # Transfer bits number to onnx layer by using wrapper
    index2name = {}
    name2index = {}
    if config is not None:
        for i, name in enumerate(config.keys()):
            # i*2 to avoid the graph optimization of onnx on constant folding / elimination
            # for example, multiply 1 will be removed from the graph
            index2name[i * 2] = name
            name2index[name] = i * 2
    for name, module in model.named_modules():
        if config is not None and name in config:
            assert type(module) in support_op
            wrapper_module = LayernameModuleWrapper(module, name2index[name])
            set_nested_attr(model, name, wrapper_module)
        elif type(module) in support_op:
            wrapper_module = LayernameModuleWrapper(module, -1)
            set_nested_attr(model, name, wrapper_module)
    # Convert torch model to onnx model and save it in model_path
    device = torch.device('cpu')
    dummy_input = torch.randn(input_shape)
    dummy_input = dummy_input.to(device)
    model.to(device)
    torch.onnx.export(model, dummy_input, model_path, verbose=False, input_names=input_names, output_names=output_names, export_params=True)
    # Load onnx model
    model_onnx = onnx.load(model_path)
    model_onnx, onnx_config = unwrapper(model_onnx, index2name, config)
    onnx.save(model_onnx, model_path)

    onnx.checker.check_model(model_onnx)
    return model_onnx, onnx_config