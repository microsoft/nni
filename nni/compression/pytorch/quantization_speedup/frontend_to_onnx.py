# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import onnx
import onnx.numpy_helper
from nni.compression.pytorch.utils.attr import set_nested_attr

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
    # Support Gemm, Conv, Relu, Clip(Relu6) and Maxpool + MatMul
    support_op = ['Gemm', 'Conv', 'Relu', 'Clip', 'MaxP', 'MatMul']
    idx = 0
    onnx_config = {}
    mul_name_list =[]
    const_name_list = []
    const_list = []
    mul_list = []
    #find mul node output name
    for node in model_onnx.graph.node:
        for op in support_op:
            if op in node.name:
                for node_input_name in node.input:
                    if 'Mul_output' in node_input_name:
                        mul_name_list.append(node_input_name)
    #find const node output name by mul node output name
    for node in model_onnx.graph.node:
        if node.output[0] in mul_name_list:
            for node_input_name in node.input:
                if 'Constant_output' in node_input_name:
                    const_name_list.append(node_input_name)    
    # find mul node and const node
    for node in model_onnx.graph.node:
        for nd_name in mul_name_list:
            if node.output[0] == nd_name:
                mul_list.append(node)   
        for nd_name in const_name_list:
            if node.output[0] == nd_name:
                const_list.append(node)
    for node in model_onnx.graph.node:
        for mul_node in mul_list:
            if mul_node.output[0] in node.input:
                # import pdb;pdb.set_trace()
                for const_node in const_list:
                    if const_node.output[0] in mul_node.input:
                        # import pdb;pdb.set_trace()
                        index = int(onnx.numpy_helper.to_array(const_node.attribute[0].t))
                        if index != -1:
                            name = index2name[index]
                            onnx_config[node.name] = config[name]
                        node.input[0] = mul_node.input[0]
                        model_onnx.graph.node.remove(const_node)
                        model_onnx.graph.node.remove(mul_node)
    return model_onnx, onnx_config

def torch_to_onnx(model, config, dummy_input, model_path, input_names, output_names,dynamic_axes=None):
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
    device = torch.device('cpu')
    model.to(device)
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
    if(dynamic_axes == None):
        dynamic_axes = {'input' : {2 : 'image_height',3:'image_wdith'}, #for image  
                        'output' : {2 : 'image_height',3:'image_wdith'}}
    # dummy_input = dummy_input.to(device)
    # model.to(device)
    torch.onnx.export(model, dummy_input, model_path, verbose=False, input_names=input_names, output_names=output_names, export_params=True,opset_version=11,dynamic_axes=dynamic_axes)
    # Load onnx model
    model_onnx = onnx.load(model_path)
    model_onnx, onnx_config = unwrapper(model_onnx, index2name, config)
    onnx.save(model_onnx, model_path)
    onnx.checker.check_model(model_onnx)
    return model_onnx, onnx_config
