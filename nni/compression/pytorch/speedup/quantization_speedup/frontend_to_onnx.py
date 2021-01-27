import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
import onnx.numpy_helper

class LayernameModuleWrapper(torch.nn.Module):
    def __init__(self, module, module_bit) -> None:
        super().__init__()
        self.module = module
        self.module_bit = module_bit
    
    def forward(self, inputs):
        inputs = inputs*self.module_bit
        inputs = self.module(inputs)
        return inputs

def _setattr(model, name, module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)

def unwrapper(model_onnx):
    """
    Fill onnx config and remove wrapper node in onnx
    """
    # Support Gemm and Conv
    support_op = ['Gemm', 'Conv']
    idx = 0
    onnx_config = {}
    while idx < len(model_onnx.graph.node):
        nd = model_onnx.graph.node[idx]
        if nd.name[0:4] in support_op and  idx > 1:
            # Grad constant node and multiply node
            const_nd = model_onnx.graph.node[idx-2]
            mul_nd = model_onnx.graph.node[idx-1]
            # Get bit number which is transferred by constant node
            bit = int(onnx.numpy_helper.to_array(const_nd.attribute[0].t))
            onnx_config[nd.name] = bit
            nd.input[0] = mul_nd.input[0]
            # Remove constant node and multiply node
            model_onnx.graph.node.remove(const_nd)
            model_onnx.graph.node.remove(mul_nd)
            idx = idx-2
        idx = idx+1
    return model_onnx, onnx_config

def torch_to_onnx(model, config, input_shape, model_path, input_names, output_names):
    """
    Convert torch model to onnx model and get layer bit config of onnx model.
    """
    # Support Gemm and Conv
    support_op = [torch.nn.Conv2d, torch.nn.Linear]
    # Transfer bit number to onnx layer by using wrapper
    for name, module in model.named_modules():
        if config is not None and name in config:
            assert type(module) in support_op
            wrapper_module = LayernameModuleWrapper(module, config[name])
            _setattr(model, name, wrapper_module)
        elif type(module) in support_op:
            wrapper_module = LayernameModuleWrapper(module, 32)
            _setattr(model, name, wrapper_module)
    # Convert torch model to onnx model and save it in model_path
    dummy_input = torch.randn(input_shape)
    model.to('cpu')
    torch.onnx.export(model, dummy_input, model_path, verbose=False, input_names=input_names, output_names=output_names, export_params=True)

    # Load onnx model
    model_onnx = onnx.load(model_path)
    model_onnx, onnx_config = unwrapper(model_onnx)
    onnx.save(model_onnx, model_path)

    onnx.checker.check_model(model_onnx)
    return model_onnx, onnx_config