import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx

def torch_to_onnx(model, input_shape, model_path, input_names, output_names):
    # convert torch model to onnx model and save it in model_path
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(model, dummy_input, model_path, verbose=True, input_names=input_names, output_names=output_names, export_params=True)

    # load onnx model
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    return model

