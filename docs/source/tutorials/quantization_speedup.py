"""
Speed Up Quantized Model with TensorRT
======================================

Quantization algorithms quantize a deep learning model usually in a simulated way. That is, to simulate the effect of low-bit computation with float32 operators, the tensors are quantized to the targeted bit number and dequantized back to float32. Such a quantized model does not have any latency reduction. Thus, there should be a speedup stage to make the quantized model really accelerated with low-bit operators. 
This tutorial demonstrates how to accelerate a quantized model with `TensorRT <https://developer.nvidia.com/tensorrt>`_ as the inference engine in NNI. More inference engines will be supported in future release.

The process of speeding up a quantized model in NNI is that 1) the model with quantized weights and configuration is converted into onnx format, 2) the onnx model is fed into TensorRT to generate an inference engine. The engine is used for low latency model inference.

There are two modes of the speedup: with calibration data and without calibration data. As TensorRT has supported post-training quantization, directly leveraging this functionality is a natural way to use TensorRT. This mode is called "with calibration data". In this mode, the quantization-aware training algorithms (e.g., `QAT <https://nni.readthedocs.io/en/stable/reference/compression/quantizer.html#qat-quantizer>`_, `LSQ <https://nni.readthedocs.io/en/stable/reference/compression/quantizer.html#lsq-quantizer>`_) only take charge of adjusting model weights to be more quantization friendly, and leave the last-step quantization to the post-training quantization of TensorRT. In the other mode "without calibration data", the post-training quantization in TensorRT is not used, instead, the quantization bit-width and the range of tensor values are fed into TensorRT for speedup (i.e., with `trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS` configured).


Prerequisite
------------
When using TensorRT to speed up a quantized model, you are highly recommended to use the PyTorch docker image provided by NVIDIA.
Users can refer to `this web page <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`__ for detailed usage of the docker image.
The docker image "nvcr.io/nvidia/pytorch:22.09-py3" has been tested for the quantization speedup in NNI.

An example command to launch the docker container is `nvidia-docker run -it nvcr.io/nvidia/pytorch:22.09-py3`.
In the docker image, users should install nni>=3.0, pytorch_lightning, pycuda.

Usage
-----

"""

# %%
import torch
import torch.nn.functional as F
from torch.optim import SGD
from nni_assets.compression.mnist_model import TorchModel, device, trainer, evaluator, test_trt

config_list = [{
    'quant_types': ['input', 'weight'],
    'quant_bits': {'input': 8, 'weight': 8},
    'op_types': ['Conv2d']
}, {
    'quant_types': ['output'],
    'quant_bits': {'output': 8},
    'op_types': ['ReLU']
}, {
    'quant_types': ['input', 'weight'],
    'quant_bits': {'input': 8, 'weight': 8},
    'op_names': ['fc1', 'fc2']
}]

model = TorchModel().to(device)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = F.nll_loss
dummy_input = torch.rand(32, 1, 28, 28).to(device)

from nni.compression.pytorch.quantization import QAT_Quantizer
quantizer = QAT_Quantizer(model, config_list, optimizer, dummy_input)
quantizer.compress()

# %%
# finetuning the model by using QAT
for epoch in range(3):
    trainer(model, optimizer, criterion)
    evaluator(model)

# %%
# export model and get calibration_config
import os
os.makedirs('log', exist_ok=True)
model_path = "./log/mnist_model.pth"
calibration_path = "./log/mnist_calibration.pth"
calibration_config = quantizer.export_model(model_path, calibration_path)

print("calibration_config: ", calibration_config)

# %%
# build tensorRT engine to make a real speedup

from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT
input_shape = (32, 1, 28, 28)
engine = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=32)
engine.compress()
test_trt(engine)

# %%
# Note that NNI also supports post-training quantization directly, please refer to complete examples for detail.
#
# For complete examples please refer to :githublink:`the code <examples/model_compress/quantization/mixed_precision_speedup_mnist.py>`.
#
# For more parameters about the class 'TensorRTModelSpeedUp', you can refer to :doc:`Model Compression API Reference <../reference/compression/quantization_speedup>`.
#
# Mnist test
# ^^^^^^^^^^
#
# on one GTX2080 GPU,
# input tensor: ``torch.randn(128, 1, 28, 28)``
#
# .. list-table::
#    :header-rows: 1
#    :widths: auto
#
#    * - quantization strategy
#      - Latency
#      - accuracy
#    * - all in 32bit
#      - 0.001199961
#      - 96%
#    * - mixed precision(average bit 20.4)
#      - 0.000753688
#      - 96%
#    * - all in 8bit
#      - 0.000229869
#      - 93.7%
#
# Cifar10 resnet18 test (train one epoch)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# on one GTX2080 GPU,
# input tensor: ``torch.randn(128, 3, 32, 32)``
#
# .. list-table::
#    :header-rows: 1
#    :widths: auto
#
#    * - quantization strategy
#      - Latency
#      - accuracy
#    * - all in 32bit
#      - 0.003286268
#      - 54.21%
#    * - mixed precision(average bit 11.55)
#      - 0.001358022
#      - 54.78%
#    * - all in 8bit
#      - 0.000859139
#      - 52.81%
