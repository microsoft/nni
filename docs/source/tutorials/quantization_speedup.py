"""
SpeedUp Model with Calibration Config
======================================


Introduction
------------

Deep learning network has been computational intensive and memory intensive 
which increases the difficulty of deploying deep neural network model. Quantization is a 
fundamental technology which is widely used to reduce memory footprint and speedup inference 
process. Many frameworks begin to support quantization, but few of them support mixed precision 
quantization and get real speedup. Frameworks like `HAQ: Hardware-Aware Automated Quantization with Mixed Precision <https://arxiv.org/pdf/1811.08886.pdf>`__\, only support simulated mixed precision quantization which will 
not speedup the inference process. To get real speedup of mixed precision quantization and 
help people get the real feedback from hardware, we design a general framework with simple interface to allow NNI quantization algorithms to connect different 
DL model optimization backends (e.g., TensorRT, NNFusion), which gives users an end-to-end experience that after quantizing their model 
with quantization algorithms, the quantized model can be directly speeded up with the connected optimization backend. NNI connects 
TensorRT at this stage, and will support more backends in the future.


Design and Implementation
-------------------------

To support speeding up mixed precision quantization, we divide framework into two part, frontend and backend.  
Frontend could be popular training frameworks such as PyTorch, TensorFlow etc. Backend could be inference 
framework for different hardwares, such as TensorRT. At present, we support PyTorch as frontend and 
TensorRT as backend. To convert PyTorch model to TensorRT engine, we leverage onnx as intermediate graph 
representation. In this way, we convert PyTorch model to onnx model, then TensorRT parse onnx 
model to generate inference engine. 


Quantization aware training combines NNI quantization algorithm 'QAT' and NNI quantization speedup tool.
Users should set config to train quantized model using QAT algorithm(please refer to :doc:`NNI Quantization Algorithms <../compression/quantizer>`  ).
After quantization aware training, users can get new config with calibration parameters and model with quantized weight. By passing new config and model to quantization speedup tool, users can get real mixed precision speedup engine to do inference.


After getting mixed precision engine, users can do inference with input data.


Note


* Recommend using "cpu"(host) as data device(for both inference data and calibration data) since data should be on host initially and it will be transposed to device before inference. If data type is not "cpu"(host), this tool will transpose it to "cpu" which may increases unnecessary overhead.
* User can also do post-training quantization leveraging TensorRT directly(need to provide calibration dataset).
* Not all op types are supported right now. At present, NNI supports Conv, Linear, Relu and MaxPool. More op types will be supported in the following release.


Prerequisite
------------
CUDA version >= 11.0

TensorRT version >= 7.2

Note

* If you haven't installed TensorRT before or use the old version, please refer to `TensorRT Installation Guide <https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html>`__\  

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

from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
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
