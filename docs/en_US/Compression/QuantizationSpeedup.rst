Speed up Mixed Precision Quantization Model (experimental)
==========================================================


Introduction
------------

Deep learning network has been computational intensive and memory intensive 
which increases the difficulty of deploying deep neural network model. Quantization is a 
fundamental technology which is widely used to reduce memory footprint and speed up inference 
process. Many frameworks begin to support quantization, but few of them support mixed precision 
quantization and get real speedup. Frameworks like `HAQ: Hardware-Aware Automated Quantization with Mixed Precision <https://arxiv.org/pdf/1811.08886.pdf>`__\, only support simulated mixed precision quantization which will 
not speed up the inference process. To get real speedup of mixed precision quantization and 
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
Users should set config to train quantized model using QAT algorithm(please refer to `NNI Quantization Algorithms <https://nni.readthedocs.io/en/stable/Compression/Quantizer.html>`__\  ).
After quantization aware training, users can get new config with calibration parameters and model with quantized weight. By passing new config and model to quantization speedup tool, users can get real mixed precision speedup engine to do inference.


After getting mixed precision engine, users can do inference with input data.


Note


* User can also do post-training quantization leveraging TensorRT directly(need to provide calibration dataset).
* Not all op types are supported right now. At present, NNI supports Conv, Linear, Relu and MaxPool. More op types will be supported in the following release.


Prerequisite
------------
CUDA version >= 11.0

TensorRT version >= 7.2

Usage
-----
quantization aware training:

.. code-block:: python

    # arrange bit config for QAT algorithm
    configure_list = [{
            'quant_types': ['weight', 'output'],
            'quant_bits': {'weight':8, 'output':8},
            'op_names': ['conv1']
        }, {
            'quant_types': ['output'],
            'quant_bits': {'output':8},
            'op_names': ['relu1']
        }
    ]

    quantizer = QAT_Quantizer(model, configure_list, optimizer)
    quantizer.compress()
    calibration_config = quantizer.export_model(model_path, calibration_path)

    engine = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=batch_size)
    # build tensorrt inference engine
    engine.compress()
    # data should be pytorch tensor
    output, time = engine.inference(data)


Note that NNI also supports post-training quantization directly, please refer to complete examples for detail.


For complete examples please refer to :githublink:`the code <examples/model_compress/quantization/mixed_precision_speedup_mnist.py>`.


For more parameters about the class 'TensorRTModelSpeedUp', you can refer to `Model Compression API Reference <https://nni.readthedocs.io/en/stable/Compression/CompressionReference.html#quantization-speedup>`__\.


Mnist test
^^^^^^^^^^^^^^^^^^^

on one GTX2080 GPU,
input tensor: ``torch.randn(128, 1, 28, 28)``

.. list-table::
   :header-rows: 1
   :widths: auto

   * - quantization strategy
     - Latency
     - accuracy
   * - all in 32bit
     - 0.001199961
     - 96%
   * - mixed precision(average bit 20.4)
     - 0.000753688
     - 96%
   * - all in 8bit
     - 0.000229869
     - 93.7%


Cifar10 resnet18 test(train one epoch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


on one GTX2080 GPU,
input tensor: ``torch.randn(128, 3, 32, 32)``


.. list-table::
   :header-rows: 1
   :widths: auto

   * - quantization strategy
     - Latency
     - accuracy
   * - all in 32bit
     - 0.003286268
     - 54.21%
   * - mixed precision(average bit 11.55)
     - 0.001358022
     - 54.78%
   * - all in 8bit
     - 0.000859139
     - 52.81%