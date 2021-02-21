Speed up Mixed Precision Quantization Model (experimental)
==========================================================


Introduction
------------

Deep learning network has been computational intensive and memory intensive 
which increase the difficulty of deploying deep neural network model. Quantization is a 
fundamental technology which is widely used to reduce memory footprint and speed up inference 
process. Many frameworks begin to support quantization, but few of them support mixed precision 
quantiation. Frameworks like haq(https://arxiv.org/pdf/1811.08886.pdf) only support simulated mixed precision quantization which will 
not speed up the inference process. To get real speedup of mixed precision quantization and 
help people get the real feedback from hardware, we design a general framework with simple interface to allow NNI to connect different 
DL model optimization backends (e.g., TensorRT, NNFusion), which gives users an end-to-end experience that after quantizing their model 
with quantization algorithms, the quantized model can be directly speeded up with the connected optimization backend. NNI connects 
TensorRT at this stage, and will support more backends in the future.


Design and Implementation
-------------------------

To support speeding up mixed precision quantization, we divide framework into two part, frontend and backend.  
Frontend could be popular training frameworks such as PyTorch, TensorFlow etc. Backend could be inference 
framework for different hardware, such as TensorRT At present, we support PyTorch as frontend and 
TensorRT as backend. To convert PyTorch model to TensorRT engine, we leverage onnx as intermediate graph 
representation. In this way, we convert PyTorch model to onnx model, then TensorRT parse onnx 
model to generate inference engine. 


For the purpose of getting mixed precision engine, we need to provide a PyTorch model, calibrate dataset 
and corresponded bit config. The calibrate set is used to calibrate quantized module. What' s more, the 
bit config a dictionary whose key is layer name and value is bit width. After getting mixed precision engine, 
we can do inference with given input data.

Prerequisite
------------
CUDA version >= 11.0

TensorRT version >= 7.2

Usage
-----

.. code-block:: python

    # arrange bit config for quantized layer
    config = {
        'conv1':8,
        'conv2':32,
        'fc1':16,
        'fc2':8,
    }

    engine = TensorRTModelSpeedUp(model, onnx_path, input_shape, config=config, extra_layer_bit=32, 
        strict_datatype=True, using_calibrate=True, calibrate_type=CalibrateType.ENTROPY2, calib_data=test_set, 
        calibration_cache = calibration_cache, batchsize=batch_size, input_names=input_names, output_names=output_names)
    # build tensorrt inference engine
    engine.build()
    # test_set should be numpy datatype
    output, time = engine.inference(test_set)

For complete examples please refer to :githublink:`the code <examples/model_compress/quantization/mixed_precision_speedup_mnist.py>`.
For more parameters about the class 'TensorRTModelSpeedUp', you can refer to :githublink:`the code <nni/compression/pytorch/speedup/quantization_speedup/integrated_tensorrt.py>`.

Mnist Example
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


Cifar10 resnet18 example(train one epoch)
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