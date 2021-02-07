Speed up Mixed Precision Quantization Model
===========================================

*This feature is in Beta version.*

Introduction
------------

Deep learning network has been computationly intensive and memory intensive 
which increase the difficulty of deploying deep neural network model. Quantization is a 
fundamental technology which is widely used to reduce memory footprint and speed up inference 
process. Many frameworks begin to support quantization, but few of them support mixed precision 
quantiation. Frameworks like haq only support simulated mixed precision quantization which will 
not speed up the inference process. To get real speed up of mixed precision quantization and 
help people get the real feedback from hardware, we design a tool which combine nni and tensorrt together.


Design and Implementation
-------------------------

To support speeding up mixed precision quantization, we divide framework into two part, frontend and backend.  
Frontend could be popular raining framework such as pytorch, tensorflow etc. Backend could be inference 
framework for different hardware, such as tensorrt. At present, we support pytorch as frontend and 
tensorrt as backend. To convert pytorch model to tensorrt engine, we leverage onnx as intermediate graph 
representation. In this way, we convert pytorch model to onnx model, then tensorrt parse onnx 
model to generate inference engine. 


For the purpose of getting mixed precision engine, we need to provide a pytorch model, calibrate dataset 
and corresponded bit config. The calibrate set is used to calibrate quantized module. What' s more, the 
bit config a dictionary whose key is layer name and value is bit width. After getting mixed precision engine, 
we can do inference with given input data.

Prerequisite
------------
CUDA version >= 11.0

TensorRt version >= 7.2.1.6

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

    """
    Parameters
    ----------
    model : pytorch model
        The model to speed up by quantization.
    onnx_path : str
        The path where user want to store onnx model.
    input_shape : tuple
        The input shape of model, shall pass it to torch.onnx.export.
    config : dict
        Config recording bit number and name of layers.
    extra_layer_bit : int
        Other layers which are not in config will be quantized to corresponding bit number.
    strict_datatype : bool
        Whether constraining layer bit to the number given in config or not. If true, all the layers 
        will be set to given bit strictly. Otherwise, these layers will be set automatically by
        tensorrt.
    using_calibrate : bool
        Whether calibrating during quantization or not. If true, user should provide calibration
        dataset. If not, user should provide scale and zero_point for each layer. Current version
        only supports using calibrating.
    calibrate_type : tensorrt.tensorrt.CalibrationAlgoType
        The algorithm of calibrating. Please refer to https://docs.nvidia.com/deeplearning/
        tensorrt/api/python_api/infer/Int8/Calibrator.html for detail
    calibrate_data : numpy array
        The data using to calibrate quantization model
    calibration_cache : str
        The path user want to store calibrate cache file
    batchsize : int
        The batch size of calibration and inference
    input_names : list
        Input name of onnx model providing for torch.onnx.export to generate onnx model
    output_name : list
        Output name of onnx model providing for torch.onnx.export to generate onnx model
    """
    engine = TensorRTModelSpeedUp(model, onnx_path, input_shape, config=config, extra_layer_bit=32, 
        strict_datatype=True, using_calibrate=True, calibrate_type=CalibrateType.ENTROPY2, calib_data=test_set, 
        calibration_cache = calibration_cache, batchsize=batch_size, input_names=input_names, output_names=output_names)
    # build tensorrt inference engine
    engine.build()
    # test_set should be numpy datatype
    output, time = engine.inference(test_set)

For complete examples please refer to :githublink:`the code <examples/model_compress/quantization/mixed_precision_speedup_mnist.py>`

Mnist Lenet Example
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