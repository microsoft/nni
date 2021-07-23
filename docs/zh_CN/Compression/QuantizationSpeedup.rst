加速混合精度量化模型（实验性）
==========================================================


介绍
------------

深度神经网络一直是计算密集型和内存密集型的， 
这增加了部署深度神经网络模型的难度。 量化是 
一种广泛用于减少内存占用和加速推理过程的基础技术 
。 许多框架开始支持量化，但很少有框架支持混合精度 
量化并取得真正的速度提升。 像 `HAQ: Hardware-Aware Automated Quantization with Mixed Precision <https://arxiv.org/pdf/1811.08886.pdf>`__\ 这样的框架只支持模拟混合精度量化，这将 
不加快推理过程。 为了获得混合精度量化的真正加速和 
帮助人们从硬件中获得真实的反馈，我们设计了一个具有简单接口的通用框架，允许 NNI 量化算法连接不同的 
DL 模型优化后端（例如 TensorRT、NNFusion），使用量化算法，在量化模型后为用户提供端到端体验 
量化模型可以直接通过连接的优化后端加速。 在这个阶段，NNI 连接了 
TensorRT，并将在未来支持更多的后端。


设计和实现
-------------------------

为了支持加速混合精度量化，我们将框架划分为两个部分，前端和后端。  
前端可以是流行的训练框架，如 PyTorch、TensorFlow 等。 后端可以是 
为不同硬件设计的推理框架，如 TensorRT。 目前，我们支持 PyTorch 作为前端和 
TensorRT 作为后端。 为了将 PyTorch 模型转换为 TensorRT 引擎，我们利用 onnx 作为中间图 
表示。 通过这种方式，我们将 PyTorch 模型转换为 onnx 模型，然后 TensorRT 解析 onnx 
模型生成推理引擎。 


量化感知训练结合了 NNI 量化算法 'QAT' 和 NNI 量化加速工具。
用户应该设置配置，使用 QAT 算法训练量化模型（请参考 `NNI量化算法 <https://nni.readthedocs.io/zh/stable/Compression/Quantizer.html>`__）。
经过量化感知训练，用户可以得到带有校准参数的新配置和带有量化权重的模型。 通过将新的配置和模型传递给量化加速工具，用户可以得到真正的混合精度加速引擎来进行推理。


在得到混合精度引擎后，用户可以使用输入数据进行推理。


注意


* 用户也可以直接利用 TensorRT 进行训练后的量化处理（需要提供校准数据集）。
* 并非所有OP类型都已支持。 目前，NNI 支持 Conv, Linear, Relu 和 MaxPool。 未来版本中将支持更多操作类型。


先决条件
------------
CUDA version >= 11.0

TensorRT version >= 7.2

用法
-----
量化感知训练：

.. code-block:: python

    # 为QAT算法设置比特配置
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
    # 建立 Tensorrt 推理引擎
    engine.compress()
    # 数据应该是 Pytorch Tensor
    output, time = engine.inference(data)


请注意，NNI还直接支持后训练量化，请参阅完整的示例以获取详细信息。


完整的例子请参考 :githublink:`这里 <examples/model_compress/quantization/mixed_precision_speedup_mnist.py>`。


关于 'TensorRTModelSpeedUp' 类的更多参数，你可以参考 `Model Compression API Reference <https://nni.readthedocs.io/zh/stable/Compression/CompressionReference.html#quantization-speedup>`__ 。


Mnist 测试
^^^^^^^^^^^^^^^^^^^

在一块 GTX2080 GPU 上
输入张量：``torch.randn(128, 1, 28, 28)``

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 量化策略
     - 延迟
     - 准确率
   * - 均为 32bit
     - 0.001199961
     - 96%
   * - 混合精度（平均 bit 20.4）
     - 0.000753688
     - 96%
   * - 均为 8bit
     - 0.000229869
     - 93.7%


Cifar10 resnet18 测试（训练一个 epoch）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


在一块 GTX2080 GPU 上
输入张量: ``torch.randn(128, 3, 32, 32)``


.. list-table::
   :header-rows: 1
   :widths: auto

   * - 量化策略
     - 延迟
     - 准确率
   * - 均为 32bit
     - 0.003286268
     - 54.21%
   * - 混合精度（平均 bit 11.55）
     - 0.001358022
     - 54.78%
   * - 均为 8bit
     - 0.000859139
     - 52.81%