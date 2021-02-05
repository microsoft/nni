支持的量化算法
========================================

支持的量化算法列表


* `Naive Quantizer <#naive-quantizer>`__
* `QAT Quantizer <#qat-quantizer>`__
* `DoReFa Quantizer <#dorefa-quantizer>`__
* `BNN Quantizer <#bnn-quantizer>`__

Naive Quantizer
---------------

Naive Quantizer 将 Quantizer 权重默认设置为 8 位，可用它来测试量化算法。

用法
^^^^^

PyTorch

.. code-block:: python

   model = nni.algorithms.compression.pytorch.quantization.NaiveQuantizer(model).compress()

----

QAT Quantizer
-------------

在 `Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference <http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf>`__ 中，作者 Benoit Jacob 和 Skirmantas Kligys 提出了一种算法在训练中量化模型。

..

   我们提出了一种方法，在训练的前向过程中模拟量化效果。 此方法不影响反向传播，所有权重和偏差都使用了浮点数保存，因此能很容易的进行量化。 然后，前向传播通过实现浮点算法的舍入操作，来在推理引擎中模拟量化的推理。


   * 权重在与输入卷积操作前进行量化。 如果在层中使用了批量归一化（参考 [17]），批量归一化参数会被在量化前被“折叠”到权重中。
   * 激活操作在推理时会被量化。 例如，在激活函数被应用到卷积或全连接层输出之后，或在增加旁路连接，或连接多个层的输出之后（如：ResNet）。


用法
^^^^^

可在训练代码前将模型量化为 8 位。

PyTorch 代码

.. code-block:: python

   from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
   model = Mnist()

   config_list = [{
       'quant_types': ['weight'],
       'quant_bits': {
           'weight': 8,
       }, # 这里可以仅使用 `int`，因为所有 `quan_types` 使用了一样的位长，参考下方 `ReLu6` 配置。
       'op_types':['Conv2d', 'Linear']
   }, {
       'quant_types': ['output'],
       'quant_bits': 8,
       'quant_start_step': 7000,
       'op_types':['ReLU6']
   }]
   quantizer = QAT_Quantizer(model, config_list)
   quantizer.compress()

查看示例进一步了解

QAT Quantizer 的用户配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

压缩算法的公共配置可在 `config_list 说明 <./QuickStart.rst>`__ 中找到。

此算法所需的配置：


* **quant_start_step:** int

在运行到某步骤前，对模型禁用量化。这让网络在进入更稳定的
状态后再激活量化，这样不会配除掉一些分数显著的值，默认为 0。

注意
^^^^

当前不支持批处理规范化折叠。

----

DoReFa Quantizer
----------------

在 `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`__ 中，作者 Shuchang Zhou 和 Yuxin Wu 提出了 DoReFa 算法在训练时量化权重，激活函数和梯度。

用法
^^^^^

要实现 DoReFa Quantizer，在训练代码前加入以下代码。

PyTorch 代码

.. code-block:: python

   from nni.algorithms.compression.pytorch.quantization import DoReFaQuantizer
   config_list = [{ 
       'quant_types': ['weight'],
       'quant_bits': 8, 
       'op_types': 'default' 
   }]
   quantizer = DoReFaQuantizer(model, config_list)
   quantizer.compress()

查看示例进一步了解

DoReFa Quantizer 的用户配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

压缩算法的公共配置可在 `config_list 说明 <./QuickStart.rst>`__ 中找到。

此算法所需的配置：

----

BNN Quantizer
-------------

在 `Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1 <https://arxiv.org/abs/1602.02830>`__ 中 

..

   引入了一种训练二进制神经网络（BNN）的方法 - 神经网络在运行时使用二进制权重。 在训练时，二进制权重和激活用于计算参数梯度。 在 forward 过程中，BNN 会大大减少内存大小和访问，并将大多数算术运算替换为按位计算，可显著提高能源效率。


用法
^^^^^

PyTorch 代码

.. code-block:: python

   from nni.algorithms.compression.pytorch.quantization import BNNQuantizer
   model = VGG_Cifar10(num_classes=10)

   configure_list = [{
       'quant_bits': 1,
       'quant_types': ['weight'],
       'op_types': ['Conv2d', 'Linear'],
       'op_names': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'classifier.0', 'classifier.3']
   }, {
       'quant_bits': 1,
       'quant_types': ['output'],
       'op_types': ['Hardtanh'],
       'op_names': ['features.6', 'features.9', 'features.13', 'features.16', 'features.20', 'classifier.2', 'classifier.5']
   }]

   quantizer = BNNQuantizer(model, configure_list)
   model = quantizer.compress()

可以查看 :githublink:`示例 <examples/model_compress/BNN_quantizer_cifar10.py>` 了解更多信息。

BNN Quantizer 的用户配置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

压缩算法的公共配置可在 `config_list 说明 <./QuickStart.rst>`__ 中找到。

此算法所需的配置：

实验
^^^^^^^^^^

我们实现了 `Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1 <https://arxiv.org/abs/1602.02830>`__ 中的一个实验，对 CIFAR-10 上的 **VGGNet** 进行了量化操作。 我们的实验结果如下：

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 模型
     - 准确率
   * - VGGNet
     - 86.93%


实验代码在 :githublink:`examples/model_compress/BNN_quantizer_cifar10.py <examples/model_compress/BNN_quantizer_cifar10.py>` 
