.. b6bdf52910e2e2c72085d03482d45340

模型压缩
========

深度神经网络（DNNs）在计算机视觉、自然语言处理、语音处理等领域取得了巨大的成功。   
然而，典型的神经网络是计算和能源密集型的，很难将其部署在计算资源匮乏
或具有严格延迟要求的设备上。 因此，一个自然的想法就是对模型进行压缩，
以减小模型大小并加速模型训练/推断，同时不会显着降低模型性能。 
模型压缩技术可以分为两类：剪枝和量化。 剪枝方法探索模型权重中的冗余，
并尝试删除/修剪冗余和非关键的权重。 量化是指通过减少权重表示或激活所需的比特数来压缩模型。
在接下来的章节中，我们将进一步阐述这两种方法: 剪枝和量化。 
此外，下图直观地展示了这两种方法的区别。  

.. image:: ../../img/prune_quant.jpg
   :target: ../../img/prune_quant.jpg
   :scale: 40%
   :alt:

NNI 提供了易于使用的工具包来帮助用户设计并使用剪枝和量化算法。
其使用了统一的接口来支持 TensorFlow 和 PyTorch。
对用户来说， 只需要添加几行代码即可压缩模型。
NNI 中也内置了一些主流的模型压缩算法。
用户可以进一步利用 NNI 的自动调优功能找到最佳的压缩模型，
该功能在自动模型压缩部分有详细介绍。
另一方面，用户可以使用 NNI 的接口自定义新的压缩算法。


NNI 具备以下几个核心特性:
* 内置许多流行的剪枝和量化算法。
* 利用最先进的策略和NNI的自动调整能力，来自动化模型剪枝和量化过程。
* 加速模型，使其有更低的推理延迟。
* 提供友好和易于使用的压缩工具，让用户深入到压缩过程和结果。
* 简洁的界面，供用户自定义自己的压缩算法。

压缩流程
---------

.. image:: ../../img/compression_pipeline.png
   :target: ../../img/compression_pipeline.png
   :alt:
   :align: center
   :scale: 30%

NNI中模型压缩的整体流程如上图所示。
为了压缩一个预先训练好的模型，可以单独或联合使用修剪和量化。
如果用户希望同时应用这两种模式，建议采用串行模式。


.. note::
  值得注意的是，NNI的pruner或quantizer并不能改变网络结构，只能模拟压缩的效果。
  真正能够压缩模型、改变网络结构、降低推理延迟的是NNI的加速工具。
  为了获得一个真正的压缩的模型，用户需要执行 :doc:`剪枝加速 <../tutorials/pruning_speedup>` or :doc:`量化加速 <../tutorials/quantization_speedup>`. 
  PyTorch和TensorFlow的接口都是统一的。目前只支持PyTorch版本，未来将支持TensorFlow版本。


模型加速
---------

模型压缩的最终目标是减少推理延迟和模型大小。
然而，现有的模型压缩算法主要是通过仿真来检测压缩模型的性能。
例如，修剪算法使用掩码，量化算法仍将值存储在float32中。
如果能给定这些算法产生的输出掩码和量化位，NNI的加速工具就可以真正地压缩模型。

下图显示了NNI如何修剪和加速您的模型。

.. image:: ../../img/nni_prune_process.png
   :target: ../../img/nni_prune_process.png
   :scale: 30%
   :align: center
   :alt:

关于用掩码进行模型加速的详细文档可以参考 :doc:`here <../tutorials/pruning_speedup>`.
关于用校准配置进行模型加速的详细文档可以参考 :doc:`here <../tutorials/quantization_speedup>`.


.. attention::

  NNI的模型剪枝框架已经升级到更高级的版本 (在 nni 2.6 版本前称为pruning v2)。
  旧版本 (`named pruning before nni v2.6 <https://nni.readthedocs.io/en/v2.6/Compression/pruning.html>`_) 不再进行维护. 
  如果出于某些原因您不得不使用，v2.6 是最后的支持旧版剪枝算法的版本。
