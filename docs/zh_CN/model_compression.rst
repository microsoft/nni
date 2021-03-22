#################
模型压缩
#################

深度神经网络（DNNs）在许多领域都取得了巨大的成功。 然而，典型的神经网络是
计算和能源密集型的，很难将其部署在计算资源匮乏
或具有严格延迟要求的设备上。 因此，一个自然的想法就是对模型进行压缩
以减小模型大小并加速模型训练/推断，同时不会显着降低模型性能。 模型压缩
技术可以分为两类：剪枝和量化。 剪枝方法探索模型权重中的冗余，
并尝试删除/修剪冗余和非关键的权重。 量化是指通过减少
权重表示或激活所需的比特数来压缩模型。

NNI 提供了易于使用的工具包来帮助用户设计并使用剪枝和量化算法。
其使用了统一的接口来支持 TensorFlow 和 PyTorch。
只需要添加几行代码即可压缩模型。
NNI 中也内置了一些流程的模型压缩算法。
用户可以进一步利用 NNI 的自动调优功能找到最佳的压缩模型，
自动模型压缩部分有详细介绍。
另一方面，用户可以使用 NNI 的接口自定义新的压缩算法。

详细信息，参考以下教程：

..  toctree::
    :maxdepth: 2

    概述 <Compression/Overview>
    快速入门 <Compression/QuickStart>
    剪枝 <Compression/pruning>
    量化 <Compression/quantization>
    工具 <Compression/CompressionUtils>
    高级用法 <Compression/advanced>
    API 参考 <Compression/CompressionReference>
