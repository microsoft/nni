#################
模型压缩
#################

NNI 提供了易于使用的工具包来帮助用户设计并使用压缩算法。
其使用了统一的接口来支持 TensorFlow 和 PyTorch。
只需要添加几行代码即可压缩模型。
NNI 中也内置了一些流程的模型压缩算法。
用户可以进一步利用 NNI 的自动调优功能找到最佳的压缩模型，
自动模型压缩部分有详细介绍。
另一方面，用户可以使用 NNI 的接口自定义新的压缩算法。

详细信息，参考以下教程：

..  toctree::
    :maxdepth: 2

    概述 <Compressor/Overview>
    快速入门 <Compressor/QuickStart>
    Pruners <Compressor/Pruner>
    Quantizers <Compressor/Quantizer>
    自动模型压缩 <Compressor/AutoCompression>
    模型加速 <Compressor/ModelSpeedup>
    模型压缩 <Compressor/CompressionUtils>
    压缩框架 <Compressor/Framework>
    自定义压缩算法 <Compressor/CustomizeCompressor>
