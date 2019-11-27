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
    Level Pruner <Compressor/Pruner>
    AGP Pruner <Compressor/Pruner>
    L1Filter Pruner <Compressor/L1FilterPruner>
    Slim Pruner <Compressor/SlimPruner>
    Lottery Ticket Pruner <Compressor/LotteryTicketHypothesis>
    FPGM Pruner <Compressor/Pruner>
    Naive Quantizer <Compressor/Quantizer>
    QAT Quantizer <Compressor/Quantizer>
    DoReFa Quantizer <Compressor/Quantizer>
    自动模型压缩 <Compressor/AutoCompression>
