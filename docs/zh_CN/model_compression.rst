#################
模型压缩
#################

Deep neural networks (DNNs) have achieved great success in many tasks. However, typical neural networks are both
computationally expensive and energy intensive, can be difficult to be deployed on devices with low computation
resources or with strict latency requirements. Therefore, a natural thought is to perform model compression to
reduce model size and accelerate model training/inference without losing performance significantly. Model compression
techniques can be divided into two categories: pruning and quantization. The pruning methods explore the redundancy
in the model weights and try to remove/prune the redundant and uncritical weights. Quantization refers to compressing
models by reducing the number of bits required to represent weights or activations.

NNI provides an easy-to-use toolkit to help user design and use model pruning and quantization algorithms.
其使用了统一的接口来支持 TensorFlow 和 PyTorch。
只需要添加几行代码即可压缩模型。
NNI 中也内置了一些流程的模型压缩算法。
用户可以进一步利用 NNI 的自动调优功能找到最佳的压缩模型，
自动模型压缩部分有详细介绍。
另一方面，用户可以使用 NNI 的接口自定义新的压缩算法。

详细信息，参考以下教程：

..  toctree::
    :maxdepth: 2

    Overview <Compression/Overview>
    Quick Start <Compression/QuickStart>
    Pruning <Compression/pruning>
    Quantization <Compression/quantization>
    Utilities <Compression/CompressionUtils>
    Framework <Compression/Framework>
    Customize Model Compression Algorithms <Compression/CustomizeCompressor>
