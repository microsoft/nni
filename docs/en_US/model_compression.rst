#################
Model Compression
#################

Deep neural networks (DNNs) have achieved great success in many tasks. However, typical neural networks are both
computationally expensive and energy intensive, can be difficult to be deployed on devices with low computation
resources or with strict latency requirements. Therefore, a natural thought is to perform model compression to
reduce model size and accelerate model training/inference without losing performance significantly. Model compression
techniques can be divided into two categories: pruning and quantization. The pruning methods explore the redundancy
in the model weights and try to remove/prune the redundant and uncritical weights. Quantization refers to compressing
models by reducing the number of bits required to represent weights or activations.

NNI provides an easy-to-use toolkit to help user design and use model pruning and quantization algorithms.
It supports Tensorflow and PyTorch with unified interface.
For users to compress their models, they only need to add several lines in their code.
There are some popular model compression algorithms built-in in NNI.
Users could further use NNI's auto tuning power to find the best compressed model,
which is detailed in Auto Model Compression.
On the other hand, users could easily customize their new compression algorithms using NNI's interface.

For details, please refer to the following tutorials:

..  toctree::
    :maxdepth: 2

    Overview <Compression/Overview>
    Quick Start <Compression/QuickStart>
    Pruning <Compression/pruning>
    Quantization <Compression/quantization>
    Utilities <Compression/CompressionUtils>
    Framework <Compression/Framework>
    Customize Model Compression Algorithms <Compression/CustomizeCompressor>
