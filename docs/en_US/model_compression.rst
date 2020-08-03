#################
Model Compression
#################

NNI provides an easy-to-use toolkit to help user design and use compression algorithms.
It supports Tensorflow and PyTorch with unified interface.
For users to compress their models, they only need to add several lines in their code.
There are some popular model compression algorithms built-in in NNI.
Users could further use NNI's auto tuning power to find the best compressed model,
which is detailed in Auto Model Compression.
On the other hand, users could easily customize their new compression algorithms using NNI's interface.

For details, please refer to the following tutorials:

..  toctree::
    :maxdepth: 2

    Overview <Compressor/Overview>
    Quick Start <Compressor/QuickStart>
    Pruners <Compressor/Pruner>
    Quantizers <Compressor/Quantizer>
    Automatic Model Compression <Compressor/AutoCompression>
    Model Speedup <Compressor/ModelSpeedup>
    Compression Utilities <Compressor/CompressionUtils>
    Compression Benchmark <Compressor/Benchmark>
    Compression Framework <Compressor/Framework>
    Customize Compression Algorithms <Compressor/CustomizeCompressor>
