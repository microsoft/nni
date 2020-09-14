#################
Pruning
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

    Pruners <Compressor/Pruner>
    Dependency Aware Mode <Compressor/DependencyAware>
