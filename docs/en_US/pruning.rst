#################
Pruning
#################

NNI provides several pruning algorithms that support fine-grained weight pruning and structural filter pruning.
It supports Tensorflow and PyTorch with unified interface.
For users to prune their models, they only need to add several lines in their code.
For the structural filter pruning, NNI also provides a dependency-aware mode. In the dependency-aware mode, the
filter pruner will get better speed gain after the speedup.

For details, please refer to the following tutorials:

..  toctree::
    :maxdepth: 2

    Pruners <Compressor/Pruner>
    Dependency Aware Mode <Compressor/DependencyAware>
