#################
Pruning
#################

A common technique to increase sparsity in neural network model weights and activations is pruning. The pruning
methods explore the redundancy in the model weights(parameters) and try to remove/prune the redundant and uncritical
weights. The redundant elements are pruned from the model, their values are zeroed and we make sure they don't
take part in the back-propagation process.

NNI provides several pruning algorithms that support fine-grained weight pruning and structural filter pruning.
It supports Tensorflow and PyTorch with unified interface.
For users to prune their models, they only need to add several lines in their code.
For the structural filter pruning, NNI also provides a dependency-aware mode. In the dependency-aware mode, the
filter pruner will get better speed gain after the speedup.

For details, please refer to the following tutorials:

..  toctree::
    :maxdepth: 2

    Pruners <Pruner>
    Dependency Aware Mode <DependencyAware>
    Model Speedup <ModelSpeedup>
