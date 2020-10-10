#################
Pruning
#################

Pruning is a common technique to compress neural network models.
The pruning methods explore the redundancy in the model weights(parameters) and try to remove/prune the redundant and uncritical weights.
The redundant elements are pruned from the model, their values are zeroed and we make sure they don't take part in the back-propagation process.

From pruning granularity perspective, fine-grained pruning or unstructured pruning refers to pruning each individual weights separately.
Coarse-grained pruning or structured pruning is pruning entire group of weights, such as a convolutional filter.

NNI provides multiple unstructured pruning and structured pruning algorithms.
It supports Tensorflow and PyTorch with unified interface.
For users to prune their models, they only need to add several lines in their code.
For the structured filter pruning, NNI also provides a dependency-aware mode. In the dependency-aware mode, the
filter pruner will get better speed gain after the speedup.

For details, please refer to the following tutorials:

..  toctree::
    :maxdepth: 2

    Pruners <Pruner>
    Dependency Aware Mode <DependencyAware>
    Model Speedup <ModelSpeedup>
    Automatic Model Pruning with NNI Tuners <AutoPruningUsingTuners>
