################
Pruning Overview
################

Pruning is a common technique to compress neural network models.
The pruning methods explore the redundancy in the model weights(parameters) and try to remove/prune the redundant and uncritical weights.
The redundant elements are pruned from the model, their values are zeroed and we make sure they don't take part in the back-propagation process.

In NNI, a pruning method is divided into multiple dimensions.

Pruning Target
--------------

Pruning target means where we apply the sparsity.
Most pruning methods prune the weight to reduce the model size and accelerate the inference latency.
Other pruning methods also apply sparsity on the input and output to accelerate the inference latency.
NNI support pruning module weight right now, and will support pruning input & output in the future.

Basic Pruners & Schedule Pruners
--------------------------------

Basic pruners generate the masks for each pruning targets (weights) for a determined sparsity ratio.
Schedule pruners decide how to allocate sparsity ratio to each pruning targets, they always work with basic pruner to generate masks.

Granularity
-----------

Fine-grained pruning or unstructured pruning refers to pruning each individual weights separately.
Coarse-grained pruning or structured pruning is pruning entire group of weights, such as a convolutional filter.

`Level Pruner <Pruner.rst#level-pruner>`__ is the only fine-grained pruner in NNI, all other pruners pruning the output channels on weights.

Pruning Mode
------------
