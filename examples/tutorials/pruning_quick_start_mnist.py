"""
Pruning Quickstart
==================

Model pruning is a technique to reduce the model size and computation by reducing model weight size or intermediate state size.
It usually has following paths:

#. Pre-training a model -> Pruning the model -> Fine-tuning the model
#. Pruning the model aware training -> Fine-tuning the model
#. Pruning the model -> Pre-training the compact model

NNI supports the above three modes and mainly focuses on the pruning stage.
Follow this tutorial for a quick look at how to use NNI to prune a model in a common practice.
"""

# %%
# Preparation
# -----------
#
# In this tutorial, we use a simple model and pre-train on MNIST dataset.
# If you are familiar with defining a model and training in pytorch, you can skip directly to `Pruning Model`_.

import torch
import torch.nn.functional as F
from torch.optim import SGD

from scripts.compression_mnist_model import TorchModel, trainer, evaluator, device

# define the model
model = TorchModel().to(device)

# show the model structure, note that pruner will wrap the model layer.
print(model)

# %%

# define the optimizer and criterion for pre-training

optimizer = SGD(model.parameters(), 1e-2)
criterion = F.nll_loss

# pre-train and evaluate the model on MNIST dataset
for epoch in range(3):
    trainer(model, optimizer, criterion)
    evaluator(model)

# %%
# Pruning Model
# -------------
#
# Using L1NormPruner pruning the model and generating the masks.
# Usually, pruners require original model and ``config_list`` as parameters.
# Detailed about how to write ``config_list`` please refer :doc:`compression config specification <../compression/compression_config_list>`.
#
# This `config_list` means all layers whose type is `Linear` or `Conv2d` will be pruned,
# except the layer named `fc3`, because `fc3` is `exclude`.
# The final sparsity ratio for each layer is 50%. The layer named `fc3` will not be pruned.

config_list = [{
    'sparsity_per_layer': 0.5,
    'op_types': ['Linear', 'Conv2d']
}, {
    'exclude': True,
    'op_names': ['fc3']
}]

# %%
# Pruners usually require `model` and `config_list` as input arguments.

from nni.algorithms.compression.v2.pytorch.pruning import L1NormPruner
pruner = L1NormPruner(model, config_list)

# show the wrapped model structure, `PrunerModuleWrapper` have wrapped the layers that configured in the config_list.
print(model)

# %%

# compress the model and generate the masks
_, masks = pruner.compress()
# show the masks sparsity
for name, mask in masks.items():
    print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

# %%
# Speed up the original model with masks, note that `ModelSpeedup` requires an unwrapped model.
# The model becomes smaller after speed-up,
# and reaches a higher sparsity ratio because `ModelSpeedup` will propagate the masks across layers.

# need to unwrap the model, if the model is wrapped before speed up
pruner._unwrap_model()

# speed up the model
from nni.compression.pytorch.speedup import ModelSpeedup

ModelSpeedup(model, torch.rand(3, 1, 28, 28).to(device), masks).speedup_model()

# %%
# the model will become real smaller after speed up
print(model)

# %%
# Fine-tuning Compacted Model
# ---------------------------
# Note that if the model has been sped up, you need to re-initialize a new optimizer for fine-tuning.
# Because speed up will replace the masked big layers with dense small ones.

optimizer = SGD(model.parameters(), 1e-2)
for epoch in range(3):
    trainer(model, optimizer, criterion)
