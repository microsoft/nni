"""
Pruning Quickstart
==================

Model pruning is a technique to reduce the model size and computation by reducing model weight size or intermediate state size.
There are three common practices for pruning a DNN model:

#. Pre-training a model -> Pruning the model -> Fine-tuning the pruned model
#. Pruning a model during training (i.e., pruning aware training) -> Fine-tuning the pruned model
#. Pruning a model -> Training the pruned model from scratch

NNI supports all of the above pruning practices by working on the key pruning stage.
Following this tutorial for a quick look at how to use NNI to prune a model in a common practice.
"""

# %%
# Preparation
# -----------
#
# In this tutorial, we use a simple model and pre-trained on MNIST dataset.
# If you are familiar with defining a model and training in pytorch, you can skip directly to `Pruning Model`_.

import torch
import torch.nn.functional as F
from torch.optim import SGD

from nni_assets.compression.mnist_model import TorchModel, trainer, evaluator, device

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
# Using L1NormPruner to prune the model and generate the masks.
# Usually, a pruner requires original model and ``config_list`` as its inputs.
# Detailed about how to write ``config_list`` please refer :doc:`compression config specification <../compression/config_list>`.
#
# The following `config_list` means all layers whose type is `Linear` or `Conv2d` will be pruned,
# except the layer named `fc3`, because `fc3` is `exclude`.
# The final sparsity ratio for each layer is 50%. The layer named `fc3` will not be pruned.

config_list = [{
    'op_types': ['Linear', 'Conv2d'],
    'exclude_op_names': ['fc3'],
    'sparse_ratio': 0.5
}]

# %%
# Pruners usually require `model` and `config_list` as input arguments.

from nni.compression.pruning import L1NormPruner
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
# Speedup the original model with masks, note that `ModelSpeedup` requires an unwrapped model.
# The model becomes smaller after speedup,
# and reaches a higher sparsity ratio because `ModelSpeedup` will propagate the masks across layers.

# need to unwrap the model, if the model is wrapped before speedup
pruner.unwrap_model()

# speedup the model, for more information about speedup, please refer :doc:`pruning_speedup`.
from nni.compression.speedup import ModelSpeedup

ModelSpeedup(model, torch.rand(3, 1, 28, 28).to(device), masks).speedup_model()

# %%
# the model will become real smaller after speedup
print(model)

# %%
# Fine-tuning Compacted Model
# ---------------------------
# Note that if the model has been sped up, you need to re-initialize a new optimizer for fine-tuning.
# Because speedup will replace the masked big layers with dense small ones.

optimizer = SGD(model.parameters(), 1e-2)
for epoch in range(3):
    trainer(model, optimizer, criterion)
