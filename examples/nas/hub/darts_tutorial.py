"""
Searching on DARTS search space
===============================

In this tutorial, we demonstrate how to search on the famous model space proposed in `DARTS <https://arxiv.org/abs/1806.09055>`__.

Through this process, you will learn:

* How to use the built-in model spaces from NNI's model space hub.
* How to use one-shot exploration strategies to explore a model space.
* How to customize evaluators to achieve the best performance.

In the end, we get a strong-performing model on CIFAR-10 dataset, which achieves xx.xx% accuracy.

.. attention::

   Running this tutorial requires a GPU.
   If you don't have one, you can set ``gpus`` in :class:`~nni.retiarii.evaluator.pytorch.Classification` to be 0,
   but do note that it will be much slower.

Use the model space
-------------------

The model space provided in DARTS originated from `NASNet <https://arxiv.org/abs/1707.07012>`__,
where the full model is constructed by repeatedly stacking a single computational unit (called a **cell**).
There are two types of cells within a network. The first type is called *normal cell*, and the second type is called *reduction cell*.
The key difference between normal and reduction cell is that the reduction cell will downsample the input feature map,
and decrease its resolution. Normal and reduction cells are stacked alternately, as shown in the following figure.

.. image:: ../../img/nasnet_cell_stack.png

A cell takes outputs from two previous cells as inputs and contains a collection of *nodes*.
Each node takes two previous nodes within the same cell (or the two cell inputs),
and applies an *operator* (e.g., convolution, or max-pooling) to each input,
and sums the outputs of operators as the output of the node.
The output of cell is the concatenation of all the nodes that are never used as inputs of another node.
We recommend reading `NDS <https://arxiv.org/pdf/1905.13214.pdf>`__ or `ENAS <https://arxiv.org/abs/1802.03268>`__ for details.

We illustrate an example of cells in the following figure.

.. image:: ../../img/nasnet_cell.png

The search space proposed in DARTS paper introduced two modifications to the original space in `NASNet <https://arxiv.org/abs/1707.07012>`__.

Firstly, the operator candidates have been narrowed down to seven:

- Max pooling 3x3
- Average pooling 3x3
- Skip connect (Identity)
- Separable convolution 3x3
- Separable convolution 5x5
- Dilated convolution 3x3
- Dilated convolution 5x5

Secondly, the output of cell is the concatenate of **all the nodes within the cell**.

As the search space is based on cell, once the normal and reduction cell has been fixed, we can stack them for indefinite times.
To save the search cost, the common practice is to reduce the number of filters (i.e., channels) and number of stacked cells
during the search phase, and increase them back when training the final searched architecture.
"""

# %%
# In the following example, we initialize a DARTS model space, with only 16 initial filters and 8 stacked cells.
# The network is specialized for CIFAR-10 dataset with 32x32 input resolution.
#
# The DARTS model space here is provided by :doc:`model space hub <./space_hub>`,
# where we have supported multiple popular model spaces for plug-and-play.
#
# .. note::
#
#    Since we are going to search on **model space** provided by DARTS with **search strategy** proposed by DARTS.
#    To avoid confusion, we refer to them as *DARTS model space* and *DARTS strategy* respectively.

from nni.retiarii.hub.pytorch import DARTS

model_space = DARTS(16, 8, 'cifar')

# %%
#
# Search on the model space
# -------------------------
#
# To begin exploring the model space, one firstly need to have an evaluator to provide the criterion of a "good model".
# As we are searching on CIFAR-10 dataset, one can easily use the :class:`~nni.retiarii.evaluator.pytorch.Classification`
# as a starting point.
#
# Note that for a typical setup of NAS, the model search should be on validation set, and the evaluation of the final searched model
# should be on test set. However, as CIFAR-10 dataset only has a training set of 50k images and a validation set (10k images),
# we have to split the original training set into a training set and a validation set.
# As we are going to use the provided by DARTS paper, the recommended train/val split is 1:1.

import nni
import numpy as np
from nni.retiarii.evaluator.pytorch import (
    Classification,
    DataLoader  # might also use torch.utils.data.DataLoader if not using multi-trial strategy
)
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# If you are to use multi-trial strategies, please wrap CIFAR10 with :func:`nni.trace`.
train_data = nni.trace(CIFAR10)(root='./data', train=True, download=True, transform=transform)

num_samples = len(train_data)
indices = np.random.permutation(num_samples)
split = num_samples // 2

train_loader = DataLoader(
    train_data, batch_size=64, num_workers=6,
    sampler=SubsetRandomSampler(indices[:split]),
)

valid_loader = DataLoader(
    train_data, batch_size=64, num_workers=6,
    sampler=SubsetRandomSampler(indices[split:]),
)

# %%
#
# .. warning:: Max epochs is set to 1 here for tutorial purposes. To get a reasonable result, this should be at least 10.

max_epochs = 1

evaluator = Classification(
    learning_rate=0.01,
    weight_decay=1e-4,
    train_dataloaders=train_loader,
    val_dataloaders=valid_loader,
    max_epochs=max_epochs，
    gpus=1,
)

# %%
#
# We will use DARTS (Differentiable ARchiTecture Search) as the search strategy to explore the model space.
# DARTS strategy belongs to the category of :ref:`one-shot strategy <one-shot-nas>`.
# The fundamental differences between One-shot strategies and :ref:`multi-trial strategies <multi-trial-nas>` is that,
# one-shot strategy combines search with model training into a single run.
# Compared to multi-trial strategies, one-shot NAS doesn't need to iteratively spawn new trials (i.e., models),
# and thus saves the excessive cost of model training.
# It's worth mentioning that one-shot NAS also suffers from multiple drawbacks despite its computational efficiency.
# We recommend
# `Weight-Sharing Neural Architecture Search: A Battle to Shrink the Optimization Gap <https://arxiv.org/abs/2008.01475>`__
# and
# `How Does Supernet Help in Neural Architecture Search? <https://arxiv.org/abs/2010.08219>`__ for interested readers.
#
# If you want to know how DARTS strategy works, here is a brief version.
# Under the hood, DARTS converts the cell into a densely connected graph, and put operators on edges (see the following figure).
# Since the operators are not decided yet, every edge is a weighted mixture of multiple operators (multiple color in the figure).
# DARTS then learns to assign the optimal "color" for each edge during the network training.
# It finally selects one "color" for each edge, and drops redundant edges.
# The weights on the edges are called *architecture weights*.
#
# .. image:: ../../img/darts_illustration.png
#
# Note that for DARTS model space, exactly two inputs are kept for every node. This fact is not reflected in the figure.
#
# :class:`~nni.retiarii.strategy.DARTS` strategy is provided as one of NNI's :doc:`built-in search strategies </nas/exploration_strategy>`.
# Using it can be as simple as one line of code.

from nni.retiarii.strategy import DARTS as DartsStrategy

strategy = DartsStrategy()

# %%
#
# Launching the experiment is similar to what we have done in the :doc:`beginner tutorial <hello_nas>`,
# except that the ``execution_engine`` argument should be set to ``oneshot``.

from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

config = RetiariiExeConfig(execution_engine='oneshot')
experiment = RetiariiExperiment(model_space, evaluator=evaluator, strategy=strategy)
experiment.run(config)

# %%
#
# We can then retrieve the best model found by the strategy with ``export_top_models``.
# Here, the retrieved model is a dict (called *architecture dict*) describing the selected normal cell and reduction cell.

exported_arch = experiment.export_top_models()[0]

exported_arch

# %%
#
# Retrain the searched model
# --------------------------
#
# What we have got in the last step, is only a cell structure.
# To get a final usable model with trained weights, we need to construct a real model based on this structure,
# and then fully train it.
#
# To construct a fixed model based on the architecture dict exported from the experiment,
# we can use :func:`nni.retiarii.fixed_arch`.
# Here, we increase the number of filters to 36, and number of cells to 20,
# so as to reasonably increase the model size and boost the performance.

from nni.retiarii import fixed_arch

with fixed_arch(exported_arch):
    final_model = DARTS(36, 20, 'cifar')

# %%
#
# We then train the model on full CIFAR-10 training dataset, and evaluate it on the original CIFAR-10 validation dataset.

train_loader = DataLoader(train_data, batch_size=96, num_workers=6)  # Use the original training data

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
valid_data = nni.trace(CIFAR10)(root='./data', train=False, download=True, transform=transform)
valid_loader = DataLoader(train_data, batch_size=256, num_workers=6)

# %%
#
# Create a new evaluator here because we can using a different data split.
# Also, we should avoid the underlying pytorch-lightning implementation of Classification evaluator from loading the wrong checkpoint.
#
# Here, to get an accuracy comparable to `pytorch-cifar repo <https://github.com/kuangliu/pytorch-cifar>`__,
# ``max_epochs`` should be further increased to at least 200.
# We only set it to 1 here for tutorial demo purposes.

max_epochs = 1

evaluator = Classification(
    learning_rate=0.01,
    weight_decay=1e-4,
    train_dataloaders=train_loader,
    val_dataloaders=valid_loader,
    max_epochs=max_epochs,
    gpus=1,
)

evaluator.fit(final_model)
