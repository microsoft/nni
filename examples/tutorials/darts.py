"""
Searching on DARTS search space
===============================

In this tutorial, we demonstrate how to search on the famous model space proposed in `DARTS <https://arxiv.org/abs/1806.09055>`__.

Through this process, you will learn:

* How to use the built-in model spaces from NNI's model space hub.
* How to use one-shot exploration strategies to explore a model space.
* How to customize evaluators to achieve the best performance.

In the end, we get a strong-performing model on CIFAR-10 dataset, which achieves xx.xx% accuracy.

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

from nni.retiarii.hub.pytorch import DARTS

model_space = DARTS(16, 8, 'cifar')

# %%
# Search on the model space
# -------------------------
#
#