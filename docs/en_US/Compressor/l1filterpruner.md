L1FilterPruner on NNI
===

## Introduction

L1FilterPruner is a general structured pruning algorithm for pruning filters in the convolutional layers.

In ['PRUNING FILTERS FOR EFFICIENT CONVNETS'](https://arxiv.org/abs/1608.08710), authors Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet and Hans Peter Graf.

![](../../img/l1filter_pruner.png)

> L1Filter Pruner prunes filters in the **convolution layers**
>
> The procedure of pruning m filters from the ith convolutional layer is as follows:
>
> 1. For each filter ![](http://latex.codecogs.com/gif.latex?F_{i,j}), calculate the sum of its absolute kernel weights![](http://latex.codecogs.com/gif.latex?s_j=\sum_{l=1}^{n_i}\sum|K_l|)
> 2. Sort the filters by ![](http://latex.codecogs.com/gif.latex?s_j).
> 3. Prune ![](http://latex.codecogs.com/gif.latex?m) filters with the smallest sum values and their corresponding feature maps. The
>      kernels in the next convolutional layer corresponding to the pruned feature maps are also
>        removed.
> 4. A new kernel matrix is created for both the ![](http://latex.codecogs.com/gif.latex?i)th and ![](http://latex.codecogs.com/gif.latex?i+1)th layers, and the remaining kernel
>      weights are copied to the new model.

## Experiment

We implemented one of the experiments in ['PRUNING FILTERS FOR EFFICIENT CONVNETS'](https://arxiv.org/abs/1608.08710) with **L1FilterPruner**, we pruned **VGG-16** for CIFAR-10 to **VGG-16-pruned-A** in the paper, in which $64\%$ parameters are pruned. Our experiments results are as follows:

| Model           | Error(paper/ours) | Parameters      | Pruned   |
| --------------- | ----------------- | --------------- | -------- |
| VGG-16          | 6.75/6.49     | 1.5x10^7 |          |
| VGG-16-pruned-A | 6.60/6.47     | 5.4x10^6 | 64.0% |

The experiments code can be found at [examples/model_compress]( https://github.com/microsoft/nni/tree/master/examples/model_compress/)





