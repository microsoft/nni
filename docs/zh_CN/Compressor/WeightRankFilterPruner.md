WeightRankFilterPruner on NNI Compressor
===

## 1. 介绍

WeightRankFilterPruner is a series of pruners which prune filters according to some importance criterion calculated from the filters' weight.

|     Pruner     |    Importance criterion     |                                                         Reference paper                                                         |
|:--------------:|:---------------------------:|:-------------------------------------------------------------------------------------------------------------------------------:|
| L1FilterPruner |     L1 norm of weights      |                           [PRUNING FILTERS FOR EFFICIENT CONVNETS](https://arxiv.org/abs/1608.08710)                            |
| L2FilterPruner |     L2 norm of weights      |                                                                                                                                 |
|   FPGMPruner   | Geometric Median of weights | [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/pdf/1811.00250.pdf) |

## 2. Pruners

### L1FilterPruner

L1FilterPruner is a general structured pruning algorithm for pruning filters in the convolutional layers.

In ['PRUNING FILTERS FOR EFFICIENT CONVNETS'](https://arxiv.org/abs/1608.08710), authors Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet and Hans Peter Graf.

![](../../img/l1filter_pruner.png)

> L1Filter Pruner 修剪**卷积层**中的过滤器
> 
> 从第 i 个卷积层修剪 m 个过滤器的过程如下：
> 
> 1. 对于每个过滤器 ![](http://latex.codecogs.com/gif.latex?F_{i,j})，计算其绝对内核权重之和![](http://latex.codecogs.com/gif.latex?s_j=\sum_{l=1}^{n_i}\sum|K_l|)
> 2. 将过滤器按 ![](http://latex.codecogs.com/gif.latex?s_j) 排序。
> 3. 修剪 ![](http://latex.codecogs.com/gif.latex?m) 具有最小求和值及其相应特征图的筛选器。 The kernels in the next convolutional layer corresponding to the pruned feature maps are also removed.
> 4. A new kernel matrix is created for both the ![](http://latex.codecogs.com/gif.latex?i)th and ![](http://latex.codecogs.com/gif.latex?i+1)th layers, and the remaining kernel weights are copied to the new model.

### L2FilterPruner

L2FilterPruner is similar to L1FilterPruner, but only replace the importance criterion from L1 norm to L2 norm

### FPGMPruner

Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, Yi Yang

"[Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/abs/1811.00250)", CVPR 2019.

FPGMPruner prune filters with the smallest geometric median

 ![](../../img/fpgm_fig1.png)

## 3. Usage

PyTorch code

```
from nni.compression.torch import L1FilterPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'], 'op_names': ['conv1', 'conv2'] }]
pruner = L1FilterPruner(model, config_list)
pruner.compress()
```

#### L1Filter Pruner 的用户配置

- **sparsity:**，指定压缩的稀疏度。
- **op_types:** 在 L1Filter Pruner 中仅支持 Conv2d。

## 4. Experiment

We implemented one of the experiments in ['PRUNING FILTERS FOR EFFICIENT CONVNETS'](https://arxiv.org/abs/1608.08710) with **L1FilterPruner**, we pruned **VGG-16** for CIFAR-10 to **VGG-16-pruned-A** in the paper, in which $64\%$ parameters are pruned. Our experiments results are as follows:

| Model           | Error(paper/ours) | Parameters | Pruned |
| --------------- | ----------------- | ---------- | ------ |
| VGG-16          | 6.75/6.49         | 1.5x10^7   |        |
| VGG-16-pruned-A | 6.60/6.47         | 5.4x10^6   | 64.0%  |

The experiments code can be found at [examples/model_compress](https://github.com/microsoft/nni/tree/master/examples/model_compress/)





