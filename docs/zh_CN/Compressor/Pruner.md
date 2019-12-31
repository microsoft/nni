NNI Compressor 中的 Pruner
===

Index of supported pruning algorithms
* [Level Pruner](#level-pruner)
* [AGP Pruner](#agp-pruner)
* [Lottery Ticket Hypothesis](#lottery-ticket-hypothesis)
* [Slim Pruner](#slim-pruner)
* [Filter Pruners with Weight Rank](#weightrankfilterpruner)
    * [FPGM Pruner](#fpgm-pruner)
    * [L1Filter Pruner](#l1filter-pruner)
    * [L2Filter Pruner](#l2filter-pruner)
* [Filter Pruners with Activation Rank](#activationrankfilterpruner)
    * [APoZ Rank Pruner](#activationapozrankfilterpruner)
    * [Activation Mean Rank Pruner](#activationmeanrankfilterpruner)

## Level Pruner

This is one basic one-shot pruner: you can set a target sparsity level (expressed as a fraction, 0.6 means we will prune 60%).

We first sort the weights in the specified layer by their absolute values. And then mask to zero the smallest magnitude weights until the desired sparsity level is reached.

### 用法

Tensorflow code
```python
from nni.compression.tensorflow import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(model_graph, config_list)
pruner.compress()
```

PyTorch code
```python
from nni.compression.torch import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(model, config_list)
pruner.compress()
```

#### Level Pruner 的用户配置
* **sparsity:** This is to specify the sparsity operations to be compressed to

***

## AGP Pruner
This is an iterative pruner, In [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878), authors Michael Zhu and Suyog Gupta provide an algorithm to prune the weight gradually.
> 我们引入了一种新的自动梯度剪枝算法。这种算法从初始的稀疏度值 si（一般为 0）开始，通过 n 步的剪枝操作，增加到最终所需的稀疏度 sf。从训练步骤 t0 开始，以 ∆t 为剪枝频率： ![](../../img/agp_pruner.png) 在神经网络训练时‘逐步增加网络稀疏度时，每训练  ∆t 步更新一次权重剪枝的二进制掩码。同时也允许训练步骤恢复因为剪枝而造成的精度损失。 根据我们的经验，∆t 设为 100 到 1000 个训练步骤之间时，对于模型最终精度的影响可忽略不计。 一旦模型达到了稀疏度目标 sf，权重掩码将不再更新。 公式背后的稀疏函数直觉。

### 用法
You can prune all weight from 0% to 80% sparsity in 10 epoch with the code below.

Tensorflow code
```python
from nni.compression.tensorflow import AGP_Pruner
config_list = [{
    'initial_sparsity': 0,
    'final_sparsity': 0.8,
    'start_epoch': 0,
    'end_epoch': 10,
    'frequency': 1,
    'op_types': 'default'
}]
pruner = AGP_Pruner(tf.get_default_graph(), config_list)
pruner.compress()
```
PyTorch code
```python
from nni.compression.torch import AGP_Pruner
config_list = [{
    'initial_sparsity': 0,
    'final_sparsity': 0.8,
    'start_epoch': 0,
    'end_epoch': 10,
    'frequency': 1,
    'op_types': ['default']
}]
pruner = AGP_Pruner(model, config_list)
pruner.compress()
```

you should add code below to update epoch number when you finish one epoch in your training code.

Tensorflow code
```python
pruner.update_epoch(epoch, sess)
```
PyTorch code
```python
pruner.update_epoch(epoch)
```
You can view example for more information

#### AGP Pruner 的用户配置
* **initial_sparsity:** This is to specify the sparsity when compressor starts to compress
* **final_sparsity:** This is to specify the sparsity when compressor finishes to compress
* **start_epoch:** This is to specify the epoch number when compressor starts to compress, default start from epoch 0
* **end_epoch:** This is to specify the epoch number when compressor finishes to compress
* **frequency:** This is to specify every *frequency* number epochs compressor compress once, default frequency=1

***

## Lottery Ticket 假设
[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635), authors Jonathan Frankle and Michael Carbin,provides comprehensive measurement and analysis, and articulate the *lottery ticket hypothesis*: dense, randomly-initialized, feed-forward networks contain subnetworks (*winning tickets*) that -- when trained in isolation -- reach test accuracy comparable to the original network in a similar number of iterations.

In this paper, the authors use the following process to prune a model, called *iterative prunning*:
> 1. 随机初始化一个神经网络 f(x;theta_0) (其中 theta_0 为 D_{theta}).
> 2. 将网络训练 j 次，得出参数 theta_j。
> 3. 在 theta_j 修剪参数的 p%，创建掩码 m。
> 4. 将其余参数重置为 theta_0 的值，创建获胜彩票 f(x;m*theta_0)。
> 5. 重复步骤 2、3 和 4。

If the configured final sparsity is P (e.g., 0.8) and there are n times iterative pruning, each iterative pruning prunes 1-(1-P)^(1/n) of the weights that survive the previous round.

### 用法

PyTorch code
```python
from nni.compression.torch import LotteryTicketPruner
config_list = [{
    'prune_iterations': 5,
    'sparsity': 0.8,
    'op_types': ['default']
}]
pruner = LotteryTicketPruner(model, config_list, optimizer)
pruner.compress()
for _ in pruner.get_prune_iterations():
    pruner.prune_iteration_start()
    for epoch in range(epoch_num):
        ...
```

The above configuration means that there are 5 times of iterative pruning. As the 5 times iterative pruning are executed in the same run, LotteryTicketPruner needs `model` and `optimizer` (**Note that should add `lr_scheduler` if used**) to reset their states every time a new prune iteration starts. Please use `get_prune_iterations` to get the pruning iterations, and invoke `prune_iteration_start` at the beginning of each iteration. `epoch_num` is better to be large enough for model convergence, because the hypothesis is that the performance (accuracy) got in latter rounds with high sparsity could be comparable with that got in the first round. Simple reproducing results can be found [here](./LotteryTicketHypothesis.md).


*Tensorflow version will be supported later.*

#### LotteryTicketPruner 的用户配置

* **prune_iterations:** The number of rounds for the iterative pruning, i.e., the number of iterative pruning.
* **sparsity:** The final sparsity when the compression is done.

***

## Slim Pruner

This is an one-shot pruner, In ['Learning Efficient Convolutional Networks through Network Slimming'](https://arxiv.org/pdf/1708.06519.pdf), authors Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan and Changshui Zhang.

![](../../img/slim_pruner.png)

> Slim Pruner **prunes channels in the convolution layers by masking corresponding scaling factors in the later BN layers**, L1 regularization on the scaling factors should be applied in batch normalization (BN) layers while training, scaling factors of BN layers are **globally ranked** while pruning, so the sparse model can be automatically found given sparsity.

### Usage

PyTorch code

```python
from nni.compression.torch import SlimPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['BatchNorm2d'] }]
pruner = SlimPruner(model, config_list)
pruner.compress()
```

#### User configuration for Slim Pruner

- **sparsity:**，指定压缩的稀疏度。
- **op_types:** Only BatchNorm2d is supported in Slim Pruner


## WeightRankFilterPruner
WeightRankFilterPruner is a series of pruners which prune the filters with the smallest importance criterion calculated from the weights in convolution layers to achieve a preset level of network sparsity

### FPGM Pruner

This is an one-shot pruner, FPGM Pruner is an implementation of paper [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/pdf/1811.00250.pdf)

FPGMPruner prune filters with the smallest geometric median

 ![](../../img/fpgm_fig1.png)
> Previous works utilized “smaller-norm-less-important” criterion to prune filters with smaller norm values in a convolutional neural network. In this paper, we analyze this norm-based criterion and point out that its effectiveness depends on two requirements that are not always met: (1) the norm deviation of the filters should be large; (2) the minimum norm of the filters should be small. To solve this problem, we propose a novel filter pruning method, namely Filter Pruning via Geometric Median (FPGM), to compress the model regardless of those two requirements. Unlike previous methods, FPGM compresses CNN models by pruning filters with redundancy, rather than those with “relatively less” importance.

#### Usage

Tensorflow code
```python
from nni.compression.tensorflow import FPGMPruner
config_list = [{
    'sparsity': 0.5,
    'op_types': ['Conv2D']
}]
pruner = FPGMPruner(model, config_list)
pruner.compress()
```
PyTorch code
```python
from nni.compression.torch import FPGMPruner
config_list = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d']
}]
pruner = FPGMPruner(model, config_list)
pruner.compress()
```
Note: FPGM Pruner is used to prune convolutional layers within deep neural networks, therefore the `op_types` field supports only convolutional layers.

you should add code below to update epoch number at beginning of each epoch.

Tensorflow code
```python
pruner.update_epoch(epoch, sess)
```
PyTorch code
```python
pruner.update_epoch(epoch)
```
You can view example for more information

#### User configuration for FPGM Pruner
* **sparsity:** How much percentage of convolutional filters are to be pruned.

***

### L1Filter Pruner

This is an one-shot pruner, In ['PRUNING FILTERS FOR EFFICIENT CONVNETS'](https://arxiv.org/abs/1608.08710), authors Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet and Hans Peter Graf. The reproduced experiment results can be found [here](l1filterpruner.md)

![](../../img/l1filter_pruner.png)

> L1Filter Pruner prunes filters in the **convolution layers**
> 
> The procedure of pruning m filters from the ith convolutional layer is as follows:
> 
> 1. For each filter ![](http://latex.codecogs.com/gif.latex?F_{i,j}), calculate the sum of its absolute kernel weights![](http://latex.codecogs.com/gif.latex?s_j=\sum_{l=1}^{n_i}\sum|K_l|)
> 2. Sort the filters by ![](http://latex.codecogs.com/gif.latex?s_j).
> 3. Prune ![](http://latex.codecogs.com/gif.latex?m) filters with the smallest sum values and their corresponding feature maps. The kernels in the next convolutional layer corresponding to the pruned feature maps are also removed.
> 4. A new kernel matrix is created for both the ![](http://latex.codecogs.com/gif.latex?i)th and ![](http://latex.codecogs.com/gif.latex?i+1)th layers, and the remaining kernel weights are copied to the new model.

#### Usage

PyTorch code

```python
from nni.compression.torch import L1FilterPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
pruner = L1FilterPruner(model, config_list)
pruner.compress()
```

#### User configuration for L1Filter Pruner

- **sparsity:** This is to specify the sparsity operations to be compressed to
- **op_types:** Only Conv1d and Conv2d is supported in L1Filter Pruner

***

### L2Filter Pruner

This is a structured pruning algorithm that prunes the filters with the smallest L2 norm of the weights. It is implemented as a one-shot pruner.

#### Usage

PyTorch code

```python
from nni.compression.torch import L2FilterPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
pruner = L2FilterPruner(model, config_list)
pruner.compress()
```

#### User configuration for L2Filter Pruner

- **sparsity:** This is to specify the sparsity operations to be compressed to
- **op_types:** Only Conv1d and Conv2d is supported in L2Filter Pruner

## ActivationRankFilterPruner
ActivationRankFilterPruner is a series of pruners which prune the filters with the smallest importance criterion calculated from the output activations of convolution layers to achieve a preset level of network sparsity

### ActivationAPoZRankFilterPruner

We implemented it as a one-shot pruner, it prunes convolutional layers based on the criterion `APoZ` which is explained in the paper [Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures](https://arxiv.org/abs/1607.03250). Iterative pruning based on `APoZ` will be supported in future release.

The APoZ is defined as:

![](../../img/apoz.png)

#### Usage

PyTorch code

```python
from nni.compression.torch import ActivationAPoZRankFilterPruner
config_list = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d']
}]
pruner = ActivationAPoZRankFilterPruner(model, config_list, statistics_batch_num=1)
pruner.compress()
```

Note: ActivationAPoZRankFilterPruner is used to prune convolutional layers within deep neural networks, therefore the `op_types` field supports only convolutional layers.

You can view example for more information

#### User configuration for ActivationAPoZRankFilterPruner

- **sparsity:** How much percentage of convolutional filters are to be pruned.
- **op_types:** Only Conv2d is supported in ActivationAPoZRankFilterPruner

***

### ActivationMeanRankFilterPruner

We implemented it as a one-shot pruner, it prunes convolutional layers based on the criterion `mean activation` which is explained in section 2.2 of the paper[Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440). Other pruning criteria mentioned in this paper will be supported in future release.

#### Usage

PyTorch code

```python
from nni.compression.torch import ActivationMeanRankFilterPruner
config_list = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d']
}]
pruner = ActivationMeanRankFilterPruner(model, config_list)
pruner.compress()
```

Note: ActivationMeanRankFilterPruner is used to prune convolutional layers within deep neural networks, therefore the `op_types` field supports only convolutional layers.

You can view example for more information

#### User configuration for ActivationMeanRankFilterPruner

- **sparsity:** How much percentage of convolutional filters are to be pruned.
- **op_types:** Only Conv2d is supported in ActivationMeanRankFilterPruner

***