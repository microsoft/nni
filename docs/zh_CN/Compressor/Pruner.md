NNI Compressor 中的 Pruner
===

## Level Pruner

这是个基本的 Pruner：可设置目标稀疏度（以分数表示，0.6 表示会剪除 60%）。

首先按照绝对值对指定层的权重排序。 然后按照所需的稀疏度，将值最小的权重屏蔽为 0。

### 用法

TensorFlow 代码
```
from nni.compression.tensorflow import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': 'default' }]
pruner = LevelPruner(config_list)
pruner(model_graph)
```

PyTorch 代码
```
from nni.compression.torch import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': 'default' }]
pruner = LevelPruner(config_list)
pruner(model)
```

#### Level Pruner 的用户配置
* **sparsity:**，指定压缩的稀疏度。

***

## AGP Pruner
在 [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878)中，作者 Michael Zhu 和 Suyog Gupta 提出了一种逐渐修建权重的算法。
> We introduce a new automated gradual pruning algorithm in which the sparsity is increased from an initial sparsity value si (usually 0) to a final sparsity value sf over a span of n pruning steps, starting at training step t0 and with pruning frequency ∆t: ![](../../img/agp_pruner.png) The binary weight masks are updated every ∆t steps as the network is trained to gradually increase the sparsity of the network while allowing the network training steps to recover from any pruning-induced loss in accuracy. In our experience, varying the pruning frequency ∆t between 100 and 1000 training steps had a negligible impact on the final model quality. Once the model achieves the target sparsity sf , the weight masks are no longer updated. The intuition behind this sparsity function in equation

### Usage
You can prune all weight from 0% to 80% sparsity in 10 epoch with the code below.

First, you should import pruner and add mask to model.

Tensorflow code
```python
from nni.compression.tensorflow import AGP_Pruner
config_list = [{
    'initial_sparsity': 0,
    'final_sparsity': 0.8,
    'start_epoch': 1,
    'end_epoch': 10,
    'frequency': 1,
    'op_types': 'default'
}]
pruner = AGP_Pruner(config_list)
pruner(tf.get_default_graph())
```
PyTorch code
```python
from nni.compression.torch import AGP_Pruner
config_list = [{
    'initial_sparsity': 0,
    'final_sparsity': 0.8,
    'start_epoch': 1,
    'end_epoch': 10,
    'frequency': 1,
    'op_types': 'default'
}]
pruner = AGP_Pruner(config_list)
pruner(model)
```

Second, you should add code below to update epoch number when you finish one epoch in your training code.

Tensorflow code
```python
pruner.update_epoch(epoch, sess)
```
PyTorch code
```python
pruner.update_epoch(epoch)
```
You can view example for more information

#### User configuration for AGP Pruner
* **initial_sparsity:** This is to specify the sparsity when compressor starts to compress
* **final_sparsity:** This is to specify the sparsity when compressor finishes to compress
* **start_epoch:** This is to specify the epoch number when compressor starts to compress
* **end_epoch:** This is to specify the epoch number when compressor finishes to compress
* **frequency:** This is to specify every *frequency* number epochs compressor compress once

***

## Sensitivity Pruner
In [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626), author Song Han and provide an algorithm to find the sensitivity of each layer and set the pruning threshold to each layer.
> We used the sensitivity results to find each layer’s threshold: for example, the smallest threshold was applied to the most sensitive layer, which is the first convolutional layer... The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights

### Usage
You can prune weight step by step and reach one target sparsity by Sensitivity Pruner with the code below.

Tensorflow code
```python
from nni.compression.tensorflow import SensitivityPruner
config_list = [{ 'sparsity':0.8, 'op_types': 'default' }]
pruner = SensitivityPruner(config_list)
pruner(tf.get_default_graph())
```
PyTorch code
```python
from nni.compression.torch import SensitivityPruner
config_list = [{ 'sparsity':0.8, 'op_types': 'default' }]
pruner = SensitivityPruner(config_list)
pruner(model)
```
Like AGP Pruner, you should update mask information every epoch by adding code below

Tensorflow code
```python
pruner.update_epoch(epoch, sess)
```
PyTorch code
```python
pruner.update_epoch(epoch)
```
You can view example for more information

#### User configuration for Sensitivity Pruner
* **sparsity:** This is to specify the sparsity operations to be compressed to

***
