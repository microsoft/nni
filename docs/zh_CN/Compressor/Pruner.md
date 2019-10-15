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
> 我们引入了一种新的自动梯度剪枝算法。这种算法从初始的稀疏度值 si（一般为 0）开始，通过 n 步的剪枝操作，增加到最终所需的稀疏度 sf。从训练步骤 t0 开始，以 ∆t 为剪枝频率： ![](../../img/agp_pruner.png) 在神经网络训练时‘逐步增加网络稀疏度时，每训练  ∆t 步更新一次权重剪枝的二进制掩码。同时也允许训练步骤恢复因为剪枝而造成的精度损失。 根据我们的经验，∆t 设为 100 到 1000 个训练步骤之间时，对于模型最终精度的影响可忽略不计。 一旦模型达到了稀疏度目标 sf，权重掩码将不再更新。 公式背后的稀疏函数直觉。

### 用法
通过下列代码，可以在 10 个 Epoch 中将权重稀疏度从 0% 剪枝到 80%。

首先，导入 Pruner 来为模型添加遮盖。

TensorFlow 代码
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
PyTorch 代码
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

其次，在训练代码中每完成一个 Epoch，更新一下 Epoch 数值。

TensorFlow 代码
```python
pruner.update_epoch(epoch, sess)
```
PyTorch 代码
```python
pruner.update_epoch(epoch)
```
查看示例进一步了解

#### AGP Pruner 的用户配置
* **initial_sparsity:** 指定了 Compressor 开始压缩的稀疏度。
* **final_sparsity:** 指定了 Compressor 压缩结束时的稀疏度。
* **start_epoch:** 指定了 Compressor 开始压缩时的 Epoch 数值。
* **end_epoch:** 指定了 Compressor 结束压缩时的 Epoch 数值。
* **frequency:** 指定了 Compressor 每过多少个 Epoch 进行一次剪枝。

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
