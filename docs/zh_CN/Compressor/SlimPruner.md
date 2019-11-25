NNI Compressor 中的 SlimPruner
===

## 1. Slim Pruner

SlimPruner 是一种结构化的修剪算法，通过修剪卷积层后对应的 BN 层相应的缩放因子来修剪通道。

在 ['Learning Efficient Convolutional Networks through Network Slimming'](https://arxiv.org/pdf/1708.06519.pdf) 中提出，作者 Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan 以及 Changshui Zhang。

![](../../img/slim_pruner.png)

> Slim Pruner **会遮盖卷据层通道之后 BN 层对应的缩放因子**，训练时在缩放因子上的 L1 正规化应在批量正规化 (BN) 层之后来做。BN 层的缩放因子在修剪时，是**全局排序的**，因此稀疏模型能自动找到给定的稀疏度。

## 2. 用法

PyTorch 代码

```
from nni.compression.torch import SlimPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['BatchNorm2d'] }]
pruner = SlimPruner(model, config_list)
pruner.compress()
```

#### Filter Pruner 的用户配置

- **sparsity:**，指定压缩的稀疏度。
- **op_types:** 在 Slim Pruner 中仅支持 BatchNorm2d。

## 3. 实验

我们实现了 ['Learning Efficient Convolutional Networks through Network Slimming'](https://arxiv.org/pdf/1708.06519.pdf) 中的一项实验。根据论文，对 CIFAR-10 上的 **VGGNet** 剪除了 $70\%$ 的通道，即约 $88.5\%$ 的参数。 我们的实验结果如下：

| 模型            | 错误率(论文/我们的) | 参数量    | 剪除率   |
| ------------- | ----------- | ------ | ----- |
| VGGNet        | 6.34/6.40   | 20.04M |       |
| Pruned-VGGNet | 6.20/6.26   | 2.03M  | 88.5% |

实验代码在 [examples/model_compress](https://github.com/microsoft/nni/tree/master/examples/model_compress/)
