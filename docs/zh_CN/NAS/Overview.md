# 神经网络结构搜索在 NNI 上的应用

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 [NASNet](https://arxiv.org/abs/1707.07012)，[ENAS](https://arxiv.org/abs/1802.03268)，[DARTS](https://arxiv.org/abs/1806.09055)，[Network Morphism](https://arxiv.org/abs/1806.10282)，以及 [Evolution](https://arxiv.org/abs/1703.01041) 等。 新的算法还在不断涌现。

但是，要实现NAS算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。 为了促进 NAS 创新（例如，设计、实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

以此为动力，NNI 的目标是提供统一的体系结构，以加速NAS上的创新，并将最新的算法更快地应用于现实世界中的问题上。

With the unified interface, there are two different modes for the architecture search. [一种](#supported-one-shot-nas-algorithms)称为 one-shot NAS，基于搜索空间构建了一个超级网络，并使用 one-shot 训练来生成性能良好的子模型。 [The other](#supported-distributed-nas-algorithms) is the traditional searching approach, where each child model in search space runs as an independent trial, the performance result is sent to tuner and the tuner generates new child model.

## 支持的 One-shot NAS 算法

NNI 现在支持以下 NAS 算法，并且正在添加更多算法。 用户可以重现算法或在自己的数据集上使用它。 鼓励用户使用 [NNI API](#use-nni-api) 实现其它算法，以使更多人受益。

| 名称                              | 算法简介                                                                                                                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ENAS](ENAS.md)                 | [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268). 在 ENAS 中，Contoller 学习在大的计算图中搜索最有子图的方式来发现神经网络。 它通过在子模型间共享参数来实现加速和出色的性能指标。        |
| [DARTS](DARTS.md)               | [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) 引入了一种在两级网络优化中使用的可微分算法。                                                                            |
| [P-DARTS](PDARTS.md)            | [Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) 基于DARTS。 它引入了一种有效的算法，可在搜索过程中逐渐增加搜索的深度。 |
| [SPOS](SPOS.md)                 | 论文 [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) 构造了一个采用统一的路径采样方法来训练简化的超网络，并使用进化算法来提高搜索神经网络结构的效率。                   |
| [CDARTS](CDARTS.md)             | [Cyclic Differentiable Architecture Search](https://arxiv.org/abs/****) 在搜索和评估的网络见构建了循环反馈的机制。 通过引入的循环的可微分架构搜索框架将两个网络集成为一个架构。                                                    |
| [ProxylessNAS](Proxylessnas.md) | [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332).                                                                |

One-shot 算法**不需要 nnictl，可单独运行**。 只实现了 PyTorch 版本。 将来的版本会支持 Tensorflow 2.x。

这是运行示例的一些常见依赖项。 PyTorch 需要高于 1.2 才能使用 `BoolTensor`.

* NNI 1.2+
* tensorboard
* PyTorch 1.2+
* git

## Supported Distributed NAS Algorithms

| Name            | Brief Introduction of Algorithm                                                                                                                                                                                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [SPOS](SPOS.md) | [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) constructs a simplified supernet trained with an uniform path sampling method, and applies an evolutionary algorithm to efficiently search for the best-performing architectures. |

```eval_rst
.. Note:: SPOS is a two-stage algorithm, whose first stage is one-shot and second stage is distributed, leveraging result of first stage as a checkpoint.
```

## Use NNI API

The programming interface of designing and searching a model is often demanded in two scenarios.

1. 在设计神经网络时，可能在层、子模型或连接上有多种选择，并且无法确定是其中一种或某些的组合的结果最好。 因此，需要简单的方法来表达候选的层或子模型。
2. 在神经网络上应用 NAS 时，需要统一的方式来表达架构的搜索空间，这样不必为不同的搜索算法来更改代码。

[Here](./NasGuide.md) is a user guide to get started with using NAS on NNI.

## Reference and Feedback

* To [report a bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md) for this feature in GitHub;
* To [file a feature or improvement request](https://github.com/microsoft/nni/issues/new?template=enhancement.md) for this feature in GitHub.