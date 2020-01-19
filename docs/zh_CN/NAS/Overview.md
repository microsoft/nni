# 神经网络结构搜索在 NNI 上的应用

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 [NASNet](https://arxiv.org/abs/1707.07012)，[ENAS](https://arxiv.org/abs/1802.03268)，[DARTS](https://arxiv.org/abs/1806.09055)，[Network Morphism](https://arxiv.org/abs/1806.10282)，以及 [Evolution](https://arxiv.org/abs/1703.01041) 等。 新的算法还在不断涌现。

但是，要实现NAS算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。 为了促进 NAS 创新（例如，设计、实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

以此为动力，NNI 的目标是提供统一的体系结构，以加速NAS上的创新，并将最新的算法更快地应用于现实世界中的问题上。

通过[统一的接口](./NasInterface.md)，有两种方式进行架构搜索。 [一种](#supported-one-shot-nas-algorithms)称为 one-shot NAS，基于搜索空间构建了一个超级网络，并使用 one-shot 训练来生成性能良好的子模型。 [第二种](./NasInterface.md#classic-distributed-search)是传统的搜索方法，搜索空间中每个子模型作为独立的 Trial 运行，将性能结果发给 Tuner，由 Tuner 来生成新的子模型。

* [支持的 One-shot NAS 算法](#supported-one-shot-nas-algorithms)
* [使用 NNI Experiment 的经典分布式 NAS](./NasInterface.md#classic-distributed-search)
* [NNI NAS 编程接口](./NasInterface.md)

## 支持的 One-shot NAS 算法

NNI 现在支持以下 NAS 算法，并且正在添加更多算法。 用户可以重现算法或在自己的数据集上使用它。 鼓励用户使用 [NNI API](#use-nni-api) 实现其它算法，以使更多人受益。

| 名称                   | 算法简介                                                                                                                                                                            |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ENAS](ENAS.md)      | [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268). 在 ENAS 中，Contoller 学习在大的计算图中搜索最有子图的方式来发现神经网络。 它通过在子模型间共享参数来实现加速和出色的性能指标。        |
| [DARTS](DARTS.md)    | [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) 引入了一种在两级网络优化中使用的可微分算法。                                                                            |
| [P-DARTS](PDARTS.md) | [Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) 基于DARTS。 它引入了一种有效的算法，可在搜索过程中逐渐增加搜索的深度。 |
| [SPOS](SPOS.md)      | 论文 [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) 构造了一个采用统一的路径采样方法来训练简化的超网络，并使用进化算法来提高搜索神经网络结构的效率。                   |
| [CDARTS](CDARTS.md)  | [Cyclic Differentiable Architecture Search](https://arxiv.org/abs/****) 在搜索和评估的网络见构建了循环反馈的机制。 通过引入的循环的可微分架构搜索框架将两个网络集成为一个架构。                                                    |

One-shot 算法**不需要 nnictl，可单独运行**。 只实现了 PyTorch 版本。 将来的版本会支持 Tensorflow 2.x。

这是运行示例的一些常见依赖项。 PyTorch 需要高于 1.2 才能使用 `BoolTensor`.

* NNI 1.2+
* tensorboard
* PyTorch 1.2+
* git

## 使用 NNI API

注意，我们正在尝试通过统一的编程接口来支持各种 NAS 算法，当前处于试验阶段。 这意味着当前编程接口将来会有变化。

### 编程接口

在两种场景下需要用于设计和搜索模型的编程接口。

1. 在设计神经网络时，可能在层、子模型或连接上有多种选择，并且无法确定是其中一种或某些的组合的结果最好。 因此，需要简单的方法来表达候选的层或子模型。
2. 在神经网络上应用 NAS 时，需要统一的方式来表达架构的搜索空间，这样不必为不同的搜索算法来更改代码。

NNI 提出的 API 在[这里](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch)。 [这里](https://github.com/microsoft/nni/tree/master/examples/nas/naive)包含了基于此 API 的 NAS 实现示例。

## **参考和反馈**
* 在 GitHub 中[提交此功能的 Bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md)；
* 在 GitHub 中[提交新功能或改进请求](https://github.com/microsoft/nni/issues/new?template=enhancement.md)；
* 了解 NNI 中[特征工程的更多信息](https://github.com/microsoft/nni/blob/master/docs/zh_CN/FeatureEngineering/Overview.md)；
* 了解 NNI 中[模型自动压缩的更多信息](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Compressor/Overview.md)；
* 了解如何[使用 NNI 进行超参数调优](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Tuner/BuiltinTuner.md)；
