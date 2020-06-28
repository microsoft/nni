# 神经网络结构搜索在 NNI 上的应用

```eval_rst
.. contents::
```

## 概述

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 [NASNet](https://arxiv.org/abs/1707.07012)，[ENAS](https://arxiv.org/abs/1802.03268)，[DARTS](https://arxiv.org/abs/1806.09055)，[Network Morphism](https://arxiv.org/abs/1806.10282)，以及 [Evolution](https://arxiv.org/abs/1703.01041) 等。 此外，新的创新不断涌现。

但是，要实现NAS算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。 为了促进 NAS 创新（例如，设计、实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

以此为动力，NNI 的目标是提供统一的体系结构，以加速NAS上的创新，并将最新的算法更快地应用于现实世界中的问题上。

通过统一的接口，有两种方法来使用神经网络架构搜索。 [一种](#supported-one-shot-nas-algorithms)称为 one-shot NAS，基于搜索空间构建了一个超级网络，并使用 one-shot 训练来生成性能良好的子模型。 <a href="#支持的经典-nas-算法"">第二种</a>是经典的搜索方法，搜索空间中每个子模型作为独立的 Trial 运行。 称之为经典的 NAS。

NNI 还提供了专门的[可视化工具](#nas-可视化)，用于查看神经网络架构搜索的过程。

## 支持的经典 NAS 算法

经典 NAS 算法的过程类似于超参调优，通过 `nnictl` 来启动 Experiment，每个子模型会作为 Trial 运行。 不同之处在于，搜索空间文件是通过运行 `nnictl ss_gen`，从用户模型（已包含搜索空间）中自动生成。 下表列出了经典 NAS 模式支持的算法。 将来版本会支持更多算法。

| 名称                                                                                                   | 算法简介                                                                      |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [Random Search（随机搜索）](https://github.com/microsoft/nni/tree/master/examples/tuners/random_nas_tuner) | 从搜索空间中随机选择模型                                                              |
| [PPO Tuner](https://nni.readthedocs.io/zh/latest/Tuner/BuiltinTuner.html#PPOTuner)                   | PPO Tuner 是基于 PPO 算法的强化学习 Tuner。 [参考论文](https://arxiv.org/abs/1707.06347) |

参考[这里](ClassicNas.md)，了解如何使用经典 NAS 算法。

## 支持的 One-shot NAS 算法

NNI 目前支持下面列出的 One-Shot NAS 算法，并且正在添加更多算法。 用户可以重现算法或在自己的数据集上使用它。 鼓励用户使用 [NNI API](#use-nni-api) 实现其它算法，以使更多人受益。

| 名称                                                                         | 算法简介                                                                                                                                                                            |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ENAS](https://nni.readthedocs.io/zh/latest/NAS/ENAS.html)                 | [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268). 在 ENAS 中，Contoller 学习在大的计算图中搜索最有子图的方式来发现神经网络。 它通过在子模型间共享参数来实现加速和出色的性能指标。        |
| [DARTS](https://nni.readthedocs.io/zh/latest/NAS/DARTS.html)               | [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) 引入了一种在两级网络优化中使用的可微分算法。                                                                            |
| [P-DARTS](https://nni.readthedocs.io/zh/latest/NAS/PDARTS.html)            | [Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) 基于DARTS。 它引入了一种有效的算法，可在搜索过程中逐渐增加搜索的深度。 |
| [SPOS](https://nni.readthedocs.io/zh/latest/NAS/SPOS.html)                 | 论文 [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) 构造了一个采用统一的路径采样方法来训练简化的超网络，并使用进化算法来提高搜索神经网络结构的效率。                   |
| [CDARTS](https://nni.readthedocs.io/zh/latest/NAS/CDARTS.html)             | [Cyclic Differentiable Architecture Search](https://arxiv.org/abs/****) 在搜索和评估的网络见构建了循环反馈的机制。 通过引入的循环的可微分架构搜索框架将两个网络集成为一个架构。                                                    |
| [ProxylessNAS](https://nni.readthedocs.io/zh/latest/NAS/Proxylessnas.html) | [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332). 它删除了代理，直接从大规模目标任务和目标硬件平台进行学习。                                  |
| [TextNAS](https://nni.readthedocs.io/zh/latest/NAS/TextNAS.html)           | [TextNAS: A Neural Architecture Search Space tailored for Text Representation](https://arxiv.org/pdf/1912.10729.pdf)。 这是专门用于文本表示的神经网络架构搜索算法。                                    |

One-shot 算法**不需要 nnictl，可单独运行**。 NNI 支持 PyTorch 和 TensorFlow 2.x。

这是运行示例的一些常见依赖项。 PyTorch 需要高于 1.2 才能使用 `BoolTensor`.

* tensorboard
* PyTorch 1.2+
* git

参考[这里](NasGuide.md)，了解如何使用 One-Shot NAS 算法。

One-Shot NAS 可以通过可视化工具来查看。 点击[这里](./Visualization.md)，了解详情。


## 使用 NNI API 来编写搜索空间

在两种场景下需要用于设计和搜索模型的编程接口。

1. 在设计神经网络时，可能在层、子模型或连接上有多种选择，并且无法确定是其中一种或某些的组合的结果最好。 因此，需要简单的方法来表达候选的层或子模型。
2. 在神经网络上应用 NAS 时，需要统一的方式来表达架构的搜索空间，这样不必为不同的搜索算法来更改代码。

要使用 NNI NAS，建议先阅读[用 NAS API 构建搜索空间](./WriteSearchSpace.md)的教程。

## NAS 可视化

为了帮助用户跟踪指定搜索空间下搜索模型的过程和状态，开发了此可视化工具。 它将搜索空间可视化为超网络，并显示子网络、层和操作的重要性，同时还能显示重要性是如何在搜索过程中变化的。 参考 [NAS 可视化](./Visualization.md)文档了解详情。

## 参考和反馈

* 在 GitHub 中[提交此功能的 Bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md)；
* 在 GitHub 中[提交新功能或改进请求](https://github.com/microsoft/nni/issues/new?template=enhancement.md)。
