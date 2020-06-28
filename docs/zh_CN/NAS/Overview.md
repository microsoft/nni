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
| [PPO Tuner](https://nni.readthedocs.io/en/latest/Tuner/BuiltinTuner.html#PPOTuner)                   | PPO Tuner 是基于 PPO 算法的强化学习 Tuner。 [参考论文](https://arxiv.org/abs/1707.06347) |

参考[这里](ClassicNas.md)，了解如何使用经典 NAS 算法。

## 支持的 One-shot NAS 算法

NNI 目前支持下面列出的 One-Shot NAS 算法，并且正在添加更多算法。 用户可以重现算法或在自己的数据集上使用它。 鼓励用户使用 [NNI API](#use-nni-api) 实现其它算法，以使更多人受益。

| 名称                                                                         | 算法简介                                                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ENAS](https://nni.readthedocs.io/en/latest/NAS/ENAS.html)                 | [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268). In ENAS, a controller learns to discover neural network architectures by searching for an optimal subgraph within a large computational graph. It uses parameter sharing between child models to achieve fast speed and excellent performance. |
| [DARTS](https://nni.readthedocs.io/en/latest/NAS/DARTS.html)               | [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) introduces a novel algorithm for differentiable network architecture search on bilevel optimization.                                                                                                                                                             |
| [P-DARTS](https://nni.readthedocs.io/en/latest/NAS/PDARTS.html)            | [Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) is based on DARTS. It introduces an efficient algorithm which allows the depth of searched architectures to grow gradually during the training procedure.                                             |
| [SPOS](https://nni.readthedocs.io/en/latest/NAS/SPOS.html)                 | [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) constructs a simplified supernet trained with a uniform path sampling method and applies an evolutionary algorithm to efficiently search for the best-performing architectures.                                                      |
| [CDARTS](https://nni.readthedocs.io/en/latest/NAS/CDARTS.html)             | [Cyclic Differentiable Architecture Search](https://arxiv.org/abs/****) builds a cyclic feedback mechanism between the search and evaluation networks. It introduces a cyclic differentiable architecture search framework which integrates the two networks into a unified architecture.                                                      |
| [ProxylessNAS](https://nni.readthedocs.io/en/latest/NAS/Proxylessnas.html) | [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332). It removes proxy, directly learns the architectures for large-scale target tasks and target hardware platforms.                                                                                                               |
| [TextNAS](https://nni.readthedocs.io/en/latest/NAS/TextNAS.html)           | [TextNAS: A Neural Architecture Search Space tailored for Text Representation](https://arxiv.org/pdf/1912.10729.pdf). It is a neural architecture search algorithm tailored for text representation.                                                                                                                                           |

One-shot algorithms run **standalone without nnictl**. NNI supports both PyTorch and Tensorflow 2.X.

Here are some common dependencies to run the examples. PyTorch needs to be above 1.2 to use `BoolTensor`.

* tensorboard
* PyTorch 1.2+
* git

Please refer to [here](NasGuide.md) for the usage of one-shot NAS algorithms.

One-shot NAS can be visualized with our visualization tool. Learn more details [here](./Visualization.md).


## Using NNI API to Write Your Search Space

The programming interface of designing and searching a model is often demanded in two scenarios.

1. 在设计神经网络时，可能在层、子模型或连接上有多种选择，并且无法确定是其中一种或某些的组合的结果最好。 因此，需要简单的方法来表达候选的层或子模型。
2. 在神经网络上应用 NAS 时，需要统一的方式来表达架构的搜索空间，这样不必为不同的搜索算法来更改代码。

For using NNI NAS, we suggest users to first go through [the tutorial of NAS API for building search space](./WriteSearchSpace.md).

## NAS 可视化

To help users track the process and status of how the model is searched under specified search space, we developed a visualization tool. It visualizes search space as a super-net and shows importance of subnets and layers/operations, as well as how the importance changes along with the search process. Please refer to [the document of NAS visualization](./Visualization.md) for how to use it.

## Reference and Feedback

* 在 GitHub 中[提交此功能的 Bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md)；
* 在 GitHub 中[提交新功能或改进请求](https://github.com/microsoft/nni/issues/new?template=enhancement.md)。
