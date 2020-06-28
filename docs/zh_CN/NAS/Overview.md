# 神经网络结构搜索在 NNI 上的应用

```eval_rst
.. contents::
```

## 概述

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 [NASNet](https://arxiv.org/abs/1707.07012)，[ENAS](https://arxiv.org/abs/1802.03268)，[DARTS](https://arxiv.org/abs/1806.09055)，[Network Morphism](https://arxiv.org/abs/1806.10282)，以及 [Evolution](https://arxiv.org/abs/1703.01041) 等。 此外，新的创新不断涌现。

但是，要实现NAS算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。 为了促进 NAS 创新（例如，设计、实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

以此为动力，NNI 的目标是提供统一的体系结构，以加速NAS上的创新，并将最新的算法更快地应用于现实世界中的问题上。

通过统一的接口，有两种方法来使用神经网络架构搜索。 [一种](#supported-one-shot-nas-algorithms)称为 one-shot NAS，基于搜索空间构建了一个超级网络，并使用 one-shot 训练来生成性能良好的子模型。 <a href="#支持的经典-nas-算法"">第二种</a>是经典的搜索方法，搜索空间中每个子模型作为独立的 Trial 运行。 称之为经典的 NAS。

NNI also provides dedicated [visualization tool](#nas-visualization) for users to check the status of the neural architecture search process.

## 支持的经典 NAS 算法

The procedure of classic NAS algorithms is similar to hyper-parameter tuning, users use `nnictl` to start experiments and each model runs as a trial. The difference is that search space file is automatically generated from user model (with search space in it) by running `nnictl ss_gen`. The following table listed supported tuning algorihtms for classic NAS mode. More algorihtms will be supported in future release.

| 名称                                                                                             | 算法简介                                                                                                                    |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [Random Search](https://github.com/microsoft/nni/tree/master/examples/tuners/random_nas_tuner) | Randomly pick a model from search space                                                                                 |
| [PPO Tuner](https://nni.readthedocs.io/en/latest/Tuner/BuiltinTuner.html#PPOTuner)             | PPO Tuner is a Reinforcement Learning tuner based on PPO algorithm. [Reference Paper](https://arxiv.org/abs/1707.06347) |

Please refer to [here](ClassicNas.md) for the usage of classic NAS algorithms.

## Supported One-shot NAS Algorithms

NNI currently supports the one-shot NAS algorithms listed below and is adding more. Users can reproduce an algorithm or use it on their own dataset. We also encourage users to implement other algorithms with [NNI API](#use-nni-api), to benefit more people.

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

## NAS Visualization

To help users track the process and status of how the model is searched under specified search space, we developed a visualization tool. It visualizes search space as a super-net and shows importance of subnets and layers/operations, as well as how the importance changes along with the search process. Please refer to [the document of NAS visualization](./Visualization.md) for how to use it.

## Reference and Feedback

* 在 GitHub 中[提交此功能的 Bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md)；
* 在 GitHub 中[提交新功能或改进请求](https://github.com/microsoft/nni/issues/new?template=enhancement.md)。
