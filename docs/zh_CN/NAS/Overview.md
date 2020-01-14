# 神经网络结构搜索在 NNI 上的应用

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 [NASNet](https://arxiv.org/abs/1707.07012)，[ENAS](https://arxiv.org/abs/1802.03268)，[DARTS](https://arxiv.org/abs/1806.09055)，[Network Morphism](https://arxiv.org/abs/1806.10282)，以及 [Evolution](https://arxiv.org/abs/1703.01041) 等。 新的算法还在不断涌现。

但是，要实现NAS算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。 为了促进 NAS 创新（例如，设计、实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

以此为动力，NNI 的目标是提供统一的体系结构，以加速NAS上的创新，并将最新的算法更快地应用于现实世界中的问题上。

通过[统一的接口](./NasInterface.md)，有两种方式进行架构搜索。 [One](#supported-one-shot-nas-algorithms) is the so-called one-shot NAS, where a super-net is built based on search space, and using one shot training to generate good-performing child model. [第二种](./NasInterface.md#classic-distributed-search)是传统的搜索方法，搜索空间中每个子模型作为独立的 Trial 运行，将性能结果发给 Tuner，由 Tuner 来生成新的子模型。

* [支持的 One-shot NAS 算法](#supported-one-shot-nas-algorithms)
* [使用 NNI Experiment 的经典分布式 NAS](./NasInterface.md#classic-distributed-search)
* [NNI NAS 编程接口](./NasInterface.md)

## 支持的 One-shot NAS 算法

NNI supports below NAS algorithms now and is adding more. User can reproduce an algorithm or use it on their own dataset. We also encourage users to implement other algorithms with [NNI API](#use-nni-api), to benefit more people.

| 名称                   | 算法简介                                                                                                                                                                                                                                                                                                                                           |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ENAS](ENAS.md)      | [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268). In ENAS, a controller learns to discover neural network architectures by searching for an optimal subgraph within a large computational graph. It uses parameter sharing between child models to achieve fast speed and excellent performance. |
| [DARTS](DARTS.md)    | [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) introduces a novel algorithm for differentiable network architecture search on bilevel optimization.                                                                                                                                                             |
| [P-DARTS](PDARTS.md) | [Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) is based on DARTS. It introduces an efficient algorithm which allows the depth of searched architectures to grow gradually during the training procedure.                                             |
| [SPOS](SPOS.md)      | [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) constructs a simplified supernet trained with an uniform path sampling method, and applies an evolutionary algorithm to efficiently search for the best-performing architectures.                                                    |

One-shot algorithms run **standalone without nnictl**. Only PyTorch version has been implemented. Tensorflow 2.x will be supported in future release.

Here are some common dependencies to run the examples. PyTorch needs to be above 1.2 to use `BoolTensor`.

* NNI 1.2+
* tensorboard
* PyTorch 1.2+
* git

## 使用 NNI API

NOTE, we are trying to support various NAS algorithms with unified programming interface, and it's in very experimental stage. It means the current programing interface may be updated in future.

### Programming interface

The programming interface of designing and searching a model is often demanded in two scenarios.

1. 在设计神经网络时，可能在层、子模型或连接上有多种选择，并且无法确定是其中一种或某些的组合的结果最好。 因此，需要简单的方法来表达候选的层或子模型。
2. 在神经网络上应用 NAS 时，需要统一的方式来表达架构的搜索空间，这样不必为不同的搜索算法来更改代码。

NNI proposed API is [here](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch). And [here](https://github.com/microsoft/nni/tree/master/examples/nas/naive) is an example of NAS implementation, which bases on NNI proposed interface.

## **参考和反馈**
* 在 GitHub 中[提交此功能的 Bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md)；
* 在 GitHub 中[提交新功能或改进请求](https://github.com/microsoft/nni/issues/new?template=enhancement.md)；
* 了解 NNI 中[特征工程的更多信息](https://github.com/microsoft/nni/blob/master/docs/zh_CN/FeatureEngineering/Overview.md)；
* 了解 NNI 中[模型自动压缩的更多信息](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Compressor/Overview.md)；
* 了解如何[使用 NNI 进行超参数调优](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Tuner/BuiltinTuner.md)；
