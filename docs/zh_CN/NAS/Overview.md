# 神经网络结构搜索在 NNI 上的应用

Automatic neural architecture search is taking an increasingly important role in finding better models. Recent research has proved the feasibility of automatic NAS and has lead to models that beat many manually designed and tuned models. Some representative works are [NASNet](https://arxiv.org/abs/1707.07012), [ENAS](https://arxiv.org/abs/1802.03268), [DARTS](https://arxiv.org/abs/1806.09055), [Network Morphism](https://arxiv.org/abs/1806.10282), and [Evolution](https://arxiv.org/abs/1703.01041). Further, new innovations keep emerging.

However, it takes a great effort to implement NAS algorithms, and it's hard to reuse the code base of existing algorithms for new ones. To facilitate NAS innovations (e.g., the design and implementation of new NAS models, the comparison of different NAS models side-by-side, etc.), an easy-to-use and flexible programming interface is crucial.

With this motivation, our ambition is to provide a unified architecture in NNI, accelerate innovations on NAS, and apply state-of-the-art algorithms to real-world problems faster.

With the unified interface, there are two different modes for architecture search. [One](#supported-one-shot-nas-algorithms) is the so-called one-shot NAS, where a super-net is built based on a search space and one-shot training is used to generate a good-performing child model. [The other](#supported-distributed-nas-algorithms) is the traditional search-based approach, where each child model within the search space runs as an independent trial. The performance result is then sent to Tuner and the tuner generates a new child model.

## 支持的 One-shot NAS 算法

NNI currently supports the NAS algorithms listed below and is adding more. Users can reproduce an algorithm or use it on their own dataset. 鼓励用户使用 [NNI API](#use-nni-api) 实现其它算法，以使更多人受益。

| 名称                              | 算法简介                                                                                                                                                                                                                                                                                      |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ENAS](ENAS.md)                 | [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268). 在 ENAS 中，Contoller 学习在大的计算图中搜索最有子图的方式来发现神经网络。 它通过在子模型间共享参数来实现加速和出色的性能指标。                                                                                                                  |
| [DARTS](DARTS.md)               | [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) 引入了一种在两级网络优化中使用的可微分算法。                                                                                                                                                                                      |
| [P-DARTS](PDARTS.md)            | [Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) 基于DARTS。 它引入了一种有效的算法，可在搜索过程中逐渐增加搜索的深度。                                                                                                           |
| [SPOS](SPOS.md)                 | [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) constructs a simplified supernet trained with a uniform path sampling method and applies an evolutionary algorithm to efficiently search for the best-performing architectures. |
| [CDARTS](CDARTS.md)             | [Cyclic Differentiable Architecture Search](https://arxiv.org/abs/****) 在搜索和评估的网络见构建了循环反馈的机制。 通过引入的循环的可微分架构搜索框架将两个网络集成为一个架构。                                                                                                                                                              |
| [ProxylessNAS](Proxylessnas.md) | [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332).                                                                                                                                                                          |

One-shot 算法**不需要 nnictl，可单独运行**。 Only the PyTorch version has been implemented. Tensorflow 2.x will be supported in a future release.

这是运行示例的一些常见依赖项。 PyTorch 需要高于 1.2 才能使用 `BoolTensor`.

* NNI 1.2+
* tensorboard
* PyTorch 1.2+
* git

## 支持的分布式 NAS 算法

| 名称                    | 算法简介                                                                                                                                                                                                                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [SPOS 的第二阶段](SPOS.md) | [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) constructs a simplified supernet trained with a uniform path sampling method, and applies an evolutionary algorithm to efficiently search for the best-performing architectures. |

```eval_rst 
.. Note:: SPOS is a two-stage algorithm, whose first stage is one-shot and the second stage is distributed, leveraging the result of the first stage as a checkpoint.   
```

## Using the NNI API

在两种场景下需要用于设计和搜索模型的编程接口。

1. When designing a neural network, there may be multiple operation choices on a layer, sub-model, or connection, and it's undetermined which one or combination performs best. 因此，需要简单的方法来表达候选的层或子模型。
2. When applying NAS on a neural network, it needs a unified way to express the search space of architectures, so that it doesn't need to update trial code for different search algorithms.

[Here](./NasGuide.md) is the user guide to get started with using NAS on NNI.

## 参考和反馈

* 在 GitHub 中[提交此功能的 Bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md)；
* 在 GitHub 中[提交新功能或改进请求](https://github.com/microsoft/nni/issues/new?template=enhancement.md)。
