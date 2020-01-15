# Neural Architecture Search (NAS) on NNI

Automatic neural architecture search is taking an increasingly important role on finding better models. Recent research works have proved the feasibility of automatic NAS, and also found some models that could beat manually designed and tuned models. Some of representative works are [NASNet][2], [ENAS][1], [DARTS][3], [Network Morphism][4], and [Evolution][5]. There are new innovations keeping emerging.

However, it takes great efforts to implement NAS algorithms, and it is hard to reuse code base of existing algorithms in new one. To facilitate NAS innovations (e.g., design and implement new NAS models, compare different NAS models side-by-side), an easy-to-use and flexible programming interface is crucial.

With this motivation, our ambition is to provide a unified architecture in NNI, to accelerate innovations on NAS, and apply state-of-art algorithms on real world problems faster.

With [the unified interface](./NasInterface.md), there are two different modes for the architecture search. [One](#supported-one-shot-nas-algorithms) is the so-called one-shot NAS, where a super-net is built based on search space, and using one shot training to generate good-performing child model. [The other](./NasInterface.md#classic-distributed-search) is the traditional searching approach, where each child model in search space runs as an independent trial, the performance result is sent to tuner and the tuner generates new child model.

* [Supported One-shot NAS Algorithms](#supported-one-shot-nas-algorithms)
* [Classic Distributed NAS with NNI experiment](./NasInterface.md#classic-distributed-search)
* [NNI NAS Programming Interface](./NasInterface.md)

## Supported One-shot NAS Algorithms

NNI supports below NAS algorithms now and is adding more. User can reproduce an algorithm or use it on their own dataset. We also encourage users to implement other algorithms with [NNI API](#use-nni-api), to benefit more people.

|Name|Brief Introduction of Algorithm|
|---|---|
| [ENAS](ENAS.md) | [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268). In ENAS, a controller learns to discover neural network architectures by searching for an optimal subgraph within a large computational graph. It uses parameter sharing between child models to achieve fast speed and excellent performance. |
| [DARTS](DARTS.md) | [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) introduces a novel algorithm for differentiable network architecture search on bilevel optimization. |
| [P-DARTS](PDARTS.md) | [Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) is based on DARTS. It introduces an efficient algorithm which allows the depth of searched architectures to grow gradually during the training procedure. |
| [SPOS](SPOS.md) | [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420) constructs a simplified supernet trained with an uniform path sampling method, and applies an evolutionary algorithm to efficiently search for the best-performing architectures. |
| [CDARTS](CDARTS.md) | [Cyclic Differentiable Architecture Search](https://arxiv.org/abs/****) builds a cyclic feedback mechanism between the search and evaluation networks. It introduces a cyclic differentiable architecture search framework which integrates the two networks into a unified architecture.|

One-shot algorithms run **standalone without nnictl**. Only PyTorch version has been implemented. Tensorflow 2.x will be supported in future release.

Here are some common dependencies to run the examples. PyTorch needs to be above 1.2 to use ``BoolTensor``.

* NNI 1.2+
* tensorboard
* PyTorch 1.2+
* git

## Use NNI API

NOTE, we are trying to support various NAS algorithms with unified programming interface, and it's in very experimental stage. It means the current programing interface may be updated in future.

### Programming interface

The programming interface of designing and searching a model is often demanded in two scenarios.

1. When designing a neural network, there may be multiple operation choices on a layer, sub-model, or connection, and it's undetermined which one or combination performs  best. So, it needs an easy way to express the candidate layers or sub-models.
2. When applying NAS on a neural network, it needs an unified way to express the search space of architectures, so that it doesn't need to update trial code for different searching algorithms.

NNI proposed API is [here](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch). And [here](https://github.com/microsoft/nni/tree/master/examples/nas/naive) is an example of NAS implementation, which bases on NNI proposed interface.

[1]: https://arxiv.org/abs/1802.03268
[2]: https://arxiv.org/abs/1707.07012
[3]: https://arxiv.org/abs/1806.09055
[4]: https://arxiv.org/abs/1806.10282
[5]: https://arxiv.org/abs/1703.01041

## **Reference and Feedback**
* To [report a bug](https://github.com/microsoft/nni/issues/new?template=bug-report.md) for this feature in GitHub;
* To [file a feature or improvement request](https://github.com/microsoft/nni/issues/new?template=enhancement.md) for this feature in GitHub;
* To know more about [Feature Engineering with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/FeatureEngineering/Overview.md);
* To know more about [Model Compression with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/Compressor/Overview.md);
* To know more about [Hyperparameter Tuning with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/Tuner/BuiltinTuner.md);
