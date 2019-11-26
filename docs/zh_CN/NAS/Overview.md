# 神经网络结构搜索在 NNI 上的应用

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 [NASNet](https://arxiv.org/abs/1707.07012)，[ENAS](https://arxiv.org/abs/1802.03268)，[DARTS](https://arxiv.org/abs/1806.09055)，[Network Morphism](https://arxiv.org/abs/1806.10282)，以及 [Evolution](https://arxiv.org/abs/1703.01041) 等。 新的算法还在不断涌现。

但是，要实现NAS算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。 为了促进 NAS 创新（例如，设计、实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

以此为动力，NNI 的目标是提供统一的体系结构，以加速NAS上的创新，并将最新的算法更快地应用于现实世界中的问题上。

With [the unified interface](.NasInterface.md), there are two different modes for the architecture search. [The one](#supported-one-shot-nas-algorithms) is the so-called one-shot NAS, where a super-net is built based on search space, and using one shot training to generate good-performing child model. [The other](.ClassicNas.md) is the traditional searching approach, where each child model in search space runs as an independent trial, the performance result is sent to tuner and the tuner generates new child model.

* [Supported One-shot NAS Algorithms](#supported-one-shot-nas-algorithms)
* [Classic Distributed NAS with NNI experiment](.NasInterface.md#classic-distributed-search)
* [NNI NAS Programming Interface](.NasInterface.md)

## Supported One-shot NAS Algorithms

NNI supports below NAS algorithms now and being adding more. User can reproduce an algorithm or use it on owned dataset. we also encourage user to implement other algorithms with [NNI API](#use-nni-api), to benefit more people.

| Name                | Brief Introduction of Algorithm                                                                                                                          |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ENAS](#enas)       | Efficient Neural Architecture Search via Parameter Sharing [Reference Paper](https://arxiv.org/abs/1802.03268)                                           |
| [DARTS](#darts)     | DARTS: Differentiable Architecture Search [Reference Paper](https://arxiv.org/abs/1806.09055)                                                            |
| [P-DARTS](#p-darts) | Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation [Reference Paper](https://arxiv.org/abs/1904.12760) |

Note, these algorithms run **standalone without nnictl**, and supports PyTorch only. Tensorflow 2.0 will be supported in future release.

### 依赖项

* NNI 1.2+
* tensorboard
* PyTorch 1.2+
* git

### ENAS

[Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268). In ENAS, a controller learns to discover neural network architectures by searching for an optimal subgraph within a large computational graph. It uses parameter sharing between child models to achieve fast speed and excellent performance.

#### 用法

ENAS in NNI is still under development and we only support search phase for macro/micro search space on CIFAR10. Training from scratch and search space on PTB has not been finished yet.

```bash
＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/enas

# search in macro search space
python3 search.py --search-for macro

# search in micro search space
python3 search.py --search-for micro

# view more options for search
python3 search.py -h
```

### DARTS

The main contribution of [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) on algorithm is to introduce a novel algorithm for differentiable network architecture search on bilevel optimization.

#### 用法

```bash
＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/darts
python3 search.py

# train the best architecture
python3 retrain.py --arc-checkpoint ./checkpoints/epoch_49.json
```

### P-DARTS

[Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) bases on [DARTS](#DARTS). It's contribution on algorithm is to introduce an efficient algorithm which allows the depth of searched architectures to grow gradually during the training procedure.

#### Usage

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/pdarts
python3 search.py

# train the best architecture, it's the same progress as darts.
cd ../darts
python3 retrain.py --arc-checkpoint ../pdarts/checkpoints/epoch_2.json
```

## 使用 NNI API

NOTE, we are trying to support various NAS algorithms with unified programming interface, and it's in very experimental stage. It means the current programing interface may be updated in future.

*previous [NAS annotation](../AdvancedFeature/GeneralNasInterfaces.md) interface will be deprecated soon.*

### Programming interface

The programming interface of designing and searching a model is often demanded in two scenarios.

1. 在设计神经网络时，可能在层、子模型或连接上有多种选择，并且无法确定是其中一种或某些的组合的结果最好。 因此，需要简单的方法来表达候选的层或子模型。
2. 在神经网络上应用 NAS 时，需要统一的方式来表达架构的搜索空间，这样不必为不同的搜索算法来更改代码。

NNI proposed API is [here](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch). And [here](https://github.com/microsoft/nni/tree/master/examples/nas/darts) is an example of NAS implementation, which bases on NNI proposed interface.
