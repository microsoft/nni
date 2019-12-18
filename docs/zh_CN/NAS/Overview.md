# 神经网络结构搜索在 NNI 上的应用

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 [NASNet](https://arxiv.org/abs/1707.07012)，[ENAS](https://arxiv.org/abs/1802.03268)，[DARTS](https://arxiv.org/abs/1806.09055)，[Network Morphism](https://arxiv.org/abs/1806.10282)，以及 [Evolution](https://arxiv.org/abs/1703.01041) 等。 新的算法还在不断涌现。

但是，要实现NAS算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。 为了促进 NAS 创新（例如，设计、实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

以此为动力，NNI 的目标是提供统一的体系结构，以加速NAS上的创新，并将最新的算法更快地应用于现实世界中的问题上。

通过[统一的接口](./NasInterface.md)，有两种方式进行架构搜索。 [第一种](#supported-one-shot-nas-algorithms)称为 one-shot NAS，基于搜索空间构建了一个超级网络，并使用 one-shot 训练来生成性能良好的子模型。 [第二种](./NasInterface.md#classic-distributed-search)是传统的搜索方法，搜索空间中每个子模型作为独立的 Trial 运行，将性能结果发给 Tuner，由 Tuner 来生成新的子模型。

* [支持的 One-shot NAS 算法](#supported-one-shot-nas-algorithms)
* [使用 NNI Experiment 的经典分布式 NAS](./NasInterface.md#classic-distributed-search)
* [NNI NAS 编程接口](./NasInterface.md)

## 支持的 One-shot NAS 算法

NNI 现在支持以下 NAS 算法，并且正在添加更多算法。 用户可以重现算法或在自己的数据集上使用它。 鼓励用户使用 [NNI API](#use-nni-api) 实现其它算法，以使更多人受益。

| 名称                  | 算法简介                                                                                                                                          |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| [ENAS](#enas)       | Efficient Neural Architecture Search via Parameter Sharing [参考论文](https://arxiv.org/abs/1802.03268)                                           |
| [DARTS](#darts)     | DARTS: Differentiable Architecture Search [参考论文](https://arxiv.org/abs/1806.09055)                                                            |
| [P-DARTS](#p-darts) | Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation [参考论文](https://arxiv.org/abs/1904.12760) |

注意，这些算法**不需要 nnictl**，独立运行，仅支持 PyTorch。 将来的版本会支持 Tensorflow 2.0。

### 依赖项

* NNI 1.2+
* tensorboard
* PyTorch 1.2+
* git

### ENAS

[Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268). 在 ENAS 中，Contoller 学习在大的计算图中搜索最有子图的方式来发现神经网络。 它通过在子模型间共享参数来实现加速和出色的性能指标。

#### 用法

NNI 中的 ENAS 还在开发中，当前仅支持在 CIFAR10 上 Macro/Micro 搜索空间的搜索阶段。 在 PTB 上从头开始训练及其搜索空间尚未完成。 [详细说明](ENAS.md)。

```bash
＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
git clone https://github.com/Microsoft/nni.git

# 搜索最好的网络架构
cd examples/nas/enas

# 在 Macro 搜索空间中搜索
python3 search.py --search-for macro

# 在 Micro 搜索空间中搜索
python3 search.py --search-for micro

# 查看更多选项
python3 search.py -h
```

### DARTS

[DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) 在算法上的主要贡献是，引入了一种在两级网络优化中使用的可微分算法。 [详细说明](DARTS.md)。

#### 用法

```bash
＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
git clone https://github.com/Microsoft/nni.git

# 搜索最好的架构
cd examples/nas/darts
python3 search.py

# 训练最好的架构
python3 retrain.py --arc-checkpoint ./checkpoints/epoch_49.json
```

### P-DARTS

[Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) 基于 [DARTS](#DARTS)。 它在算法上的主要贡献是引入了一种有效的算法，可在搜索过程中逐渐增加搜索的深度。

#### 用法

```bash
＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
git clone https://github.com/Microsoft/nni.git

# 搜索最好的架构
cd examples/nas/pdarts
python3 search.py

# 训练最好的架构，过程与 darts 相同。
cd ../darts
python3 retrain.py --arc-checkpoint ../pdarts/checkpoints/epoch_2.json
```

## 使用 NNI API

注意，我们正在尝试通过统一的编程接口来支持各种 NAS 算法，当前处于试验阶段。 这意味着当前编程接口将来会有变化。

### 编程接口

在两种场景下需要用于设计和搜索模型的编程接口。

1. 在设计神经网络时，可能在层、子模型或连接上有多种选择，并且无法确定是其中一种或某些的组合的结果最好。 因此，需要简单的方法来表达候选的层或子模型。
2. 在神经网络上应用 NAS 时，需要统一的方式来表达架构的搜索空间，这样不必为不同的搜索算法来更改代码。

NNI 提出的 API 在[这里](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch)。 [这里](https://github.com/microsoft/nni/tree/master/examples/nas/darts)包含了基于此 API 的 NAS 实现示例。
