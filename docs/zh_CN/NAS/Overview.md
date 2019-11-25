# 神经网络结构搜索在 NNI 上的应用

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 [NASNet](https://arxiv.org/abs/1707.07012)，[ENAS](https://arxiv.org/abs/1802.03268)，[DARTS](https://arxiv.org/abs/1806.09055)，[Network Morphism](https://arxiv.org/abs/1806.10282)，以及 [Evolution](https://arxiv.org/abs/1703.01041) 等。 新的算法还在不断涌现。

但是，要实现NAS算法需要花费大量的精力，并且很难在新算法中重用现有算法的代码。 为了促进 NAS 创新（例如，设计、实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

以此为动力，NNI 的目标是提供统一的体系结构，以加速NAS上的创新，并将最新的算法更快地应用于现实世界中的问题上。

## 支持的算法

NNI 现在支持以下 NAS 算法，并且正在添加更多算法。 用户可以重现算法或在自己的数据集上使用它。 鼓励用户使用 [NNI API](#use-nni-api) 实现其它算法，以使更多人受益。

请注意，这些算法在不需要 nnictl 的情况下独立运行，并且仅支持 PyTorch。

### 依赖项

* 安装最新的 NNI
* PyTorch 1.2+
* git

### DARTS

[DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) 在算法上的主要贡献是，引入了一种在两级网络优化中使用的可微分算法。

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
cd examples/nas/darts
python3 retrain.py --arc-checkpoint ./checkpoints/epoch_2.json
```

## 使用 NNI API

注意，我们正在尝试通过统一的编程接口来支持各种 NAS 算法，当前处于试验阶段。 这意味着当前编程接口可能会进行重大变化。

*先前的 [NAS annotation](../AdvancedFeature/GeneralNasInterfaces.md) 接口会很快被弃用。*

### 编程接口

在两种场景下需要用于设计和搜索模型的编程接口。

1. 在设计神经网络时，可能在层、子模型或连接上有多种选择，并且无法确定是其中一种或某些的组合的结果最好。 因此，需要简单的方法来表达候选的层或子模型。
2. 在神经网络上应用 NAS 时，需要统一的方式来表达架构的搜索空间，这样不必为不同的搜索算法来更改代码。

NNI 提出的 API 在[这里](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch)。 [这里](https://github.com/microsoft/nni/tree/master/examples/nas/darts)包含了基于此 API 的 NAS 实现示例。
