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

[Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760) bases on [DARTS](#DARTS). It's contribution on algorithm is to introduce an efficient algorithm which allows the depth of searched architectures to grow gradually during the training procedure.

#### Usage

```bash
# In case NNI code is not cloned. If the code is cloned already, ignore this line and enter code folder.
git clone https://github.com/Microsoft/nni.git

# search the best architecture
cd examples/nas/pdarts
python3 search.py

# train the best architecture, it's the same progress as darts.
cd examples/nas/darts
python3 retrain.py --arc-checkpoint ./checkpoints/epoch_2.json
```

## Use NNI API

NOTE, we are trying to support various NAS algorithms with unified programming interface, and it's in very experimental stage. It means the current programing interface may be updated significantly.

*previous [NAS annotation](../AdvancedFeature/GeneralNasInterfaces.md) interface will be deprecated soon.*

### Programming interface

The programming interface of designing and searching a model is often demanded in two scenarios.

1. When designing a neural network, there may be multiple operation choices on a layer, sub-model, or connection, and it's undetermined which one or combination performs  best. So, it needs an easy way to express the candidate layers or sub-models.
2. When applying NAS on a neural network, it needs an unified way to express the search space of architectures, so that it doesn't need to update trial code for different searching algorithms.

NNI proposed API is [here](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch). And [here](https://github.com/microsoft/nni/tree/master/examples/nas/darts) is an example of NAS implementation, which bases on NNI proposed interface.
