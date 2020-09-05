# 编写搜索空间

通常，搜索空间是要在其中找到最好结构的候选项。 无论是经典 NAS 还是 One-Shot NAS，不同的搜索算法都需要搜索空间。 NNI 提供了统一的 API 来表达神经网络架构的搜索空间。

搜索空间可基于基础模型来构造。 这也是在已有模型上使用 NAS 的常用方法。 以 [PyTorch 上的 MNIST](https://github.com/pytorch/examples/blob/master/mnist/main.py) 为例。 注意，NNI 为 PyTorch 和 TensorFlow 提供了同样的搜索空间 API。

```python
from nni.nas.pytorch import mutables

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = mutables.LayerChoice([
            nn.Conv2d(1, 32, 3, 1),
            nn.Conv2d(1, 32, 5, 3)
        ])  # 尝试 3x3 和 5x5 的核
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # ... 与原始代码一样 ...
        return output
```

以上示例在 conv1 上添加了 conv5x5 的选项。 修改非常简单，只需要声明 `LayerChoice` 并将原始的 conv3x3 和新的 conv5x5 作为参数即可。 就这么简单！ 不需要修改 forward 函数。 可将 conv1 想象为没有 NAS 的模型。

如何表示可能的连接？ 通过 `InputChoice` 来实现。 要在 MNIST 示例上使用跳过连接，需要增加另一层 conv3。 下面的示例中，从 conv2 的可能连接加入到了 conv3 的输出中。

```python
from nni.nas.pytorch import mutables

class Net(nn.Module):
    def __init__(self):
        # ... 相同 ...
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 1, 1)
        # 声明只从搜索策略中选择一个或零个候选项
        self.skipcon = mutables.InputChoice(n_candidates=1)
        # ... 相同 ...

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x0 = self.skipcon([x])  # 从 [x] 中选择一个或 None
        x = self.conv3(x)
        if x0 is not None:  # 跳接可用
            x += x0
        x = F.max_pool2d(x, 2)
        # ... 相同 ...
        return output
```

Input Choice 可被视为可调用的模块，它接收张量数组，输出其中部分的连接、求和、平均（默认为求和），或没有选择时输出 `None`。 与 Layer Choice 一样，Input Choice 要**在 `__init__` 中初始化，并在 `forward` 中调用。 这会让搜索算法找到这些 Choice，并进行所需的准备。</p>

`LayerChoice` 和 `InputChoice` 都是 **Mutable**。 Mutable 表示 "可变化的"。 与传统深度学习层、模型都是固定的不同，使用 Mutable 的模块，是一组可能选择的模型。

用户可为每个 Mutable 指定 **key**。 默认情况下，NNI 会分配全局唯一的，但如果需要共享 Choice（例如，两个 `LayerChoice` 有同样的候选操作，希望共享同样的 Choice。即，如果一个选择了第 i 个操作，第二个也要选择第 i 个操作），那么就应该给它们相同的 key。 key 标记了此 Choice，并会在存储的检查点中使用。 如果要增加导出架构的可读性，可为每个 Mutable 的 key 指派名称。 Mutable 高级用法（如，`LayerChoice` 和 `InputChoice`），参考 [Mutables](./NasReference.md)。

定义了搜索空间后，下一步是从中找到最好的模型。 参考 [经典 NAS 算法](./ClassicNas.md)和 [One-Shot NAS 算法](./NasGuide.md)来查看如何从定义的搜索空间中进行搜索。