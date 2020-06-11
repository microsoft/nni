# NAS 快速入门

NNI 提供的 NAS 功能有两个关键组件：用于表示搜索空间的 API 和 NAS 的训练方法。 前者为用户提供了表示性能好的模块的方法（即，通过搜索空间指定的候选模型）。 后者能让用户可以轻松的在自己的模型上使用最新的 NAS 训练方法。

这里，通过简单的示例，来一步步演示如何使用 NNI NAS API 调优自己的模型架构。 示例的完整代码可在[这里](https://github.com/microsoft/nni/tree/master/examples/nas/naive)找到。

## 使用 NAS API 编写模型

可通过两个 NAS API `LayerChoice` 和 `InputChoice` 来定义神经网络模型，而不需要编写具体的模型。 例如，如果认为在第一卷积层有两种操作可能会有效，可通过 `LayerChoice` 来为代码中的 `self.conv1` 赋值。 同样，第二个卷积层 `self.conv2` 也可以从中选择一个。 此处，指定了 4 个候选的神经网络。 `self.skipconnect` 使用 `InputChoice` 来指定两个选项，即是否添加跳跃的连接。

```python
import torch.nn as nn
from nni.nas.pytorch.mutables import LayerChoice, InputChoice

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = LayerChoice([nn.Conv2d(3, 6, 3, padding=1), nn.Conv2d(3, 6, 5, padding=2)])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = LayerChoice([nn.Conv2d(6, 16, 3, padding=1), nn.Conv2d(6, 16, 5, padding=2)])
        self.conv3 = nn.Conv2d(16, 16, 1)

        self.skipconnect = InputChoice(n_candidates=1)
        self.bn = nn.BatchNorm2d(16)

        self.gap = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
```

有关 `LayerChoice` 和 `InputChoice` 的详细描述可参考[ NAS 指南](NasGuide.md)。

## 选择 NAS Trainer

实例化模型后，需要通过 NAS Trainer 来训练模型。 不同的 Trainer 会使用不同的方法来从指定的神经网络模块中搜索出最好的。 NNI 提供了几种流行的 NAS 训练方法，如 DARTS，ENAS。 以下以 `DartsTrainer` 为例。 在 Trainer 实例化后，调用`trainer.train()` 开始搜索。

```python
trainer = DartsTrainer(net,
                       loss=criterion,
                       metrics=accuracy,
                       optimizer=optimizer,
                       num_epochs=2,
                       dataset_train=dataset_train,
                       dataset_valid=dataset_valid,
                       batch_size=64,
                       log_frequency=10)
trainer.train()
```

## 导出最佳模型

搜索（即`trainer.train()`）完成后，需要拿到最好的模型，只需要调用 `trainer.export("final_arch.json")` 来将找到的神经网络架构导出到文件。

## NAS 可视化

正在研究 NAS 的可视化，并将很快发布此功能。

## 重新训练导出的最佳模型

重新训练找到（导出）的网络架构非常容易。 第一步，实例化上面定义的模型。 第二步，在模型上调用 `apply_fixed_architecture`。 然后，此模型会作为找到的（导出的）模型。 之后，可以使用传统方法来训练此模型。

```python
model = Net()
apply_fixed_architecture(model, "final_arch.json")
```
