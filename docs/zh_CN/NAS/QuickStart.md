# NAS 快速入门

NNI 提供的 NAS 功能有两个关键组件：用于表示搜索空间的 API 和 NAS 的训练方法。 前者为用户提供了表示性能好的模块的方法（即，通过搜索空间指定的候选模型）。 后者能让用户可以轻松的在自己的模型上使用最新的 NAS 训练方法。

这里，通过简单的示例，来一步步演示如何使用 NNI NAS API 调优自己的模型架构。 示例的完整代码可在[这里](https://github.com/microsoft/nni/tree/master/examples/nas/naive)找到。

## 使用 NAS API 编写模型

Instead of writing a concrete neural model, you can write a class of neural models using two NAS APIs `LayerChoice` and `InputChoice`. For example, you think either of two operations might work in the first convolution layer, then you can get one from them using `LayerChoice` as shown by `self.conv1` in the code. Similarly, the second convolution layer `self.conv2` also chooses one from two operations. To this line, four candidate neural networks are specified. `self.skipconnect` uses `InputChoice` to specify two choices, i.e., adding skip connection or not.

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

For detailed description of `LayerChoice` and `InputChoice`, please refer to [the guidance](NasGuide.md)

## Choose a NAS trainer

After the model is instantiated, it is time to train the model using NAS trainer. Different trainers use different approaches to search for the best one from a class of neural models that you specified. NNI provides popular NAS training approaches, such as DARTS, ENAS. Here we use `DartsTrainer` as an example below. After the trainer is instantiated, invoke `trainer.train()` to do the search.

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

## Export the best model

After the search (i.e., `trainer.train()`) is done, we want to get the best performing model, then simply call `trainer.export("final_arch.json")` to export the found neural architecture to a file.

## NAS visualization

We are working on visualization of NAS and will release soon.

## Retrain the exported best model

It is simple to retrain the found (exported) neural architecture. Step one, instantiate the model you defined above. Step two, invoke `apply_fixed_architecture` on the model. Then the model becomes the found (exported) one, you can use traditional model training to train this model.

```python
model = Net()
apply_fixed_architecture(model, "final_arch.json")
```
