# 指南：在 NNI 上使用 NAS

```eval_rst
.. contents::

.. Note:: 此 API 初始试验阶段。 当前接口可能会更改。
```

![](../../img/nas_abstract_illustration.png)

现代神经架构搜索（NAS）方法通常包含 [三个维度](https://arxiv.org/abs/1808.05377)：搜索空间、搜索策略和性能估计策略。 搜索空间通常是要搜索的一个有限的神经网络架构，而搜索策略会采样来自搜索空间的架构，评估性能，并不断演进。 理想情况下，搜索策略会找到搜索空间中最好的架构，并返回给用户。 在获得了 "最好架构" 后，很多方法都会有 "重新训练" 的步骤，会像普通神经网络模型一样训练。

## 实现搜索空间

假设已经有了基础的模型，该如何使用 NAS 来提升？ 以 [PyTorch 上的 MNIST](https://github.com/pytorch/examples/blob/master/mnist/main.py) 为例，代码如下：

```python
from nni.nas.pytorch import mutables

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = mutables.LayerChoice([
            nn.Conv2d(1, 32, 3, 1),
            nn.Conv2d(1, 32, 5, 3)
        ])  # try 3x3 kernel and 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # ... 与原始的一样 ...
        返回输出
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
        # 声明搜索策略，来选择最多一个选项
        self.skipcon = mutables.InputChoice(n_candidates=1)
        # ... 相同 ...

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x0 = self.skipcon([x])  # 从 [x] 中选择 0 或 1 个
        x = self.conv3(x)
        if x0 is not None:  # 允许跳过连接
            x += x0
        x = F.max_pool2d(x, 2)
        # ... 相同 ...
        返回输出
```

Input Choice 可被视为可调用的模块，它接收张量数组，输出其中部分的连接、求和、平均（默认为求和），或没有选择时输出 `None`。 与 Layer Choice 一样，Input Choice 要**在 `__init__` 中初始化，并在 `forward` 中调用。 稍后的例子中会看到搜索算法如何识别这些 Choice，并进行相应的准备。</p>

`LayerChoice` 和 `InputChoice` 都是 **Mutable**。 Mutable 表示 "可变化的"。 与传统深度学习层、模型都是固定的不同，使用 Mutable 的模块，是一组可能选择的模型。

用户可为每个 Mutable 指定 **key**。 默认情况下，NNI 会分配全局唯一的，但如果需要共享 Choice（例如，两个 `LayerChoice` 有同样的候选操作，希望共享同样的 Choice。即，如果一个选择了第 i 个操作，第二个也要选择第 i 个操作），那么就应该给它们相同的 key。 key 标记了此 Choice，并会在存储的检查点中使用。 如果要增加导出架构的可读性，可为每个 Mutable 的 key 指派名称。 高级用法参考 [Mutable](./NasReference.md#mutables)。

## 使用搜索算法

搜索空间的探索方式和 Trial 生成方式不同，至少有两种不同的方法用来搜索。 一种是分布式运行 NAS，可从头枚举运行所有架构。或者利用更多高级功能，如 [SMASH](https://arxiv.org/abs/1708.05344), [ENAS](https://arxiv.org/abs/1802.03268), [DARTS](https://arxiv.org/abs/1808.05377), [FBNet](https://arxiv.org/abs/1812.03443), [ProxylessNAS](https://arxiv.org/abs/1812.00332), [SPOS](https://arxiv.org/abs/1904.00420), [Single-Path NAS](https://arxiv.org/abs/1904.02877),  [Understanding One-shot](http://proceedings.mlr.press/v80/bender18a) 以及 [GDAS](https://arxiv.org/abs/1910.04465)。 由于很多不同架构搜索起来成本较高，另一类方法，即 One-Shot NAS，在搜索空间中，构建包含有所有候选网络的超网络，每一步中选择一个或几个子网络来训练。

当前，NNI 支持数种 One-Shot 方法。 例如，`DartsTrainer` 使用 SGD 来交替训练架构和模型权重，`ENASTrainer` [使用 Controller 来训练模型](https://arxiv.org/abs/1802.03268)。 新的、更高效的 NAS Trainer 在研究界不断的涌现出来。

### One-Shot NAS

每个 One-Shot NAS 都实现了 Trainer，可在每种算法说明中找到详细信息。 这是如何使用 `EnasTrainer` 的简单示例。

```python
# 此处与普通模型训练相同
model = Net()
dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)

# 使用 NAS
def top1_accuracy(output, target):
    # ENAS 使用此函数来计算奖励
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size

def metrics_fn(output, target):
    # 指标函数接收输出和目标，并计算出指标 dict
    return {"acc1": reward_accuracy(output, target)}

from nni.nas.pytorch import enas
trainer = enas.EnasTrainer(model,
                           loss=criterion,
                           metrics=metrics_fn,
                           reward_function=top1_accuracy,
                           optimizer=optimizer,
                           batch_size=128
                           num_epochs=10,  # 10 epochs
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid,
                           log_frequency=10)  # 每 10 步打印
trainer.train()  # 训练
trainer.export(file="model_dir/final_architecture.json")  # 将最终架构导出到文件
```

用户可直接通过 `python3 train.py` 开始训练，不需要使用 `nnictl`。 训练完成后，可通过 `trainer.export()` 导出找到的最好的模型。

通常，Trainer 会有些可定制的参数，例如，损失函数，指标函数，优化器，以及数据集。 These should satisfy the needs from most usages, and we do our best to make sure our built-in trainers work on as many models, tasks and datasets as possible. But there is no guarantee. For example, some trainers have assumption that the task has to be a classification task; some trainers might have a different definition of "epoch" (e.g., an ENAS epoch = some child steps + some controller steps); most trainers do not have support for distributed training: they won't wrap your model with `DataParallel` or `DistributedDataParallel` to do that. So after a few tryouts, if you want to actually use the trainers on your very customized applications, you might very soon need to [customize your trainer](#extend-the-ability-of-one-shot-trainers).

### Distributed NAS

Neural architecture search is originally executed by running each child model independently as a trial job. We also support this searching approach, and it naturally fits in NNI hyper-parameter tuning framework, where tuner generates child model for next trial and trials run in training service.

To use this mode, there is no need to change the search space expressed with NNI NAS API (i.e., `LayerChoice`, `InputChoice`, `MutableScope`). After the model is initialized, apply the function `get_and_apply_next_architecture` on the model. One-shot NAS trainers are not used in this mode. Here is a simple example:

```python
model = Net()

# get the chosen architecture from tuner and apply it on model
get_and_apply_next_architecture(model)
train(model)  # your code for training the model
acc = test(model)  # test the trained model
nni.report_final_result(acc)  # report the performance of the chosen architecture
```

The search space should be generated and sent to tuner. As with NNI NAS API the search space is embedded in user code, users could use "[nnictl ss_gen](../Tutorial/Nnictl.md)" to generate search space file. Then, put the path of the generated search space in the field `searchSpacePath` of `config.yml`. The other fields in `config.yml` can be filled by referring [this tutorial](../Tutorial/QuickStart.md).

You could use [NNI tuners](../Tuner/BuiltinTuner.md) to do the search. Currently, only PPO Tuner supports NAS search space.

We support standalone mode for easy debugging, where you could directly run the trial command without launching an NNI experiment. This is for checking whether your trial code can correctly run. The first candidate(s) are chosen for `LayerChoice` and `InputChoice` in this standalone mode.

A complete example can be found [here](https://github.com/microsoft/nni/tree/master/examples/nas/classic_nas/config_nas.yml).

### Retrain with Exported Architecture

After the searching phase, it's time to train the architecture found. Unlike many open-source NAS algorithms who write a whole new model specifically for retraining. We found that searching model and retraining model are usual very similar, and therefore you can construct your final model with the exact model code. For example

```python
model = Net()
apply_fixed_architecture(model, "model_dir/final_architecture.json")
```

The JSON is simply a mapping from mutable keys to one-hot or multi-hot representation of choices. For example

```json
{
    "LayerChoice1": [false, true, false, false],
    "InputChoice2": [true, true, false]
}
```

After applying, the model is then fixed and ready for a final training. The model works as a single model, although it might contain more parameters than expected. This comes with pros and cons. The good side is, you can directly load the checkpoint dumped from supernet during search phase and start retrain from there. However, this is also a model with redundant parameters, which may cause problems when trying to count the number of parameters in model. For deeper reasons and possible workaround, see [Trainers](./NasReference.md#retrain).

Also refer to [DARTS](./DARTS.md) for example code of retraining.