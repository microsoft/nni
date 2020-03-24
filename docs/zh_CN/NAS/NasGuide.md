# 指南：在 NNI 上使用 NAS

```eval_rst
.. contents::

.. Note:: 此 API 初始试验阶段。 当前接口可能会更改。
```

![](../../img/nas_abstract_illustration.png)

现代神经架构搜索（NAS）方法通常包含 [三个维度](https://arxiv.org/abs/1808.05377)：搜索空间、搜索策略和性能估计策略。 Search space often contains a limited number of neural network architectures to explore, while the search strategy samples architectures from search space, gets estimations of their performance, and evolves itself. Ideally, the search strategy should find the best architecture in the search space and report it to users. After users obtain the "best architecture", many methods use a "retrain step", which trains the network with the same pipeline as any traditional model.

## 实现搜索空间

Assuming we've got a baseline model, what should we do to be empowered with NAS? 以 [PyTorch 上的 MNIST](https://github.com/pytorch/examples/blob/master/mnist/main.py) 为例，代码如下：

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

以上示例在 conv1 上添加了 conv5x5 的选项。 The modification is as simple as declaring a `LayerChoice` with the original conv3x3 and a new conv5x5 as its parameter. 就这么简单！ You don't have to modify the forward function in any way. You can imagine conv1 as any other module without NAS.

如何表示可能的连接？ This can be done using `InputChoice`. To allow for a skip connection on the MNIST example, we add another layer called conv3. 下面的示例中，从 conv2 的可能连接加入到了 conv3 的输出中。

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

Input choice can be thought of as a callable module that receives a list of tensors and outputs the concatenation/sum/mean of some of them (sum by default), or `None` if none is selected. 与 Layer Choice 一样，Input Choice 要**在 `__init__` 中初始化，并在 `forward` 中调用。 We will see later that this is to allow search algorithms to identify these choices and do necessary preparations.</p>

`LayerChoice` 和 `InputChoice` 都是 **Mutable**。 Mutable 表示 "可变化的"。 As opposed to traditional deep learning layers/modules which have fixed operation types once defined, models with mutable are essentially a series of possible models.

用户可为每个 Mutable 指定 **key**。 By default, NNI will assign one for you that is globally unique, but in case users want to share choices (for example, there are two `LayerChoice`s with the same candidate operations and you want them to have the same choice, i.e., if first one chooses the i-th op, the second one also chooses the i-th op), they can give them the same key. The key marks the identity for this choice and will be used in the dumped checkpoint. 如果要增加导出架构的可读性，可为每个 Mutable 的 key 指派名称。 高级用法参考 [Mutable](./NasReference.md)。

## 使用搜索算法

Aside from using a search space, there are at least two other ways users can do search. One runs NAS distributedly, which can be as naive as enumerating all the architectures and training each one from scratch, or can involve leveraging more advanced technique, such as [SMASH](https://arxiv.org/abs/1708.05344), [ENAS](https://arxiv.org/abs/1802.03268), [DARTS](https://arxiv.org/abs/1808.05377), [FBNet](https://arxiv.org/abs/1812.03443), [ProxylessNAS](https://arxiv.org/abs/1812.00332), [SPOS](https://arxiv.org/abs/1904.00420), [Single-Path NAS](https://arxiv.org/abs/1904.02877),  [Understanding One-shot](http://proceedings.mlr.press/v80/bender18a) and [GDAS](https://arxiv.org/abs/1910.04465). Since training many different architectures is known to be expensive, another family of methods, called one-shot NAS, builds a supernet containing every candidate in the search space as its subnetwork, and in each step, a subnetwork or combination of several subnetworks is trained.

Currently, several one-shot NAS methods are supported on NNI. For example, `DartsTrainer`, which uses SGD to train architecture weights and model weights iteratively, and `ENASTrainer`, which [uses a controller to train the model](https://arxiv.org/abs/1802.03268). New and more efficient NAS trainers keep emerging in research community and some will be implemented in future releases of NNI.

### One-Shot NAS

Each one-shot NAS algorithm implements a trainer, for which users can find usage details in the description of each algorithm. 这是如何使用 `EnasTrainer` 的简单示例。

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

Users can directly run their training file through `python3 train.py` without `nnictl`. After training, users can export the best one of the found models through `trainer.export()`.

Normally, the trainer exposes a few arguments that you can customize. For example, the loss function, the metrics function, the optimizer, and the datasets. These should satisfy most usages needs and we do our best to make sure our built-in trainers work on as many models, tasks, and datasets as possible. But there is no guarantee. For example, some trainers have the assumption that the task is a classification task; some trainers might have a different definition of "epoch" (e.g., an ENAS epoch = some child steps + some controller steps); most trainers do not have support for distributed training: they won't wrap your model with `DataParallel` or `DistributedDataParallel` to do that. So after a few tryouts, if you want to actually use the trainers on your very customized applications, you might need to [customize your trainer](#extend-the-ability-of-one-shot-trainers).

### 分布式 NAS

Neural architecture search was originally executed by running each child model independently as a trial job. We also support this searching approach, and it naturally fits within the NNI hyper-parameter tuning framework, where Tuner generates child models for the next trial and trials run in the training service.

To use this mode, there is no need to change the search space expressed with the NNI NAS API (i.e., `LayerChoice`, `InputChoice`, `MutableScope`). 模型初始化后，在模型上调用 `get_and_apply_next_architecture`。 One-shot NAS Trainer 不能在此模式中使用。 简单示例：

```python
model = Net()

# 从 Tuner 中获得选择的架构，并应用到模型上
get_and_apply_next_architecture(model)
train(model)  # 训练模型的代码
acc = test(model)  # 测试训练好的模型
nni.report_final_result(acc)  # 报告所选架构的性能
```

The search space should be generated and sent to Tuner. As with the NNI NAS API, the search space is embedded in the user code. Users can use "[nnictl ss_gen](../Tutorial/Nnictl.md)" to generate the search space file. Then put the path of the generated search space in the field `searchSpacePath` of `config.yml`. The other fields in `config.yml` can be filled by referring [this tutorial](../Tutorial/QuickStart.md).

You can use the [NNI tuners](../Tuner/BuiltinTuner.md) to do the search. Currently, only PPO Tuner supports NAS search spaces.

We support a standalone mode for easy debugging, where you can directly run the trial command without launching an NNI experiment. 可以通过此方法来检查 Trial 代码是否可正常运行。 在独立模式下，`LayerChoice` 和 `InputChoice` 会选择最开始的候选项。

[此处](https://github.com/microsoft/nni/tree/master/examples/nas/classic_nas/config_nas.yml)是完整示例。

### 使用导出的架构重新训练

After the search phase, it's time to train the found architecture. 与很多开源 NAS 算法不同，它们为重新训练专门写了新的模型。 We found that the search model and retraining model are usually very similar, and therefore you can construct your final model with the exact same model code. 例如：

```python
model = Net()
apply_fixed_architecture(model, "model_dir/final_architecture.json")
```

JSON 文件是从 Mutable key 到 Choice 的表示。 例如：

```json
{
    "LayerChoice1": [false, true, false, false],
    "InputChoice2": [true, true, false]
}
```

After applying, the model is then fixed and ready for final training. 虽然它可能包含了更多的参数，但可作为单个模型来使用。 这各有利弊。 The good side is, you can directly load the checkpoint dumped from supernet during the search phase and start retraining from there. However, this is also a model with redundant parameters and this may cause problems when trying to count the number of parameters in the model. For deeper reasons and possible workarounds, see [Trainers](./NasReference.md).

Also, refer to [DARTS](./DARTS.md) for code exemplifying retraining.
