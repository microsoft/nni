# One-Shot NAS algorithms

除了 [经典 NAS 算法](./ClassicNas.md)，还可以使用更先进的 One-Shot NAS 算法来从搜索空间中找到更好的模型。 One-Shot NAS 算法已有了大量的相关工作，如 [SMASH](https://arxiv.org/abs/1708.05344), [ENAS](https://arxiv.org/abs/1802.03268), [DARTS](https://arxiv.org/abs/1808.05377), [FBNet](https://arxiv.org/abs/1812.03443), [ProxylessNAS](https://arxiv.org/abs/1812.00332), [SPOS](https://arxiv.org/abs/1904.00420), [Single-Path NAS](https://arxiv.org/abs/1904.02877),  [Understanding One-shot](http://proceedings.mlr.press/v80/bender18a) 以及 [GDAS](https://arxiv.org/abs/1910.04465)。 One-shot NAS algorithms usually build a supernet containing every candidate in the search space as its subnetwork, and in each step, a subnetwork or combination of several subnetworks is trained.

Currently, several one-shot NAS methods are supported on NNI. For example, `DartsTrainer`, which uses SGD to train architecture weights and model weights iteratively, and `ENASTrainer`, which [uses a controller to train the model](https://arxiv.org/abs/1802.03268). New and more efficient NAS trainers keep emerging in research community and some will be implemented in future releases of NNI.

## Search with One-shot NAS Algorithms

Each one-shot NAS algorithm implements a trainer, for which users can find usage details in the description of each algorithm. Here is a simple example, demonstrating how users can use `EnasTrainer`.

```python
# this is exactly same as traditional model training
model = Net()
dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)

# use NAS here
def top1_accuracy(output, target):
    # this is the function that computes the reward, as required by ENAS algorithm
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size

def metrics_fn(output, target):
    # metrics function receives output and target and computes a dict of metrics
    return {"acc1": top1_accuracy(output, target)}

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
                           log_frequency=10)  # print log every 10 steps
trainer.train()  # training
trainer.export(file="model_dir/final_architecture.json")  # export the final architecture to file
```

`model` is the one with [user defined search space](./WriteSearchSpace.md). Then users should prepare training data and model evaluation metrics. To search from the defined search space, a one-shot algorithm is instantiated, called trainer (e.g., EnasTrainer). The trainer exposes a few arguments that you can customize. For example, the loss function, the metrics function, the optimizer, and the datasets. These should satisfy most usage requirements and we do our best to make sure our built-in trainers work on as many models, tasks, and datasets as possible.

**Note that** when using one-shot NAS algorithms, there is no need to start an NNI experiment. Users can directly run this Python script (i.e., `train.py`) through `python3 train.py` without `nnictl`. After training, users can export the best one of the found models through `trainer.export()`.

Each trainer in NNI has its targeted scenario and usage. Some trainers have the assumption that the task is a classification task; some trainers might have a different definition of "epoch" (e.g., an ENAS epoch = some child steps + some controller steps). Most trainers do not have support for distributed training: they won't wrap your model with `DataParallel` or `DistributedDataParallel` to do that. So after a few tryouts, if you want to actually use the trainers on your very customized applications, you might need to [customize your trainer](./Advanced.md#extend-the-ability-of-one-shot-trainers).

Furthermore, one-shot NAS can be visualized with our NAS UI. [See more details.](./Visualization.md)

### Retrain with Exported Architecture

After the search phase, it's time to train the found architecture. Unlike many open-source NAS algorithms who write a whole new model specifically for retraining. We found that the search model and retraining model are usually very similar, and therefore you can construct your final model with the exact same model code. For example

```python
model = Net()
apply_fixed_architecture(model, "model_dir/final_architecture.json")
```

The JSON is simply a mapping from mutable keys to choices. Choices can be expressed in:

* string: 根据名称来指定候选项。
* number: 根据索引来指定候选项。
* string 数组: 根据名称来指定候选项。
* number 数组: 根据索引来指定候选项。
* boolean 数组: 可直接选定多项的数组。

For example,

```json
{
    "LayerChoice1": "conv5x5",
    "LayerChoice2": 6,
    "InputChoice3": ["layer1", "layer3"],
    "InputChoice4": [1, 2],
    "InputChoice5": [false, true, false, false, true]
}
```

After applying, the model is then fixed and ready for final training. The model works as a single model, and unused parameters and modules are pruned.

Also, refer to [DARTS](./DARTS.md) for code exemplifying retraining.
