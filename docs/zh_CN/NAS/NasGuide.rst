One-Shot NAS 算法
=======================

除了 `经典 NAS 算法 <./ClassicNas.rst>`_，还可以使用更先进的 One-Shot NAS 算法来从搜索空间中找到更好的模型。 One-Shot NAS 算法已有了大量的相关工作，如  `SMASH <https://arxiv.org/abs/1708.05344>`__\ , `ENAS <https://arxiv.org/abs/1802.03268>`__\ , `DARTS <https://arxiv.org/abs/1808.05377>`__\ , `FBNet <https://arxiv.org/abs/1812.03443>`__\ , `ProxylessNAS <https://arxiv.org/abs/1812.00332>`__\ , `SPOS <https://arxiv.org/abs/1904.00420>`__\ , `Single-Path NAS <https://arxiv.org/abs/1904.02877>`__\ ,  `Understanding One-shot <http://proceedings.mlr.press/v80/bender18a>`__ and `GDAS <https://arxiv.org/abs/1910.04465>`__。 One-Shot NAS 算法通常会构建一个超网络，其中包含的子网作为此搜索空间的候选项。每一步，会训练一个或多个子网的组合。

当前，NNI 支持数种 One-Shot 方法。 例如，``DartsTrainer`` 使用 SGD 迭代训练体系结构权重和模型权重，``ENASTrainer`` `使用控制器训练模型 <https://arxiv.org/abs/1802.03268>`__。 新的、更高效的 NAS Trainer 在研究界不断的涌现出来，NNI 会在将来的版本中实现其中的一部分。

使用 One-Shot NAS 算法进行搜索
-----------------------------------

每个 One-Shot NAS 算法都实现了 Trainer，可在每种算法说明中找到详细信息。 这是如何使用 ``EnasTrainer`` 的简单示例。

.. code-block:: python

   # 与传统模型训练完全相同
   model = Net()
   dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
   dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)

   # 这里使用 NAS
   def top1_accuracy(output, target):
       # 这是ENAS算法要求的计算奖励的函数
       batch_size = target.size(0)
       _, predicted = torch.max(output.data, 1)
       return (predicted == target).sum().item() / batch_size

   def metrics_fn(output, target):
       # metrics 函数接收 output 和 target 并计算出 dict metrics
       return {"acc1": top1_accuracy(output, target)}

   from nni.algorithms.nas.pytorch import enas
   trainer = enas.EnasTrainer(model,
                              loss=criterion,
                              metrics=metrics_fn,
                              reward_function=top1_accuracy,
                              optimizer=optimizer,
                              batch_size=128
                              num_epochs=10,  # 10 epochs
                              dataset_train=dataset_train,
                              dataset_valid=dataset_valid,
                              log_frequency=10)  # 每 10s 打印一次 log
   trainer.train()  # training
   trainer.export(file="model_dir/final_architecture.json")  # 把最终的架构导出到文件

``model`` 是 `具有用户定义搜索空间的模型 <./WriteSearchSpace.rst>`__。 然后需要准备搜索数据和模型评估指标。 要从定义的搜索空间中进行搜索，需要实例化 One-Shot 算法，即 Trainer（如，EnasTrainer）。 Trainer 会提供一些可以自定义的参数。 如，损失函数，指标函数，优化器以及数据集。 这些功能可满足大部分需求，NNI 会尽力让内置 Trainer 能够处理更多的模型、任务和数据集。

**注意** ，在使用 One-Shot NAS 算法时，不需要启动 NNI Experiment。 不需要 ``nnictl`` ，可直接运行 Python 脚本（即：``train.py`` )，如：``python3 train.py``。 训练完成后，可通过 ``trainer.export()`` 导出找到的最好的模型。

NNI 中每个 Trainer 都用其对应的场景和用法。 一些 Trainer 假定任务是分类任务；一些 Trainer 对 "epoch" 有不同的定义（如：ENAS 的每个 Epoch 是 一些子步骤加上 Controller 的步骤）。 大部分 Trainer 不支持分布式训练：没有使用 ``DataParallel`` 或 ``DistributedDataParallel`` 来包装模型。 如果通过试用，想要在定制的应用中使用 Trainer，可能需要 `自定义 Trainer <./Advanced.rst>`__。

此外，可以使用 NAS 可视化来显示 One-Shot NAS。 `请看细节。 <./Visualization.rst>`__

使用导出的架构重新训练
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

搜索阶段后，就该训练找到的架构了。 与很多开源 NAS 算法不同，它们为重新训练专门写了新的模型。 我们发现搜索模型和重新训练模型的过程非常相似，因而可直接将一样的模型代码用到最终模型上。 例如：

.. code-block:: python

   model = Net()
   apply_fixed_architecture(model, "model_dir/final_architecture.json")

此 JSON 是从 Mutable 键值到 Choice 的映射。 Choice 可为：


* string: 根据名称来指定候选项。
* number: 根据索引来指定候选项。
* string 数组: 根据名称来指定候选项。
* number 数组: 根据索引来指定候选项。
* boolean 数组: 可直接选定多项的数组。

例如：

.. code-block:: json

   {
       "LayerChoice1": "conv5x5",
       "LayerChoice2": 6,
       "InputChoice3": ["layer1", "layer3"],
       "InputChoice4": [1, 2],
       "InputChoice5": [false, true, false, false, true]
   }

应用后，模型会被固定，并准备好进行最终训练。 该模型作为单独的模型来工作，未使用的参数和模块已被剪除。

也可参考 `DARTS <./DARTS.rst>`__ 的重新训练代码。
