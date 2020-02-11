# NNI 上的 ProxylessNAS

## 介绍

论文 [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/pdf/1812.00332.pdf) 去掉了代理，直接从大规模目标任务和目标硬件平台上学习架构。 它解决了可微分 NAS 大量内存消耗的问题，从而将计算成本较低到普通训练的水平，同时仍然能使用大规模的候选集。 参考论文了解详情。

## 用法

要使用 ProxylessNAS 训练、搜索方法，用户要在模型中使用 [NNI NAS interface](NasGuide.md) 来指定搜索空间，例如，`LayerChoice`，`InputChoice`。 定义并实例化模型，然后实例化 ProxylessNasTrainer，并将模型传入，剩下的工作由 Trainer 来完成。
```python
trainer = ProxylessNasTrainer(model,
                              model_optim=optimizer,
                              train_loader=data_provider.train,
                              valid_loader=data_provider.valid,
                              device=device,
                              warmup=True,
                              ckpt_path=args.checkpoint_path,
                              arch_path=args.arch_path)
trainer.train()
trainer.export(args.arch_path)
```
[此处](https://github.com/microsoft/nni/tree/master/examples/nas/proxylessnas)是完整示例。

**ProxylessNasTrainer 的输入参数**

* **model** (*PyTorch 模型, 必需*) - 需要调优、搜索的模型。 它具有可变项以指定搜索空间。
* **model_optim** (*PyTorch 优化器, 必需*) - 训练模型所需要的优化器。
* **device** (*device, 必需*) - 用于训练、搜索的 device。 Trainer 会使用数据并行化。
* **train_loader** (*PyTorch DataLoader, 必需*) - 训练数据集的 DataLoader。
* **valid_loader** (*PyTorch DataLoader, 必需*) - 验证数据集的 DataLoader。
* **label_smoothing** (*float, 可选, 默认为 0.1*) - 标签平滑度。
* **n_epochs** (*int, 可选, 默认为 120*) - 训练、搜索的 Epoch 数量。
* **init_lr** (*float, 可选, 默认为 0.025*) - 训练的初始学习率。
* **binary_mode** (*'two', 'full', 或 'full_v2', 可选, 默认为 'full_v2'*) - Mutabor 中二进制权重的 forward, backword 模式。 'full' 表示前向传播所有候选操作，'two' 表示仅前向传播两个采样操作，'full_v2' 表示在反向传播时重新计算非激活的操作。
* **arch_init_type** (*'normal' 或 'uniform', 可选, 默认为 'normal'*) - 初始化架构参数的方法。
* **arch_init_ratio** (*float, 可选, 默认为 1e-3*) - 初始化架构参数的比例。
* **arch_optim_lr** (*float, 可选, 默认为 1e-3*) - 架构参数优化器的学习率。
* **arch_weight_decay** (*float, 可选, 默认为 0*) - 架构参数优化器的权重衰减。
* **grad_update_arch_param_every** (*int, 可选, 默认为 5*) - 多少个迷你批处理后更新权重。
* **grad_update_steps** (*int, 可选, 默认为 1*) - 在每次权重更新时，训练架构权重的次数。
* **warmup** (*bool, 可选, 默认为 True*) - 是否需要热身。
* **warmup_epochs** (*int, 可选, 默认为 25*) - 热身的 Epoch 数量。
* **arch_valid_frequency** (*int, 可选, 默认为 = 1*) - 输出验证集结果的频率。
* **load_ckpt** (*bool, 可选, 默认为 False*) - 是否加载检查点。
* **ckpt_path** (*str, 可选, 默认为 None*) - 检查点路径。如果 load_ckpt 为 True，ckpt_path 不能为 None。
* **arch_path** (*str, 可选, 默认为 None*) - 选择架构的路径。


## 实现

NNI 上的实现基于[官方实现](https://github.com/mit-han-lab/ProxylessNAS)。 官方实现支持两种搜索方法：梯度下降和强化学习，还支持不同的硬件，包括 'mobile', 'cpu', 'gpu8', 'flops'。 在当前的 NNI 实现中，支持梯度下降训练方法，不支持不同的硬件。 完整支持正在进行中。

下面将介绍实现的细节。 像 NNI 上其它 one-shot NAS 算法一样，ProxylessNAS 由两部分组成：*搜索空间* 和 *训练方法*。 For users to flexibly define their own search space and use built-in ProxylessNAS training approach, we put the specified search space in [example code](https://github.com/microsoft/nni/tree/master/examples/nas/proxylessnas) using [NNI NAS interface](NasGuide.md), and put the training approach in [SDK](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch/proxylessnas).

![](../../img/proxylessnas.png)

ProxylessNAS training approach is composed of ProxylessNasMutator and ProxylessNasTrainer. ProxylessNasMutator instantiates MixedOp for each mutable (i.e., LayerChoice), and manage architecture weights in MixedOp. **For DataParallel**, architecture weights should be included in user model. Specifically, in ProxylessNAS implementation, we add MixedOp to the corresponding mutable (i.e., LayerChoice) as a member variable. The mutator also exposes two member functions, i.e., `arch_requires_grad`, `arch_disable_grad`, for the trainer to control the training of architecture weights.

ProxylessNasMutator also implements the forward logic of the mutables (i.e., LayerChoice).

## 重现结果

进行中...
