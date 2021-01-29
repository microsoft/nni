NNI 上的 ProxylessNAS
======================================

介绍
------------

论文 `ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware <https://arxiv.org/pdf/1812.00332.pdf>`__ 删除了 proxy，它直接学习了用于大规模目标任务和目标硬件平台的体系结构。 它解决了可微分 NAS 大量内存消耗的问题，从而将计算成本较低到普通训练的水平，同时仍然能使用大规模的候选集。 参考论文了解详情。

用法
-----

要使用 ProxylessNAS 训练/搜索方法，用户需要使用 `NNI NAS 接口 <NasGuide.rst>`__ 在其模型中指定搜索空间，例如， ``LayerChoice``\ , ``InputChoice``。 定义并实例化模型，然后实例化 ProxylessNasTrainer，并将模型传入，剩下的工作由 Trainer 来完成。

.. code-block:: python

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

完整的示例代码参考 :githublink:`这里 <examples/nas/proxylessnas>`。

**ProxylessNasTrainer 的输入参数**


* **model** (*PyTorch model, 必填*\ ) - 用户要调整/搜索的模型。 它具有可变项以指定搜索空间。
* **model_optim** (*PyTorch optimizer, 必填*\ ) - optimizer 用户希望训练模型。
* **device** (*device, 必填*\ ) - 用户提供的用于进行训练/搜索的设备。 Trainer 会使用数据并行化。
* **train_loader** (*PyTorch data loader, 必填*\ ) - 用于训练集的数据加载器。
* **valid_loader** (*PyTorch data loader, 必填*\ ) - 用于验证集的数据加载器。
* **label_smoothing** (*float, 可选, 默认值为 0.1*\ ) - 标签平滑度。
* **n_epochs** (*int, 可选, 默认值为 120*\ ) - 要训练/搜索的 Epoch 数量。
* **init_lr** (*float, 可选, 默认值为 0.025*\ ) - 训练模型的初始学习率。
* **binary_mode** (*'two', 'full', 或 'full_v2', 可选, 默认为 'full_v2'*\ ) - Mutabor 中二进制权重的 forward, backword 模式。 'full' 表示前向传播所有候选操作，'two' 表示仅前向传播两个采样操作，'full_v2' 表示在反向传播时重新计算非激活的操作。
* **arch_init_type** (*'normal' 或 'uniform', 可选, 默认为 'normal'*\ ) - 初始化架构参数的方法。
* **arch_init_ratio** (*float, 可选, 默认值为 1e-3*\ ) - 初始架构参数的比率。
* **arch_optim_lr** (*float, 可选, 默认值为 1e-3*\ ) - 架构参数优化器的学习率。
* **arch_weight_decay** (*float, 可选, 默认值为 0*\ ) - 架构参数优化器的权重衰减。
* **grad_update_arch_param_every** (*int, 可选, 默认值为 5*\ ) - 多少个迷你批处理后更新权重。
* **grad_update_steps** (*int, 可选, 默认为 1*) - 在每次权重更新时，训练架构权重的次数。
* **warmup** (*bool, 可选, 默认值是 True*\ ) - 是否做热身。
* **warmup_epochs** (*int, 可选, 默认值是 25*\ ) - 预热期间要进行的时期数。
* **arch_valid_frequency** (*int, 可选, 默认值为 1*\ ) - 打印确认结果的频率。
* **load_ckpt** (*bool, 可选, 默认值为 False*\ ) - 是否加载 checkpoint。
* **ckpt_path** (*str, 可选, 默认为 None*\ ) - 检查点路径。如果 load_ckpt 为 True，ckpt_path 不能为 None。
* **arch_path** (*str, 可选, 默认值为 None*\ ) - 存储所选架构的路径。

实现
--------------

在NNI上的实现基于 `官方实现 <https://github.com/mit-han-lab/ProxylessNAS>`__ 。 官方实现支持两种搜索方法：梯度下降和强化学习，还支持不同的硬件，包括 'mobile', 'cpu', 'gpu8', 'flops'。 在当前的 NNI 实现中，支持梯度下降训练方法，不支持不同的硬件。 完整支持正在进行中。

下面将介绍实现的细节。 像 NNI 上其它 one-shot NAS 算法一样，ProxylessNAS 由两部分组成：*搜索空间* 和 *训练方法*。 为了让用户灵活地定义自己的搜索空间并使用内置的 ProxylessNAS 训练方法，我们将指定的搜索空间放在  :githublink:`示例代码 <examples/nas/proxylessnas>` 使用 :githublink:`NNI NAS 接口<src/sdk/pynni/nni/nas/pytorch/proxylessnas>`。


.. image:: ../../img/proxylessnas.png
   :target: ../../img/proxylessnas.png
   :alt: 


ProxylessNAS 搜索方法由 ProxylessNasMutator 和 ProxylessNasTrainer 组成。 ProxylessNasMutator 为每个可变量初始化了 MixedOp (即, LayerChoice)，并会在 MixedOp 中管理架构权重。 **对于数据并行化**，架构权重会在用户模型中。 具体地说，在 ProxylessNAS 视线中，为可变变量 (即, LayerChoice) 添加了 MixedOp 作为成员变量。 Mutator 也公开了两个成员函数：``arch_requires_grad`` 和 ``arch_disable_grad``，用于 Trainer 来控制架构权重的训练。

ProxylessNasMutator 还实现了可变量的前向逻辑 (即, LayerChoice)。

重现结果
-----------------

为了重现结果，首先运行了搜索过程。我们发现虽然需要跑许多 Epoch，但选择的架构会在头几个 Epoch 就收敛了。 这可能是由超参或实现造成的，正在分析中。 找到架构的测试精度为 top1: 72.31, top5: 90.26。
