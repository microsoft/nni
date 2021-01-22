ENAS
====

介绍
------------

`Efficient Neural Architecture Search via Parameter Sharing <https://arxiv.org/abs/1802.03268>`__ 这篇论文使用子模型之间的参数共享来加速NAS进程。 在 ENAS 中，Contoller 学习在大的计算图中搜索最有子图的方式来发现神经网络。 Controller 通过梯度策略训练，从而选择出能在验证集上有最大期望奖励的子图。 同时对与所选子图对应的模型进行训练，以最小化规范交叉熵损失。

NNI 基于官方的 `Tensorflow <https://github.com/melodyguan/enas>`_ 实现，包括通用的强化学习的 Controller，以及能交替训练目标网络和 Controller 的 Trainer。 根据论文，也对 CIFAR10 实现了 Macro 和 Micro 搜索空间来展示如何使用 Trainer。 NNI 中从头训练的代码还未完成，当前还没有重现结果。

示例
--------

CIFAR10 Macro/Micro 搜索空间
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:githublink:`示例代码 <examples/nas/enas>`

.. code-block:: bash

   ＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
   git clone https://github.com/Microsoft/nni.git

   # 搜索最优结构
   cd examples/nas/enas

   # 在 macro 搜索空间中搜索
   python3 search.py --search-for macro

   # 在 micro 搜索空间中搜索
   python3 search.py --search-for micro

   # 查看更多的搜索选择
   python3 search.py -h

参考
---------

PyTorch
^^^^^^^

..  autoclass:: nni.algorithms.nas.pytorch.enas.EnasTrainer
    :members:

..  autoclass:: nni.algorithms.nas.pytorch.enas.EnasMutator
    :members:
