单路径 One-Shot (SPOS)
===========================

介绍
------------

在 `Single Path One-Shot Neural Architecture Search with Uniform Sampling <https://arxiv.org/abs/1904.00420>`__  中提出的 one-shot NAS 方法，通过构造简化的通过统一路径采样方法训练的超网络来解决 One-Shot 模型训练的问题。这样所有架构（及其权重）都得到了完全且平等的训练。 然后，采用进化算法无需任何微调即可有效的搜索出性能最佳的体系结构。

在 NNI 上的实现基于  `官方 Repo <https://github.com/megvii-model/SinglePathOneShot>`__。 实现了一个训练超级网络的 Trainer，以及一个利用 NNI 框架能力来加速进化搜索阶段的进化 Tuner。 还展示了 

示例
--------

此示例是论文中的搜索空间，使用 flops 限制来执行统一的采样方法。

:githublink:`示例代码 <examples/nas/spos>`

必需组件
^^^^^^^^^^^^

由于使用了 DALI 来加速 ImageNet 的数据读取，需要 NVIDIA DALI >= 0.16。 `安装指导 <https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/installation.html>`__。

从 `这里 <https://1drv.ms/u/s!Am_mmG2-KsrnajesvSdfsq_cN48?e=aHVppN>`__  (由 `Megvii <https://github.com/megvii-model>`__\  维护) 下载 flops 查找表。
将 ``op_flops_dict.pkl`` 和 ``checkpoint-150000.pth.tar`` (如果不需要重新训练超网络) 放到 ``data`` 目录中。

准备标准格式的 ImageNet (参考 `这里的脚本
<https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4>`__\)。 将其链接到 ``data/imagenet`` 会更方便。

准备好后，应具有以下代码结构：

.. code-block:: bash

   spos
   ├── architecture_final.json
   ├── blocks.py
   ├── config_search.yml
   ├── data
   │   ├── imagenet
   │   │   ├── train
   │   │   └── val
   │   └── op_flops_dict.pkl
   ├── dataloader.py
   ├── network.py
   ├── readme.md
   ├── scratch.py
   ├── supernet.py
   ├── tester.py
   ├── tuner.py
   └── utils.py

步骤 1. 训练超网络
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python supernet.py

会将检查点导出到 ``checkpoints`` 目录中，为下一步做准备。

注意：数据加载的官方 Repo `与通常的方法有所不同 <https://github.com/megvii-model/SinglePathOneShot/issues/5>`__\ ，使用了 BGR 张量，以及 0 到 255 之间的值来与自己的深度学习框架对齐。 选项 ``--spos-preprocessing`` 会模拟原始的使用行为，并能使用预训练的检查点。

步骤 2. 进化搜索
^^^^^^^^^^^^^^^^^^^^^^^^

单路径 One-Shot 利用进化算法来搜索最佳架构。 tester 负责通过训练图像的子集来测试采样的体系结构，重新计算所有批处理规范，并在完整的验证集上评估架构。

为了使 Tuner 识别 flops 限制并能计算 flops，在 ``tuner.py`` 中创建了新的 ``EvolutionWithFlops`` Tuner，其继承于 SDK 中的 tuner。

要为 NNI 框架准备好搜索空间，首先运行

.. code-block:: bash

   nnictl ss_gen -t "python tester.py"

将生成 ``nni_auto_gen_search_space.json`` 文件，这是搜索空间的序列化形式。

默认情况下，它将使用前面下载的 ``checkpoint-150000.pth.tar``。 如果要使用从自行训练的检查点，在 ``config_search.yml`` 中的命令上指定 ``---checkpoint``。

然后使用进化 Tuner 搜索。

.. code-block:: bash

   nnictl create --config config_search.yml

从每个 Epoch 导出的最终架构可在 Tuner 工作目录下的 ``checkpoints`` 中找到，默认值为 ``$HOME/nni-experiments/your_experiment_id/log``。

步骤 3. 从头开始训练
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python scratch.py

默认情况下，它将使用 ``architecture_final.json``. 该体系结构由官方仓库提供（转换成了 NNI 格式）。 通过 ``--fixed-arc`` 选项，可使用任何结构（例如，步骤 2 中找到的结构）。

参考
---------

PyTorch
^^^^^^^

..  autoclass:: nni.algorithms.nas.pytorch.spos.SPOSEvolution
    :members:

..  autoclass:: nni.algorithms.nas.pytorch.spos.SPOSSupernetTrainer
    :members:

..  autoclass:: nni.algorithms.nas.pytorch.spos.SPOSSupernetTrainingMutator
    :members:

已知的局限
-----------------


* 仅支持 Block 搜索。 尚不支持通道搜索。
* 仅提供 GPU 版本。

当前重现结果
----------------------------

重现中。 由于官方版本和原始论文之间的不同，我们将当前结果与官方 Repo（我们运行的结果）和论文进行了比较。


* 进化阶段几乎与官方 Repo 一致。 进化算法显示出了收敛趋势，在搜索结束时达到约 65% 的精度。 但此结果与论文不一致。 详情参考 `此 issue <https://github.com/megvii-model/SinglePathOneShot/issues/6>`__。
* 重新训练阶段未匹配。 我们的重新训练代码，使用了作者发布的架构，获得了 72.14% 的准确率，与官方发布的 73.61%，和原始论文中的 74.3% 有一定差距。
