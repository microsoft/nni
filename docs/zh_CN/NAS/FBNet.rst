FBNet
======

.. note:: 这个 One-Shot NAS 仍然在 NNI NAS 1.0 下实现，将在 v2.4 中迁移到 `Retiarii 框架 <https://github.com/microsoft/nni/issues/3814>`__。

对于 facial landmark 的移动应用，基于 PFLD 模型的基本架构，我们应用 FBNet（Block-wise DNAS）设计了一个在延迟和准确率之间权衡的简洁模型。 参考资料如下：


* `FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search <https://arxiv.org/abs/1812.03443>`__
* `PFLD: A Practical Facial Landmark Detector <https://arxiv.org/abs/1902.10859>`__

FBNet 是一种分块可微分 NAS 方法（Block-wise DNAS），通过使用 Gumbel Softmax 随机采样和可微分训练来选择最佳候选构建块。 在要搜索的每一层（或阶段），并排规划不同的候选块（就像结构重新参数化的有效性一样），从而对超网络进行充分的预训练。 对预训练的超网进一步采样来对子网进行微调，以实现更好的性能。

.. image:: ../../img/fbnet.png
   :target: ../../img/fbnet.png
   :alt:


PFLD 是一种用于实时应用的轻量级 facial landmark 模型。 通过使用 PeleeNet 的 stem 块、深度卷积的平均池化和 eSE 模块，PLFD 的结构首先被简化加速。

为了在延迟和准确性之间取得更好的平衡，FBNet 被进一步应用于简化的 PFLD，以便在每个特定层搜索最佳块。 搜索空间以 FBNet 空间为基础，并通过使用深度卷积的平均池化和 eSE 模块等对移动部署进行优化。


实验
------------

为了验证 FBNet 应用于 PFLD 的有效性，我们选择具有 106 个 landmark 的开源数据集作为基准：

* `106个点的 Facial Landmark Localization 大挑战 <https://arxiv.org/abs/1905.03469>`__

基线模型表示为 MobileNet-V3 PFLD（`参考基线 <https://github.com/Hsintao/pfld_106_face_landmarks>`__），搜索到的模型表示为 Subnet。 实验结果如下，其中延迟在高通625 CPU（ARMv8）上测试：


.. list-table::
   :header-rows: 1
   :widths: auto

   * - 模型
     - 大小
     - 延迟
     - 验证 NME
   * - MobileNet-V3 PFLD
     - 1.01MB
     - 10ms
     - 6.22%
   * - Subnet
     - 693KB
     - 1.60ms
     - 5.58%


示例
--------

`示例代码 <https://github.com/microsoft/nni/tree/master/examples/nas/oneshot/pfld>`__

请在示例目录下运行下面的脚本。

此处使用的 Python 依赖如下所示：

.. code-block:: bash

   numpy==1.18.5
   opencv-python==4.5.1.48
   torch==1.6.0
   torchvision==0.7.0
   onnx==1.8.1
   onnx-simplifier==0.3.5
   onnxruntime==1.7.0

数据准备
-----------------

首先，您应该将数据集 `106points dataset <https://drive.google.com/file/d/1I7QdnLxAlyG2Tq3L66QYzGhiBEoVfzKo/view?usp=sharing>`__ 下载到路径 ``./data/106points`` 。 数据集包括训练集和测试集：

.. code-block:: bash

   ./data/106points/train_data/imgs
   ./data/106points/train_data/list.txt
   ./data/106points/test_data/imgs
   ./data/106points/test_data/list.txt


快速入门
-----------

1. 搜索
^^^^^^^^^^

基于简化的 PFLD 架构，以构建超网为例，首先应配置多阶段搜索空间和搜索的超参数。

.. code-block:: bash

   from lib.builder import search_space
   from lib.ops import PRIMITIVES
   from lib.supernet import PFLDInference, AuxiliaryNet
   from nni.algorithms.nas.pytorch.fbnet import LookUpTable, NASConfig,

   # 超参数配置
   # search_space 定义多阶段搜索空间
   nas_config = NASConfig(
          model_dir="./ckpt_save",
          nas_lr=0.01,
          mode="mul",
          alpha=0.25,
          beta=0.6,
          search_space=search_space,
      )
   # 管理信息的查询表
   lookup_table = LookUpTable(config=nas_config, primitives=PRIMITIVES)
   # 创建超网
   pfld_backbone = PFLDInference(lookup_table)


在创建了搜索空间和超参数的超网后，我们可以运行以下命令开始搜索和训练超网。

.. code-block:: bash

   python train.py --dev_id "0,1" --snapshot "./ckpt_save" --data_root "./data/106points"

训练过程中会显示验证准确率，准确率最高的模型会被保存为 ``./ckpt_save/supernet/checkpoint_best.pth``。


2. 微调
^^^^^^^^^^^^

在对超网进行预训练后，我们可以运行以下命令对子网进行采样并进行微调：

.. code-block:: bash

   python retrain.py --dev_id "0,1" --snapshot "./ckpt_save" --data_root "./data/106points" \
                     --supernet "./ckpt_save/supernet/checkpoint_best.pth"

训练过程中会显示验证准确率，准确率最高的模型会被保存为 ``./ckpt_save/subnet/checkpoint_best.pth``。


3. 导出
^^^^^^^^^^

在对子网进行微调后，我们可以运行以下命令来导出 ONNX 模型。

.. code-block:: bash

   python export.py --supernet "./ckpt_save/supernet/checkpoint_best.pth" \
                    --resume "./ckpt_save/subnet/checkpoint_best.pth"

ONNX 模型被保存为 ``./output/subnet.onnx``，可以通过使用 `MNN <https://github.com/alibaba/MNN>`__ 进一步转换为移动推理引擎。

我们提供了预训练超网和子网的 checkpoint：

* `超网 <https://drive.google.com/file/d/1TCuWKq8u4_BQ84BWbHSCZ45N3JGB9kFJ/view?usp=sharing>`__
* `子网 <https://drive.google.com/file/d/160rkuwB7y7qlBZNM3W_T53cb6MQIYHIE/view?usp=sharing>`__
* `ONNX 模型 <https://drive.google.com/file/d/1s-v-aOiMv0cqBspPVF3vSGujTbn_T_Uo/view?usp=sharing>`__