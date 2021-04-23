FBNet：基于可微分网络结构搜索的硬件感知型高效率卷积网络设计
=======================================================================================

针对人脸关键点的移动端应用，我们基于PFLD模型的基础架构，通过FBNet的模块级网络结构搜索，设计了精简高效、性能良好的模型，参考论文包括：

* `FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search <https://arxiv.org/abs/1812.03443>`__
* `PFLD: A Practical Facial Landmark Detector <https://arxiv.org/abs/1902.10859>`__

FBNet是一种可微分的、模块级的网络结构搜索方法，通过Gumbel Softmax实现随机采样与可微分训练，为每个待搜索的网络层确定最佳构建模块。在每个待搜索的网络层，由于不同候选模块的并排训练（类似于结构重参数化），可实现超网络的充分训练，从而获得优越的精度性能。

.. image:: ../../img/fbnet.png
   :target: ../../img/fbnet.png
   :alt:


PFLD是一种面向实时应用的、轻量型人脸关键点模型，其网络主干基于MobileNet-V2设计。我们首先简化了PFLD的基本结构，其中Stem-block选自PeleeNet，Average pooling通过Depthwise Convolution实现，并且引入了eSE module以提升计算效率。

为了进一步实现精度与速度的有效折中，我们将FBNet的搜索策略，应用于PFLD关键网络层的模块级搜索。并且，我们基于FBNet的搜索空间（FBNet space），优化了一些基本构建模块，可以确保移动端部署时的推理速度，基本优化点包括Average pooling通过Depthwise Convolution实现、eSE module的使用等。

实验结果
------------------

为了检验PFLD的精度稳健性、以及FBNet的搜索效果，我们选择包含106关键点的公开数据集作为验证场景，数据集介绍如下：

* `Grand Challenge of 106-Point Facial Landmark Localization <https://arxiv.org/abs/1905.03469>`__

选择的基线模型以MobileNet-V3为主干，表示为MobileNet-V3 PFLD（`参考开源实现 <https://github.com/Hsintao/pfld_106_face_landmarks>`__）。搜索模型通过超网络预训练、子网络采样与微调获得，表示为Subnet。搜索结果如下，其中推理延迟的测试平台为高通625处理器 (ARMv8)：

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Model
     - Size
     - Latency
     - Validation NME
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

`示例代码 <https://github.com/microsoft/nni/tree/master/examples/nas/oneshot/fbnet>`__

请在示例目录下运行下面的脚本。

准备数据
----------------

首先你需要下载 `106points dataset <https://drive.google.com/file/d/1I7QdnLxAlyG2Tq3L66QYzGhiBEoVfzKo/view?usp=sharing>`__ 到目录 ``./data/106points`` 里，包括训练集与验证集，目录结构如下：

.. code-block:: bash

   ./data/106points/train_data/imgs
   ./data/106points/train_data/list.txt
   ./data/106points/test_data/imgs
   ./data/106points/test_data/list.txt

快速入门
-----------

1. 搜索预训练
^^^^^^^^^

首先构建搜索训练环境，基本的环境依赖如下：

.. code-block:: bash

   numpy==1.18.5
   opencv-python==4.5.1.48
   torch==1.6.0
   torchvision==0.7.0
   onnx==1.8.1
   onnx-simplifier==0.3.5
   onnxruntime==1.7.0

基于PFLD的基础模型结构，不同阶段的搜索空间设置、以及搜索训练相关的参数配置，参考 `示例 <https://github.com/microsoft/nni/tree/master/examples/nas/oneshot/fbnet/lib/config.py>`__

在指定搜索空间与参数配置之后，可以通过运行以下命令，执行网络结构搜索、与超网络预训练：

.. code-block:: bash

   python train.py --net "supernet" -as --dev_id "0,1" --snapshot "./ckpt_save" --data_root "./data/106points"

搜索训练过程中会显示验证精度，训练的最佳超网络保存为 ``./ckpt_save/supernet/checkpoint_min_nme.pth``。


2. 子网络微调
^^^^^^^^^^^

完成超网络搜索与预训练之后，可以通过运行以下命令，执行子网络结构采样、与微调训练：

.. code-block:: bash

   python train.py --net "subnet" --dev_id "0,1" --snapshot "./ckpt_save" --data_root "./data/106points" \
                   --supernet "./ckpt_save/supernet/checkpoint_min_nme.pth"

微调训练过程中会显示验证精度，训练的最佳子网络保存为 ``./ckpt_save/subnet/checkpoint_min_nme.pth``。


3. 导出ONNX模型
^^^^^^^^^

完成子网络微调之后，可以通过运行以下命令，导出ONNX模型：

.. code-block:: bash

   python export.py --supernet "./ckpt_save/supernet/checkpoint_min_nme.pth" \
                    --resume "./ckpt_save/subnet/checkpoint_min_nme.pth"

ONNX模型保存为 ``./output/subnet.onnx``，可进一步通过 `MNN <https://github.com/alibaba/MNN>`__ 转换为移动端推理引擎。

我们提供了超网络预训练模型、与子网络微调模型：

* `超网络 <https://drive.google.com/file/d/1TCuWKq8u4_BQ84BWbHSCZ45N3JGB9kFJ/view?usp=sharing>`__
* `子网络 <https://drive.google.com/file/d/160rkuwB7y7qlBZNM3W_T53cb6MQIYHIE/view?usp=sharing>`__
* `子网络ONNX <https://drive.google.com/file/d/1s-v-aOiMv0cqBspPVF3vSGujTbn_T_Uo/view?usp=sharing>`__
