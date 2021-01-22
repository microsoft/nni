神经网络结构搜索的对比
=====================================

*匿名作者*

训练和比较 NAS（神经网络架构搜索）的模型，包括 Autokeras，DARTS，ENAS 和 NAO。

源码链接如下：


* 
  Autokeras: `https://github.com/jhfjhfj1/autokeras <https://github.com/jhfjhfj1/autokeras>`__

* 
  DARTS: `https://github.com/quark0/darts <https://github.com/quark0/darts>`__

* 
  ENAS: `https://github.com/melodyguan/enas <https://github.com/melodyguan/enas>`__

* 
  NAO: `https://github.com/renqianluo/NAO <https://github.com/renqianluo/NAO>`__

实验说明
----------------------

为了避免算法仅仅在 **CIFAR-10** 数据集上过拟合，还对比了包括 Fashion-MNIST, CIFAR-100, OUI-Adience-Age, ImageNet-10-1 (ImageNet的子集) 和 ImageNet-10-2 (ImageNet 的另一个子集) 在内的其它 5 个数据集。 分别从 ImageNet 中抽取 10 种不同类别标签的子集，组成 ImageNet10-1 和 ImageNet10-2 数据集 。

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 数据集
     - 训练数据集大小
     - 类别标签数
     - 数据集说明
   * - `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`__
     - 60,000
     - 10
     - T恤上衣，裤子，套头衫，连衣裙，外套，凉鞋，衬衫，运动鞋，包和踝靴。
   * - `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`__
     - 50,000
     - 10
     - 飞机，汽车，鸟类，猫，鹿，狗，青蛙，马，船和卡车。
   * - `CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`__
     - 50,000
     - 100
     - 和 CIFAR-10 类似，但总共有 100 个类，每个类有 600 张图。
   * - `OUI-Adience-Age <https://talhassner.github.io/home/projects/Adience/Adience-data.html>`__
     - 26,580
     - 8
     - 8 个年龄组类别 (0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60-)。
   * - `ImageNet-10-1 <http://www.image-net.org/>`__
     - 9,750
     - 10
     - 咖啡杯、电脑键盘、餐桌、衣柜、割草机、麦克风、秋千、缝纫机、里程表和燃气泵。
   * - `ImageNet-10-2 <http://www.image-net.org/>`__
     - 9,750
     - 10
     - 鼓，班吉，口哨，三角钢琴，小提琴，管风琴，原声吉他，长号，长笛和萨克斯。


没有改变源码中的 Fine-tuning 方法。 为了匹配每个任务，改变了源码中模型的输入图片大小和输出类别数目的部分。

所有 NAS 方法模型搜索时间和重训练时间都是 **两天**。  所有结果都是基于 **三次重复实验**。 评估计算机有一块 Nvidia Tesla P100 GPU、112GB 内存和 2.60GHz CPU (Intel E5-2690)。

NAO 需要太多的计算资源，因此只使用提供 Pipeline 脚本的 NAO-WS。

对于 AutoKeras，使用了 0.2.18 版本的代码, 因为这是开始实验时的最新版本。

NAS 结果对比
---------------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - NAS
     - AutoKeras (%)
     - ENAS (macro) (%)
     - ENAS (micro) (%)
     - DARTS (%)
     - NAO-WS (%)
   * - Fashion-MNIST
     - 91.84
     - 95.44
     - 95.53
     - **95.74**
     - 95.20
   * - CIFAR-10
     - 75.78
     - 95.68
     - **96.16**
     - 94.23
     - 95.64
   * - CIFAR-100
     - 43.61
     - 78.13
     - 78.84
     - **79.74**
     - 75.75
   * - OUI-Adience-Age
     - 63.20
     - **80.34**
     - 78.55
     - 76.83
     - 72.96
   * - ImageNet-10-1
     - 61.80
     - 77.07
     - 79.80
     - **80.48**
     - 77.20
   * - ImageNet-10-2
     - 37.20
     - 58.13
     - 56.47
     - 60.53
     - **61.20**


很遗憾，我们无法复现论文中所有的结果。

论文中提供的最佳或平均结果：

.. list-table::
   :header-rows: 1
   :widths: auto

   * - NAS
     - AutoKeras(%)
     - ENAS (macro) (%)
     - ENAS (micro) (%)
     - DARTS (%)
     - NAO-WS (%)
   * - CIFAR- 10
     - 88.56(best)
     - 96.13(best)
     - 97.11(best)
     - 97.17(average)
     - 96.47(best)


AutoKeras，由于其算法中的随机因素，它在所有数据集中的表现相对较差。

对于 ENAS，ENAS（macro）在 OUI-Adience-Age 数据集中表现较好，并且 ENAS（micro）在 CIFAR-10 数据集中表现较好。

对于 DARTS，在某些数据集上具有良好的结果，但在某些数据集中具有比较大的方差。 DARTS 三次实验中的差异在 OUI-Audience-Age 数据集上可达 5.37％（绝对值），在 ImageNet-10-1 数据集上可达4.36％（绝对值）。

NAO-WS 在 ImageNet-10-2 中表现良好，但在 OUI-Adience-Age 中表现非常差。

参考
---------


#. 
   Jin, Haifeng, Qingquan Song, and Xia Hu. "Efficient neural architecture search with network morphism." *arXiv preprint arXiv:1806.10282* (2018).

#. 
   Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018).

#. 
   Pham, Hieu, et al. "Efficient Neural Architecture Search via Parameters Sharing." international conference on machine learning (2018): 4092-4101.

#. 
   Luo, Renqian, et al. "Neural Architecture Optimization." neural information processing systems (2018): 7827-7838.
