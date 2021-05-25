TextNAS
=======

介绍
------------

这是 `TextNAS: A Neural Architecture Search Space tailored for Text Representation <https://arxiv.org/pdf/1912.10729.pdf>`__ 提出的 TextNAS 算法的实现。 TextNAS 是用于文本表示的神经网络架构搜索算法，具体来说，TextNAS 基于由适配各种自然语言任务的操作符所组成的新的搜索空间，TextNAS 还支持单个网络中的多路径集成，来平衡网络的宽度和深度。 

TextNAS 的搜索空间包含： 

.. code-block:: bash

   * 滤波器尺寸为 1, 3, 5, 7 的一维卷积操作 
   * 循环操作符（双向 GRU） 
   * 自注意操作符
   * 池化操作符（最大值、平均值）


遵循 ENAS 算法，TextNAS 也用了参数共享来加速搜索速度，并采用了强化学习的 Controller 来进行架构采样和生成。 参考 TextNAS 论文了解更多细节。

准备
-----------

准备词向量和 SST 数据集，并按如下结构放到 data 目录中：

.. code-block:: bash

   textnas
   ├── data
   │   ├── sst
   │   │   └── trees
   │   │       ├── dev.txt
   │   │       ├── test.txt
   │   │       └── train.txt
   │   └── glove.840B.300d.txt
   ├── dataloader.py
   ├── model.py
   ├── ops.py
   ├── README.md
   ├── search.py
   └── utils.py

以下链接有助于查找和下载相应的数据集：


* `GloVe: Global Vectors for Word Representation <https://nlp.stanford.edu/projects/glove/>`__

  * `glove.840B.300d.txt <http://nlp.stanford.edu/data/glove.840B.300d.zip>`__

* `Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank <https://nlp.stanford.edu/sentiment/>`__

  * `trainDevTestTrees_PTB.zip <https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip>`__

示例
--------

搜索空间
^^^^^^^^^^^^

:githublink:`示例代码 <examples/nas/textnas>`

.. code-block:: bash

   ＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
   git clone https://github.com/Microsoft/nni.git

   # 搜索最优结构
   cd examples/nas/textnas

   # 查看更多的搜索选择
   python3 search.py -h

在每个搜索 Epoch 后，会直接测试 10 个采样的结构。 10 个 Epoch 后的性能预计为 40% - 42%。

默认情况下，20 个采样结构会被导出到 ``checkpoints`` 目录中，以便进行下一步处理。

重新训练
^^^^^^^^^^^^

.. code-block:: bash

   ＃如果未克隆 NNI 代码。 如果代码已被克隆，请忽略此行并直接进入代码目录。
   git clone https://github.com/Microsoft/nni.git

   # 搜索最优结构
   cd examples/nas/textnas

   # default to retrain on sst-2
   sh run_retrain.sh

参考
---------

TextNAS 直接使用了 EnasTrainer，参考 `ENAS <./ENAS.rst>`__ 了解 Trainer 的 API。
