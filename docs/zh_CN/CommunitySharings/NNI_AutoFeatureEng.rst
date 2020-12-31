.. role:: raw-html(raw)
   :format: html


来自知乎的评论：:raw-html:`<an open source project with highly reasonable design>` - 作者 Garvin Li
========================================================================================================================

本文由 NNI 用户在知乎论坛上发表。 在这篇文章中，Garvin 分享了在使用 NNI 进行自动特征工程方面的体验。 我们认为本文对于有兴趣使用 NNI 进行特征工程的用户非常有用。 经作者许可，将原始文章摘编如下。  

**原文**\ : `如何看待微软最新发布的AutoML平台NNI？By Garvin Li <https://www.zhihu.com/question/297982959/answer/964961829?utm_source=wechat_session&utm_medium=social&utm_oi=28812108627968&from=singlemessage&isappinstalled=0>`__

01 AutoML概述
---------------------

作者认为 AutoML 不光是调参，
也可以针对机器学习过程不同阶段，
包括特征工程、神经网络架构搜索等。

02 NNI 概述
------------------

NNI (Neural Network Intelligence) 是一个微软开源的自动机器学习工具包。
通过自动而有效的方法来帮助用户设计并调优机器学习模型，神经网络架构，
或复杂系统的参数
。

链接： `https://github.com/Microsoft/nni <https://github.com/Microsoft/nni>`__

总体看微软的工具都有一个比较大的特点，
技术可能不一定多新颖，但是设计都非常赞。
NNI 的 AutoFeatureENG 基本包含了用户对于 AutoFeatureENG 的一切幻想。
底层的框架的设计都极为合理。

03 细说NNI - AutoFeatureENG
--------------------------------

本文使用了此项目： `https://github.com/SpongebBob/tabular_automl_NNI <https://github.com/SpongebBob/tabular_automl_NNI>`__。 


新用户可以使用 NNI 轻松高效地进行 AutoFeatureENG。 使用是非常简单的，安装下文件中的 require，然后 pip install NNI。


.. image:: https://pic3.zhimg.com/v2-8886eea730cad25f5ac06ef1897cd7e4_r.jpg
   :target: https://pic3.zhimg.com/v2-8886eea730cad25f5ac06ef1897cd7e4_r.jpg
   :alt: 

NNI把 AutoFeatureENG 拆分成 exploration 和 selection 两个模块。 exploration 主要是特征衍生和交叉，selection 讲的是如何做特征筛选。

04 特征 Exploration
----------------------

对于功能派生，NNI 提供了许多可自动生成新功能的操作， `列表如下 <https://github.com/SpongebBob/tabular_automl_NNI/blob/master/AutoFEOp.rst>`__

**count**：传统的统计，统计一些数据的出现频率

**target**：特征和目标列的一些映射特征

**embedding**：把特征看成句子，用 *word2vector* 的方式制作向量

**crosscount**：特征间除法，有点类似CTR

**aggregete**：特征的 min/max/var/mean

**nunique**：统计唯一特征的数量。

**histsta**：特征存储桶的统计信息，如直方图统计信息。

具体特征怎么交叉，哪一列和哪一列交叉，每一列特征用什么方式衍生呢？可以通过 **search_space. json** 这个文件控制。


.. image:: https://pic1.zhimg.com/v2-3c3eeec6eea9821e067412725e5d2317_r.jpg
   :target: https://pic1.zhimg.com/v2-3c3eeec6eea9821e067412725e5d2317_r.jpg
   :alt: 


图片展示了定义搜索空间的过程。 NNI 为 1 阶运算提供计数编码，并为 2 阶运算提供聚合的统计（min max var mean median nunique）。 

例如，希望以下列方式搜索列名称 {"C1"、"..."，"C26"} 上的频率编码（valuecount）功能的功能：


.. image:: https://github.com/JSong-Jia/Pic/blob/master/images/pic%203.jpg
   :target: https://github.com/JSong-Jia/Pic/blob/master/images/pic%203.jpg
   :alt: 


可以在列 {"C1",...,"C26"} x {"C1",...,"C26"} 上定义交叉频率编码（交叉维度的值计数）方法：


.. image:: https://github.com/JSong-Jia/Pic/blob/master/images/pic%204.jpg
   :target: https://github.com/JSong-Jia/Pic/blob/master/images/pic%204.jpg
   :alt: 


Exploration 的目的就是长生出新的特征。 在代码里可以用 **get_next_parameter** 的方式获取 tuning 的参数：

..

   RECEIVED_PARAMS = nni.get_next_parameter()


05 特征 Selection
--------------------

为了避免特征泛滥的情况，避免过拟合，一定要有 Selection 的机制挑选特征。 在 NNI-AutoFeatureENG 的 Selection 中，主要使用了微软开发的梯度提升框架 LightGBM（Light Gradient Boosting Machine）。


.. image:: https://pic2.zhimg.com/v2-7bf9c6ae1303692101a911def478a172_r.jpg
   :target: https://pic2.zhimg.com/v2-7bf9c6ae1303692101a911def478a172_r.jpg
   :alt: 


了解 xgboost 或者 GBDT 算法同学应该知道，这种树形结构的算法是很容易计算出每个特征对于结果的影响的。 所以使用 lightGBM 可以天然的进行特征筛选。

弊病就是，如果下游是个 *LR* （逻辑回归）这种线性算法，筛选出来的特征是否具备普适性。


.. image:: https://pic4.zhimg.com/v2-d2f919497b0ed937acad0577f7a8df83_r.jpg
   :target: https://pic4.zhimg.com/v2-d2f919497b0ed937acad0577f7a8df83_r.jpg
   :alt: 


06 总结
----------

NNI 的 AutoFeature 模块是给整个行业制定了一个教科书般的标准，告诉大家这个东西要怎么做，有哪些模块，使用起来非常方便。 但是如果只是基于这样简单的模式，不一定能达到很好的效果。

对 NNI 的建议
------------------

我觉得在Exploration方面可以引用一些 DNN（如：xDeepFM） 的特征组合方式，提取更高维度的特征。

在 Selection 方面可以有更多的智能化方案，比如可以基于下游的算法自动选择 Selection 机制。

总之 NNI 在设计曾给了我一些启发，还是一个挺好的开源项目，推荐给大家~ 建议 AI 研究人员使用它来加速研究。

大家用的时候如果是 Mac 电脑可能会遇到 gcc 的问题，因为开源项目自带的脚本是基于 gcc7 编译的， 可以用下面的方法绕过去：

.. code-block:: bash

 brew install libomp
