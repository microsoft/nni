#######################
用例与解决方案
#######################

与文档其他部分中展示功能用法的教程和示例不同，本部分主要介绍端到端方案和用例，以帮助用户进一步了解NNI如何为他们提供帮助。 NNI 可广泛应用于各种场景。 除了官方的教程和示例之外，也支持社区贡献者分享自己的自动机器学习实践经验，特别是使用 NNI 的实践经验。

用例与解决方案
=======================
..  toctree::
    :maxdepth: 2

    自动模型调优（HPO/NAS）<automodel>
    自动系统调优（AutoSys）<autosys>
    模型压缩<model_compression>
    特征工程<feature_engineering>
    性能测量，比较和分析<perf_compare>
    在 Google Colab 中使用 NNI <NNI_colab_support>
    自动补全 nnictl 命令 <AutoCompletion>

其它代码库和参考
====================================
经作者许可的一些 NNI 用法示例和相关文档。

外部代码库
===================== 
   * 使用 NNI 的 `矩阵分解超参调优 <https://github.com/microsoft/recommenders/blob/master/examples/04_model_select_and_optimize/nni_surprise_svd.ipynb>`__ 。
   * 使用 NNI 为 scikit-learn 开发的超参搜索 `scikit-nni <https://github.com/ksachdeva/scikit-nni>`__ 。

相关文章
=================
  * `使用AdaptDL 和 NNI进行经济高效的超参调优 - 2021年2月23日 <https://medium.com/casl-project/cost-effective-hyper-parameter-tuning-using-adaptdl-with-nni-e55642888761>`__
  * `（中文博客）NNI v2.0 新功能概述 - 2021年1月21日 <https://www.msra.cn/zh-cn/news/features/nni-2>`__
  * `（中文博客）2019年 NNI 新功能概览 - 2019年12月26日 <https://mp.weixin.qq.com/s/7_KRT-rRojQbNuJzkjFMuA>`__
  * `使用 NNI 为 scikit-learn 开发的超参搜索 - 2019年11月6日 <https://towardsdatascience.com/find-thy-hyper-parameters-for-scikit-learn-pipelines-using-microsoft-nni-f1015b1224c1>`__ 
  * `（中文博客）自动机器学习工具（Advisor、NNI 和 Google Vizier）对比 - 2019年8月5日 <http://gaocegege.com/Blog/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/katib-new#%E6%80%BB%E7%BB%93%E4%B8%8E%E5%88%86%E6%9E%90>`__  
  * `超参优化的对比 <./HpoComparison.rst>`__ 
  * `神经网络架构搜索对比 <./NasComparison.rst>`__ 
  * `TPE 并行化顺序算法 <./ParallelizingTpeSearch.rst>`__ 
  * `自动调优 SVD（在推荐系统中使用 NNI ） <./RecommendersSvd.rst>`__ 
  * `使用 NNI 为 SPTAG 自动调参 <./SptagAutoTune.rst>`__ 
