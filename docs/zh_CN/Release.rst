更改日志
=========

发布 1.9 - 10/22/2020
========================

主要更新
-------------

神经网络架构搜索
^^^^^^^^^^^^^^^^^^^^^^^^^^


* 在 NAS 中增加 regularized evolution 算法 (#2802)
* 在搜索空间集合中增加 NASBench201 (#2766)

模型压缩
^^^^^^^^^^^^^^^^^


* AMC Pruner 改进：支持 resnet，复现 AMC 论文中的实验（示例代码使用默认参数） (#2876 #2906)
* 在一些 Pruner 中支持“约束感知”以提高模型压缩的效率 (#2657)
* 在 TensorFlow 版本的模型压缩代码中支持 "tf.keras.Sequential" (#2887)
* 在模型 FLOPS 计数器中支持自定义的 op (#2795)
* 在 QAT quantizer 中增加量化的偏置 (#2914)

训练平台
^^^^^^^^^^^^^^^^


* 支持在远程模式中使用 "preCommand" 配置 Python 环境 (#2875)
* 在 Windows 下支持 AML 训练平台 (#2882)
* 为远程训练平台添加 reuse 模式 (#2923)

Web 界面和 nnictl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 重新设计 Web 界面的 "Overview" 页面 (#2914)
* 升级 node, yarn 和 FabricUI，激活 Eslint (#2894 #2873 #2744)
* 在超参调优图表和 "Trials detail" 页面的 Trial 表格中增加/删除列 (#2900)
* 美化 Web 界面的 JSON 格式显示 (#2863)
* 支持使用 nnictl 命令自动补全 (#2857)

UT & IT
-------


* 为 Experiment 导入导出数据增加集成测试 (#2878)
* 为用户安装的内置 Tuner 增加集成测试 (#2859)
* 为 nnictl 增加单元测试 (#2912)

文档
-------------


* 重构了模型压缩的文档结构 (#2919)

修复的 Bug
--------------------


* 修复正确使用 naïve evolution Tuner，Trial 失败的 Bug (#2695)
* 修复警告 "WARNING (nni.protocol) IPC pipeline not exists, maybe you are importing tuner/assessor from trial code?" (#2864)
* 修复保存/加载 Experiment 搜索空间的问题 (#2886)
* 修复 Experiment 导入数据的 Bug (#2878)
* 修复远程模式下 annotation 出现的问题 (python 3.8 ast 更新的问题) (#2881)
* 在 Web 界面上自定义 Trial 时，支持为类型是 "choice" 的超参数配置布尔值 (#3003)

发布 1.8 - 8/27/2020
=======================

主要更新
-------------

训练平台
^^^^^^^^^^^^^^^^


* 在 Web 界面直接访问 Trial 日志 (仅支持本地模式) (#2718)
* 添加 OpenPAI Trial Job 详情链接 (#2703)
* 在可重用的环境中支持 GPU 调度器 (#2627) (#2769)
* 为 ``trial_runner`` 中的 ``web_channel`` 增加超时时间 (#2710)
* 在 AzureML 模式下展示环境配置错误信息 (#2724)
* 为在 OpenPAI 模式复制数据增加更多日志信息 (#2702)

Web 界面，nnictl 和 nnicli
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 改进超参数并行坐标图的绘制 (#2691) (#2759)
* 为 Trial Job 列表添加分页 (#2738) (#2773)
* 使面板可以在鼠标点击其它区域时关闭 (#2734)
* 从 Web 界面中去掉多阶段支持 (#2760)
* 支持保存和加载 Experiment (#2750)
* 在导出结果的命令中增加导出中间结果的选项 (#2706)
* 增加了依据最高/最低指标列出 Trial 的 `命令 <https://github.com/microsoft/nni/blob/v1.8/docs/zh_CN/Tutorial/Nnictl.rst#nnictl-trial>`__ (#2747)
* 提升了 `nnicli <https://github.com/microsoft/nni/blob/v1.8/docs/zh_CN/nnicli_ref.rst>`__ 的用户体验，并附上 `示例 <https://github.com/microsoft/nni/blob/v1.8/examples/notebooks/retrieve_nni_info_with_python.ipynb>`__ (#2713)

神经网络架构搜索
^^^^^^^^^^^^^^^^^^^^^^^^^^


* `搜索空间集合：ENAS and DARTS <https://github.com/microsoft/nni/blob/v1.8/docs/zh_CN/NAS/SearchSpaceZoo.rst>`__ (#2589)
* 用于在 NAS 基准测试中查询中间结果的 API (#2728)

模型压缩
^^^^^^^^^^^^^^^^^


* 支持 TorchModuleGraph 的 List/Tuple Construct/Unpack 操作 (#2609)
* 模型加速改进: 支持 DenseNet 和 InceptionV3 (#2719)
* 支持多个连续 tuple 的 unpack 操作 (#2768)
* `比较支持的 Pruner 的表现的文档 <https://github.com/microsoft/nni/blob/v1.8/docs/zh_CN/CommunitySharings/ModelCompressionComparison.rst>`__ (#2742)
* 新的 Pruner：`Sensitivity pruner <https://github.com/microsoft/nni/blob/v1.8/docs/zh_CN/Compressor/Pruner.md#sensitivity-pruner>`__ (#2684) and `AMC pruner <https://github.com/microsoft/nni/blob/v1.8/docs/zh_CN/Compressor/Pruner.rst>`__ (#2573) (#2786)
* 支持 TensorFlow v2 的模型压缩 (#2755)

不兼容的改动
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 默认 Experiment 目录从 ``$HOME/nni/experiments`` 更新至 ``$HOME/nni-experiments``。 如果希望查看通过之前的 NNI 版本创建的 Experiment，可以将这些 Experiment 目录从 ``$HOME/nni/experiments`` 手动移动至 ``$HOME/nni-experiments``。 (#2686) (#2753)
* 不再支持 Python 3.5 和 scikit-learn 0.20 (#2778) (#2777) (2783) (#2787) (#2788) (#2790)

其它
^^^^^^


* 更新 Docker 镜像中的 Tensorflow 版本 (#2732) (#2735) (#2720)

示例
--------


* 在 Assessor 示例中移除 gpuNum (#2641)

文档
-------------


* 改进自定义 Tuner 的文档 (#2628)
* 修复几处文档中的输入错误和语法错误 (#2637 #2638, 感谢 @tomzx)
* 改进 AzureML 训练平台的文档 (#2631)
* 改进中文翻译的 CI 流程 (#2654)
* 改进 OpenPAI 训练平台的文档 (#2685)
* 改进社区分享的文档 (#2640)
* 增加对 Colab 进行支持的教程 (#2700)
* 改进模型压缩的文档结构 (#2676)

修复的 Bug
----------------------


* 修复训练平台的目录生成错误 (#2673)
* 修复 Remote 训练平台使用 chmod 时的 Bug (#2689)
* 通过内联导入 ``_graph_utils`` 修复依赖问题 (#2675)
* 修复了 ``SimulatedAnnealingPruner`` 中的掩码问题 (#2736)
* 修复了中间结果的图的缩放问题 (#2738)
* 修复了在查询 NAS 基准测试时字典没有经过排序的问题 (#2728)
* 修复了 Gradient Selector Dataloader Iterator 的导入问题 (#2690)
* 修复了对在 Remote 训练平台下添加数十台机器的支持 (#2725)
* 修复了 Web 界面的几个样式问题 (#2762 #2737)
* 修复了对指标中不常见类型的支持，包括 NaN 和 Infinity (#2782)
* 修复 nnictl experiment delete (#2791)

发布 1.7 - 7/8/2020
======================

主要功能
--------------

训练平台
^^^^^^^^^^^^^^^^


* 支持 AML (Azure Machine Learning) 作为训练平台。
* OpenPAI 任务可被重用。 当 Trial 完成时， OpenPAI 任务不会停止， 而是等待下一个 Trial。 改进 `新的 OpenPAI 模式的文档 <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/TrainingService/PaiMode.rst#openpai-configurations>`__.
* `支持在向训练平台上传代码目录时使用 .nniignore 忽略代码目录中的文件和目录 <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/TrainingService/Overview.rst#how-to-use-training-service>`__.

神经网络架构搜索（NAS）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  `为 NAS 基准测试 (NasBench101, NasBench201, NDS) 提供了友好的 API <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/NAS/Benchmarks.rst>`__。

* 
  `在 TensorFlow 2.X 支持 Classic NAS（即非权重共享模式） <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/NAS/ClassicNas.rst>`__。

模型压缩
^^^^^^^^^^^^^^^^^


* 改进模型加速：跟踪层之间的更多依赖关系，自动解决掩码冲突，支持剪枝 ResNet 的加速
* 增加新的 Pruner，包括三个模型剪枝算法： `NetAdapt Pruner <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/Compressor/Pruner.md#netadapt-pruner>`__\ , `SimulatedAnnealing Pruner <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/Compressor/Pruner.md#simulatedannealing-pruner>`__\ , `AutoCompress Pruner <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/Compressor/Pruner.md#autocompress-pruner>`__\ , and `ADMM Pruner <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/Compressor/Pruner.rst#admm-pruner>`__.
* 增加 `模型灵敏度分析工具 <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/Compressor/CompressionUtils.rst>`__ 来帮助用户发现各层对剪枝的敏感性。
* 
  `用于模型压缩和 NAS 的简易 FLOPs 计算工具 <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/Compressor/CompressionUtils.rst#model-flops-parameters-counter>`__.

* 
  更新 Lottery Ticket Pruner 以导出中奖彩票

示例
^^^^^^^^


* 在 NNI 上使用新的 `自定义 Tuner OpEvo <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/TrialExample/OpEvoExamples.rst>`__ 自动优化张量算子。

内置 Tuner、Assessor、Advisor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `允许自定义 Tuner、Assessor、Advisor 被安装为内置算法 <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/Tutorial/InstallCustomizedAlgos.rst>`__.

Web 界面
^^^^^^^^^^^^^^


* 支持更友好的嵌套搜索空间可视化。
* 在超参数图中展示 Trial 的字典的键
* 增强 Trial 持续时间展示

其它
^^^^^^


* 提供工具函数用于合并从 NNI 获取到的参数
* 支持在 OpenPAI 模式中设置 paiStorageConfigName

文档
-------------


* 改进 `模型压缩文档 <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/Compressor/Overview.rst>`__
* 改进 `NAS 基准测试的文档 <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/NAS/Benchmarks.rst>`__
  和 `示例 <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/NAS/BenchmarksExample.ipynb>`__ 。
* 改进 `AzureML 训练平台的文档 <https://github.com/microsoft/nni/blob/v1.7/docs/zh_CN/TrainingService/AMLMode.rst>`__
* 主页迁移到 readthedoc。

修复的 Bug
------------------


* 修复模型图中含有共享的 nn.Module 时的问题
* 修复 ``make build`` 时的 nodejs OOM
* 修复 NASUI Bug
* 修复持续时间和中间结果图片更新问题
* 修复小的 Web 界面表格样式问题

发布 1.6 - 5/26/2020
-----------------------

主要功能
^^^^^^^^^^^^^^

新功能和改进
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 将 IPC 限制提高至 100W
* 修改非本机训练平台中，将上传代码到存储的逻辑
* SDK 版本支持 ``__version__``
* 支持 Windows 下开发模式安装

Web 界面
^^^^^^^^^^^^^^^


* 显示 Trial 的错误消息
* 完善主页布局
* 重构概述页面的最佳 Trial 模块
* 从 Web 界面中去掉多阶段支持
* 在概述页面为 Trial 并发添加工具提示。
* 在超参图中显示最好的 Trial

超参优化更新
^^^^^^^^^^^^^^^^^^


* 改进 PBT 的错误处理，并支持恢复 Experiment

NAS 更新
^^^^^^^^^^^


* NAS 支持 TensorFlow 2.0 (预览版) `TF2.0 NAS 示例 <https://github.com/microsoft/nni/tree/v1.6/examples/nas/naive-tf>`__
* LayerChoice 使用 OrderedDict
* 优化导出格式
* 应用固定架构后，将 LayerChoice 替换成选择的模块

模型压缩改进
^^^^^^^^^^^^^^^^^^^^^^^^^


* 模型压缩支持 PyTorch 1.4

训练平台改进
^^^^^^^^^^^^^^^^^^^^^^^^


* 改进 OpenPAI YAML 的合并逻辑
* 支持 Windows 在远程模式中作为远程机器 `远程模式 <https://github.com/microsoft/nni/blob/v1.6/docs/zh_CN/TrainingService/RemoteMachineMode.rst#windows>`__


修复的 Bug
^^^^^^^^^^^^^^


* 修复开发模式安装
* 当检查点没有 state_dict 时，SPOS 示例会崩溃
* 修复失败 Trial 造成的表格排序问题
* 支持多 Python 环境（如 conda，pyenv 等）

发布 1.5 - 4/13/2020
-----------------------

新功能和文档
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

超参优化
^^^^^^^^^^^^^^^^^^^^^^^^^^


* 全新 Tuner： `Population Based Training (PBT) <https://github.com/microsoft/nni/blob/v1.5/docs/zh_CN/Tuner/PBTTuner.rst>`__
* Trial 现在可以返回无穷大和 NaN 结果

神经网络架构搜索
^^^^^^^^^^^^^^^^^^^^^^^^^^


* 全新 NAS 算法：`TextNAS <https://github.com/microsoft/nni/blob/v1.5/docs/zh_CN/NAS/TextNAS.rst>`__
* 在 Web 界面 支持 ENAS 和 DARTS的 `可视化 <https://github.com/microsoft/nni/blob/v1.5/docs/zh_CN/NAS/Visualization.rst>`__ 

模型压缩
^^^^^^^^^^^^^^^^^


* 全新 Pruner: `GradientRankFilterPruner <https://github.com/microsoft/nni/blob/v1.5/docs/zh_CN/Compressor/Pruner.rst#gradientrankfilterpruner>`__
* 默认情况下，Compressor 会验证配置
* 重构：可将优化器作为 Pruner 的输入参数，从而更容易支持 DataParallel 和其它迭代剪枝方法。 这是迭代剪枝算法用法上的重大改动。
* 重构了模型压缩示例
* 改进 `模型压缩算法 <https://github.com/microsoft/nni/blob/v1.5/docs/zh_CN/Compressor/Framework.rst>`__

训练平台
^^^^^^^^^^^^^^^^


* Kubeflow 现已支持 pytorchjob crd v1 (感谢贡献者 @jiapinai)
* 实验性地支持 `DLTS <https://github.com/microsoft/nni/blob/v1.5/docs/zh_CN/TrainingService/DLTSMode.rst>`__ 

文档的整体改进
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 语法、拼写以及措辞上的修改 (感谢贡献者 @AHartNtkn)

修复的 Bug
^^^^^^^^^^^^^^^^^^^^^^^^^


* ENAS 不能使用多个 LSTM 层 (感谢贡献者 @marsggbo)
* NNI 管理器的计时器无法取消订阅 (感谢贡献者 @guilhermehn)
* NNI 管理器可能会耗尽内存 (感谢贡献者 @Sundrops)
* 批处理 Tuner 不支持自定义 Trial （#2075）
* Experiment 启动失败后，无法终止 (#2080)
* 非数字的指标会破坏网页界面 (#2278)
* lottery ticket Pruner 中的 Bug
* 其它小问题

发布 1.4 - 2/19/2020
-----------------------

主要功能
^^^^^^^^^^^^^^

神经网络架构搜索
^^^^^^^^^^^^^^^^^^^^^^^^^^


* 支持 `C-DARTS <https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/NAS/CDARTS.rst>`__ 算法并增加 `the 示例 <https://github.com/microsoft/nni/tree/v1.4/examples/nas/cdarts>`__ using it
* 初步支持 `ProxylessNAS <https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/NAS/Proxylessnas.rst>`__ 并增加 `示例 <https://github.com/microsoft/nni/tree/v1.4/examples/nas/proxylessnas>`__
* 为 NAS 框架增加单元测试

模型压缩
^^^^^^^^^^^^^^^^^


* 为压缩模型增加 DataParallel，并提供 `示例 <https://github.com/microsoft/nni/blob/v1.4/examples/model_compress/multi_gpu.py>`__
* 支持模型压缩的 `加速 <https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/Compressor/ModelSpeedup.rst>`__ （试用版）

训练平台
^^^^^^^^^^^^^^^^


* 通过允许指定 OpenPAI 配置文件路径，来支持完整的 OpenPAI 配置
* 为新的 OpenPAI 模式（又称，paiK8S）增加示例配置 YAML 文件
* 支持删除远程模式下使用 sshkey 的 Experiment （感谢外部贡献者 @tyusr）

Web 界面
^^^^^^^^^^^^


* Web 界面重构：采用 fabric 框架

其它
^^^^^^


* 支持 `在前台运行 NNI 的 Experiment <https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/Tutorial/Nnictl#manage-an-experiment>`__\ , 即 ``nnictl create/resume/view`` 的 ``--foreground`` 参数
* 支持取消 UNKNOWN 状态的 Trial。
* 支持最大 50MB 的搜索空间文件 （感谢外部贡献者 @Sundrops）

文档
^^^^^^^^^^^^^


* 改进 NNI readthedocs 的 `索引目录结果 <https://nni.readthedocs.io/zh/latest/>`__ of NNI readthedocs
* 改进 `NAS 文档 <https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/NAS/NasGuide.rst>`__
* 增加 `PAI 模式的文档 <https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/TrainingService/PaiMode.rst>`__
* 为 `NAS <https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/NAS/QuickStart.md>`__ 和 `模型压缩 <https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/Compressor/QuickStart.md>`__ 增加快速入门指南
* 改进 `EfficientNet 的文档 <https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/TrialExample/EfficientNet.rst>`__

修复的 Bug
^^^^^^^^^^^^^^^^^^^


* 修复在指标数据和 JSON 格式中对 NaN 的支持
* 修复搜索空间 ``randint`` 类型的 out-of-range Bug
* 修复模型压缩中导出 ONNX 模型时的错误张量设备的 Bug
* 修复新 OpenPAI 模式（又称，paiK8S）下，错误处理 nnimanagerIP 的 Bug

发布 1.3 - 12/30/2019
------------------------

主要功能
^^^^^^^^^^^^^^

支持神经网络架构搜索算法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 增加 `但路径一次性 <https://github.com/microsoft/nni/tree/v1.3/examples/nas/spos/>`__ 算法和示例

模型压缩算法支持
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 增加 `知识蒸馏 <https://github.com/microsoft/nni/blob/v1.3/docs/zh_CN/TrialExample/KDExample.rst>`__ 算法和示例
* Pruners

  * `L2Filter Pruner <https://github.com/microsoft/nni/blob/v1.3/docs/zh_CN/Compressor/Pruner.rst#3-l2filter-pruner>`__
  * `ActivationAPoZRankFilterPruner <https://github.com/microsoft/nni/blob/v1.3/docs/zh_CN/Compressor/Pruner.md#1-activationapozrankfilterpruner>`__
  * `ActivationMeanRankFilterPruner <https://github.com/microsoft/nni/blob/v1.3/docs/zh_CN/Compressor/Pruner.md#2-activationmeanrankfilterpruner>`__

* `BNN Quantizer <https://github.com/microsoft/nni/blob/v1.3/docs/zh_CN/Compressor/Quantizer.md#bnn-quantizer>`__
  训练平台
* 
  OpenPAI 的 NFS 支持

    从 OpenPAI v0.11开始，HDFS 不再用作默认存储，可将 NFS、AzureBlob 或其他存储用作默认存储。 在本次版本中，NNI 扩展了对 OpenPAI 最近改动的支持，可与 OpenPAI v0.11 及后续版本的默认存储集成。

* 
  Kubeflow 更新适配

    适配 Kubeflow 0.7 对 tf-operator 的新支持。

工程（代码和生成自动化）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 启用 `ESLint <https://eslint.org/>`__ 静态代码分析

小改动和 Bug 修复
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 正确识别内置 Tuner 和定制 Tuner
* Dispatcher 基类的日志
* 修复有时 Tuner、Assessor 的失败会终止 Experiment 的 Bug。
* 修复本机作为远程计算机的 `问题 <https://github.com/microsoft/nni/issues/1852>`__
* SMAC Tuner 中 Trial 配置的去重 `ticket <https://github.com/microsoft/nni/issues/1364>`__

发布 1.2 - 12/02/2019
------------------------

主要功能
^^^^^^^^^^^^^^


* `特征工程 <https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/FeatureEngineering/Overview.rst>`__

  * 新增特征工程接口
  * 新增特征选择算法：`Gradient feature selector <https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/FeatureEngineering/GradientFeatureSelector.md>`__ & `GBDT selector <https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/FeatureEngineering/GBDTSelector.md>`__
  * `特征工程示例 <https://github.com/microsoft/nni/tree/v1.2/examples/feature_engineering>`__

* 神经网络结构搜索在 NNI 上的应用

  * `全新 NAS 接口 <https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/NAS/NasInterface.md>`__
  * NAS 算法：`ENAS <https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/NAS/Overview.md#enas>`__\ , `DARTS <https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/NAS/Overview.md#darts>`__\ , `P-DARTS <https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/NAS/Overview.rst#p-darts>`__ (PyTorch)
  * 经典模式下的 NAS（每次 Trial 独立运行）

* 模型压缩

  * `全新模型剪枝算法 <https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/Compressor/Overview.md>`__: lottery ticket 修剪, L1Filter Pruner, Slim Pruner, FPGM Pruner
  * `全新模型量化算法 <https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/Compressor/Overview.md>`__\ : QAT quantizer, DoReFa quantizer
  * 支持导出压缩后模型的 API。

* 训练平台

  * 支持 OpenPAI 令牌身份验证

* 示例：

  * `使用 NNI 自动调优 rocksdb 配置示例 <https://github.com/microsoft/nni/tree/v1.2/examples/trials/systems/rocksdb-fillrandom>`__.
  * `支持 TensorFlow 2.0 的 MNIST Trial 示例 <https://github.com/microsoft/nni/tree/v1.2/examples/trials/mnist-tfv2>`__.

* 改进

  * 远程训练平台中不需要 GPU 的 Trial 任务改为使用随机调度，不再使用轮询调度。
  * 添加 pylint 规则来检查拉取请求，新的拉取请求需要符合 `pylint 规则 <https://github.com/microsoft/nni/blob/v1.2/pylintrc>`__。

* Web 门户和用户体验

  * 支持用户添加自定义 Trial。
  * 除了超参外，用户可放大缩小详细图形。

* 文档

  * 改进了 NNI API 文档，增加了更多的 docstring。

Bug 修复
^^^^^^^^^^^^^


* 修复当失败的 Trial 没有指标时，表格的排序问题。 -Issue #1773
* 页面切换时，保留选择的（最大、最小）状态。 -PR#1710
* 使超参数图的默认指标 yAxis 更加精确。 -PR#1736
* 修复 GPU 脚本权限问题。 -Issue #1665

发布 1.1 - 10/23/2019
------------------------

主要功能
^^^^^^^^^^^^^^


* 全新 tuner: `PPO Tuner <https://github.com/microsoft/nni/blob/v1.1/docs/zh_CN/Tuner/PPOTuner.md>`__
* `查看早停 Experiment <https://github.com/microsoft/nni/blob/v1.1/docs/zh_CN/Tutorial/Nnictl.md#view>`__
* Tuner 可使用专门的 GPU 资源（参考 `tutorial <https://github.com/microsoft/nni/blob/v1.1/docs/zh_CN/Tutorial/ExperimentConfig.md>`__ 中的 ``gpuIndices`` 了解详情）
* 改进 WEB 界面

  * Trial 详情页面可列出每个 Trial 的超参，以及开始结束时间（需要通过 "add column" 添加）
  * 优化大型 Experiment 的显示性能

* 更多示例

  * `EfficientNet PyTorch 示例 <https://github.com/ultmaster/EfficientNet-PyTorch>`__
  * `Cifar10 NAS 示例 <https://github.com/microsoft/nni/blob/v1.1/examples/trials/nas_cifar10/README.rst>`__

* `模型压缩工具包 - Alpha 阶段 <https://github.com/microsoft/nni/blob/v1.1/docs/zh_CN/Compressor/Overview.md>`__：我们很高兴的宣布 NNI 的模型压缩工具包发布了。它还处于试验阶段，会根据使用反馈来改进。 诚挚邀请您使用、反馈，或更多贡献

修复的 Bug
^^^^^^^^^^


* 当搜索空间结束后，多阶段任务会死锁 (issue #1204)
* 没有日志时，``nnictl`` 会失败 (issue #1548)

发布1.0 - 9/2/2019
----------------------

主要功能
^^^^^^^^^^^^^^


* 
  Tuners 和 Assessors


  * 支持自动特征生成和选择 -Issue#877  -PR #1387

    * 提供自动特征接口
    * 基于 Beam 搜索的 Tuner
    * `增加 Pakdd 示例 <https://github.com/microsoft/nni/tree/v1.9/examples/trials/auto-feature-engineering>`__

  * 添加并行算法提高 TPE 在高并发下的性能。  -PR #1052
  * 为 hyperband 支持多阶段    -PR #1257

* 
  训练平台


  * 支持私有 Docker Registry -PR #755


  * 改进

    * 增加 RestFUL API 的 Python 包装，支持通过代码获取指标的值  PR #1318
    * 新的 Python API : get_experiment_id(), get_trial_id()  -PR #1353   -Issue #1331 &amp; -Issue#1368
    * 优化 NAS 搜索空间 -PR #1393

      * 使用 _type 统一 NAS 搜索空间 -- "mutable_type"e
      * 更新随机搜索 Tuner

    * 将 gpuNum 设为可选      -Issue #1365
    * 删除 OpenPAI 模式下的 outputDir 和 dataDir 配置   -Issue #1342
    * 在 Kubeflow 模式下创建 Trial 时，codeDir 不再被拷贝到 logDir   -Issue #1224

* 
  Web 门户和用户体验


  * 在 Web 界面的搜索过程中显示最好指标的曲线  -Issue #1218
  * 在多阶段 Experiment 中，显示参数列表的当前值   -Issue1210  -PR #1348
  * 在 AddColumn 中增加 "Intermediate count" 选项。      -Issue #1210
  * 在 Web 界面中支持搜索参数的值 -Issue #1208
  * 在默认指标图中，启用指标轴的自动缩放   -Issue #1360
  * 在命令行中为 nnictl 命令增加详细文档的连接    -Issue #1260
  * 用户体验改进：显示 Error 日志 -Issue #1173

* 
  文档


  * 更新文档结构  -Issue #1231
  * (已删除) 多阶段文档的改进 -Issue #1233 -PR #1242

    * 添加配置示例

  * `改进 WebUI 描述 <Tutorial/WebUI.rst>`__  -PR #1419

Bug 修复
^^^^^^^^^^^^^^^^^^


* (Bug 修复)修复 0.9 版本中的链接  -Issue #1236
* (Bug 修复)自动完成脚本
* (Bug 修复) 修复管道中仅检查脚本中最后一个命令退出代码的问题。  -PR #1417
* (Bug 修复) Tuner 的 quniform -Issue #1377
* (Bug fix) 'quniform' 在 GridSearch 和其它 Tuner 之间的含义不同。   * -Issue #1335
* (Bug 修复)"nnictl experiment list" 将 "RUNNING" 状态的 Experiment 显示为了 "INITIALIZED" -PR #1388
* (Bug 修复) 在 NNI dev 安装模式下无法安装 SMAC。   -Issue #1376
* (Bug 修复) 无法点击中间结果的过滤按钮   -Issue #1263
* (Bug 修复) API "/api/v1/nni/trial-jobs/xxx" 在多阶段 Experiment 无法显示 Trial 的所有参数    -Issue #1258
* (Bug 修复) 成功的 Trial 没有最终结果，但 Web 界面显示成了 ×××(FINAL)  -Issue #1207
* (Bug 修复) nnictl stop -Issue #1298
* (Bug 修复) 修复安全警告
* (Bug 修复) 超参页面损坏 -Issue #1332
* (Bug 修复) 运行 flake8 测试来查找 Python 语法错误和未定义的名称 -PR #1217

发布 0.9 - 7/1/2019
----------------------

主要功能
^^^^^^^^^^^^^^


* 生成 NAS 编程接口

  * 为 NAS 接口增加 ``enas-mode``  and ``oneshot-mode``: `PR #1201 <https://github.com/microsoft/nni/pull/1201#issue-291094510>`__

* 
  `有 Matern 核的高斯过程 Tuner <Tuner/GPTuner.rst>`__

* 
  (已删除) 支持多阶段 Experiment


  * 为多阶段 Experiment 增加新的训练平台：pai 模式从 v0.9 开始支持多阶段 Experiment。
  * 为以下内置 Tuner 增加多阶段的功能：

    * TPE, Random Search, Anneal, Naïve Evolution, SMAC, Network Morphism, Metis Tuner。

* 
  Web 界面


  * 在 Web 界面中可比较 Trial。 详情参考 `查看 Trial 状态 <Tutorial/WebUI.rst>`__
  * 允许用户调节 Web 界面的刷新间隔。 详情参考 `查看 Summary 界面 <Tutorial/WebUI.rst>`__
  * 更友好的显示中间结果。 详情参考 `查看 Trial 状态 <Tutorial/WebUI.rst>`__

* `命令行接口 <Tutorial/Nnictl.rst>`__

  * ``nnictl experiment delete``：删除一个或多个 Experiment，包括其日志，结果，环境信息核缓存。 用于删除无用的 Experiment 结果，或节省磁盘空间。
  * ``nnictl platform clean``：用于清理目标平台的磁盘空间。 所提供的 YAML 文件包括了目标平台的信息，与 NNI 配置文件的格式相同。
    Bug 修复和其它更新

* 改进 Tuner 安装过程：增加 < `sklearn <https://scikit-learn.org/stable/>`__ 依赖。
* (Bug 修复) 连接 OpenPAI 失败的 HTTP 代码 - `Issue #1076 <https://github.com/microsoft/nni/issues/1076>`__
* (Bug 修复) 为 OpenPAI 平台验证文件名 - `Issue #1164 <https://github.com/microsoft/nni/issues/1164>`__
* (Bug 修复) 更新 Metis Tunerz 中的 GMM
* (Bug 修复) Web 界面负数的刷新间隔时间 - `Issue #1182 <https://github.com/microsoft/nni/issues/1182>`__ , `Issue #1185 <https://github.com/microsoft/nni/issues/1185>`__
* (Bug 修复) 当只有一个超参时，Web 界面的超参无法正确显示 - `Issue #1192 <https://github.com/microsoft/nni/issues/1192>`__

发布 0.8 - 6/4/2019
----------------------

主要功能
^^^^^^^^^^^^^^


* 在 Windows 上支持 NNI 的 OpenPAI 和远程模式

  * NNI 可在 Windows 上使用 OpenPAI 模式
  * NNI 可在 Windows 上使用 OpenPAI 模式

* GPU 的高级功能

  * 在本机或远程模式上，可在同一个 GPU 上运行多个 Trial。
  * 在已经运行非 NNI 任务的 GPU 上也能运行 Trial

* 支持 Kubeflow v1beta2 操作符

  * 支持 Kubeflow TFJob/PyTorchJob v1beta2

* `通用 NAS 编程接口 <https://github.com/microsoft/nni/blob/v0.8/docs/zh_CN/GeneralNasInterfaces.md>`__

  * 实现了 NAS 的编程接口，可通过 NNI Annotation 很容易的表达神经网络架构搜索空间
  * 提供新命令 ``nnictl trial codegen`` 来调试 NAS 代码生成部分
  * 提供 NAS 编程接口教程，NAS 在 MNIST 上的示例，用于 NAS 的可定制的随机 Tuner

* 支持在恢复 Experiment 时，同时恢复 Tuner 和 Advisor 的状态
* 在恢复 Experiment 时，Tuner 和 Advisor 会导入已完成的 Trial 的数据。
* Web 界面

  * 改进拷贝 Trial 参数的设计
  * 在 hyper-parameter 图中支持 'randint' 类型
  * 使用 ComponentUpdate 来避免不必要的刷新

Bug 修复和其它更新
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 修复 ``nnictl update`` 不一致的命令行风格
* SMAC Tuner 支持导入数据
* 支持 Experiment 状态从 ERROR 回到 RUNNING
* 修复表格的 Bug
* 优化嵌套搜索空间
* 优化 'randint' 类型，并支持下限
* `超参调优算法的对比 <CommunitySharings/HpoComparison.rst>`__
* `NAS 算法对比 <CommunitySharings/NasComparison.rst>`__
* `NNI 在推荐上的应用 <CommunitySharings/RecommendersSvd.rst>`__

发布 0.7 - 4/29/2018
-----------------------

主要功能
^^^^^^^^^^^^^^


* `在 WIndows 上支持 NNI <Tutorial/InstallationWin.rst>`__

  * NNI 可在 Windows 上使用本机模式

* `全新 advisor: BOHB <Tuner/BohbAdvisor.rst>`__

  * 支持新的 BOHB Advisor，这是一个健壮而有效的超参调优算法，囊括了贝叶斯优化和 Hyperband 的优点

* `支持通过 nnictl 来导入导出 Experiment 数据 <Tutorial/Nnictl.rst>`__

  * 在 Experiment 执行完后，可生成分析结果报告
  * 支持将先前的调优数据导入到 Tuner 和 Advisor 中

* `为 NNI Trial 任务指定 GPU 设备 <Tutorial/ExperimentConfig.rst#localConfig>`__

  * 通过 gpuIndices 配置来为 Trial 任务指定GPU。如果 Experiment 配置文件中有 gpuIndices，则只有指定的 GPU 会被用于 NNI 的 Trial 任务。

* 改进 Web 界面

  * 在 Web 界面上使用十进制格式的指标
  * 添加多阶段训练相关的提示
  * 可将超参复制为 Python dict 格式
  * 可将提前终止的 Trial 数据传入 Tuner。

* 为 nnictl 提供更友好的错误消息

  * 为 YAML 文件格式错误提供更有意义的错误信息

Bug 修复
^^^^^^^^^^^^^^


* 运行 nnictl stop 的异步 Dispatcher 模式时，无法杀掉所有的 Python 线程
* nnictl --version 不能在 make dev-install 下使用
* OpenPAI 平台下所有的 Trial 任务状态都是 'WAITING'

发布 0.6 - 4/2/2019
----------------------

主要功能
^^^^^^^^^^^^^^


* `版本检查 <TrainingService/PaiMode.rst>`__

  * 检查 nniManager 和 trialKeeper 的版本是否一致

* `为早停的任务报告最终指标 <https://github.com/microsoft/nni/issues/776>`__

  * 如果 includeIntermediateResults 为 true，最后一个 Assessor 的中间结果会被发送给 Tuner 作为最终结果。 includeIntermediateResults 的默认值为 false。

* `独立的 Tuner/Assessor <https://github.com/microsoft/nni/issues/841>`__

  * 增加两个管道来分离 Tuner 和 Assessor 的消息

* 使日志集合功能可配置
* 为所有 Trial 增加中间结果的视图

Bug 修复
^^^^^^^^^^^^^^


* `为 OpenPAI 增加 shmMB 配置 <https://github.com/microsoft/nni/issues/842>`__
* 修复在指标为 dict 时，无法显示任何结果的 Bug。
* 修复 hyperband 中浮点类型的计算问题
* 修复 SMAC Tuner 中搜索空间转换的错误
* 修复 Web 界面中解析 Experiment 的错误格式
* 修复 Metis Tuner 冷启动时的错误

发布 0.5.2 - 3/4/2019
------------------------

改进
^^^^^^^^^^^^


* 提升 Curve fitting Assessor 的性能。

文档
^^^^^^^^^^^^^


* 发布中文文档网站：https://nni.readthedocs.io/zh/latest/
* 调试和维护：https://nni.readthedocs.io/zh/latest/Tutorial/HowToDebug.html
* Tuner、Assessor 参考：https://nni.readthedocs.io/zh/latest/sdk_reference.html#tuner

Bug 修复和其它更新
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 修复了在某些极端条件下，不能正确存储任务的取消状态。
* 修复在使用 SMAC Tuner 时，解析搜索空间的错误。
* 修复 CIFAR-10 样例中的 broken pipe 问题。
* 为本地训练服务和 NNI 管理器添加单元测试。
* 为远程服务器、OpenPAI 和 Kubeflow 训练平台在 Azure 中增加集成测试。
* 在 OpenPAI 客户端中支持 Pylon 路径。

发布 0.5.1 - 1/31/2018
-------------------------

改进
^^^^^^^^^^^^


* 支持配置 `log 目录 <https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig.md>`__ 
* 支持 `不同级别的日志 <https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig.md>`__，使其更易于调试。

文档
^^^^^^^^^^^^^


* 重新组织文档，新的主页位置：https://nni.readthedocs.io/zh/latest/

Bug 修复和其它更新
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 修复了 Python 虚拟环境中安装的 Bug，并重构了安装逻辑。
* 修复了在最新的 OpenPAI 下存取 HDFS 失败的问题。
* 修复了有时刷新 stdout 会造成 Experiment 崩溃的问题。

发布 0.5.0 - 01/14/2019
--------------------------

主要功能
^^^^^^^^^^^^^^

支持新的 Tuner 和 Assessor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 支持 `Metis tuner <Tuner/MetisTuner.rst>`__ 对于\ **在线**\ 超参调优的场景，Metis 算法已经被证明非常有效。
* 支持 `ENAS customized tuner <https://github.com/countif/enas_nni>`__。由 GitHub 社区用户所贡献。它是神经网络的搜索算法，能够通过强化学习来学习神经网络架构，比 NAS 的性能更好。
* 支持 `Curve fitting assessor <Assessor/CurvefittingAssessor.rst>`__，通过曲线拟合的策略来实现提前终止 Trial。
* `权重共享的 <https://github.com/microsoft/nni/blob/v0.5/docs/AdvancedNAS.md>`__ 高级支持：为 NAS Tuner 提供权重共享，当前支持 NFS。

改进训练平台
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `FrameworkController 训练服务 <TrainingService/FrameworkControllerMode.rst>`__：支持使用在 Kubernetes 上使用 FrameworkController 运行。

  * FrameworkController 是 Kubernetes 上非常通用的控制器（Controller），能用来运行基于各种机器学习框架的分布式作业，如 TensorFlow，Pytorch， MXNet 等。
  * NNI 为作业定义了统一而简单的规范。
  * 如何使用 FrameworkController 的 MNIST 样例。

改进用户体验
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 为 OpenPAI, Kubeflow 和 FrameworkController 模式提供更好的日志支持。

  * 改进后的日志架构能将尝试的 stdout/stderr 通过 HTTP POST 方式发送给 NNI 管理器。 NNI 管理器将 Trial 的 stdout/stderr 消息存储在本地日志文件中。
  * 在 WEB 界面上显示 Trial 日志的链接。

* 支持将最终结果显示为键值对。

发布 0.4.1 - 12/14/2018
--------------------------

主要功能
^^^^^^^^^^^^^^

支持新的 Tuner
^^^^^^^^^^^^^^^^^^


* 支持 `network morphism <Tuner/NetworkmorphismTuner.rst>`__ Tuner

改进训练服务
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


将 `Kubeflow 训练平台 <TrainingService/KubeflowMode.rst>`__ 的依赖从 kubectl CLI 迁移到 `Kubernetes API <https://kubernetes.io/docs/concepts/overview/kubernetes-api/>`__ 客户端。
* Kubeflow 训练服务支持 `Pytorch-operator <https://github.com/kubeflow/pytorch-operator>`__。
* 改进将本地代码文件上传到 OpenPAI HDFS 的性能。
* 修复 OpenPAI 在 WEB 界面的 Bug：当 OpenPAI 认证过期后，Web 界面无法更新 Trial 作业的状态。

改进 NNICTL
^^^^^^^^^^^^^^^^^^^


* 在 nnictl 和 WEB 界面中显示 NNI 的版本信息。 可使用 **nnictl -v** 来显示安装的 NNI 版本。

改进 WEB 界面
^^^^^^^^^^^^^^^^^^


* 在 Experiment 运行中可修改并发数量
* 增加指向 NNI Github 的反馈链接，可直接创建问题
* 可根据指标，定制选择（最大或最小）的前 10 个 Trial。
* 为 dispatcher 和 nnimanager 提供下载日志的功能
* 为指标数值图提供自动缩放的数轴
* 改进 Annotation，支持在搜索空间中显示实际的选项

新样例
^^^^^^^^^^^^


* `FashionMnist <https://github.com/microsoft/nni/tree/v1.9/examples/trials/network_morphism>`__ 使用 network morphism Tuner
* 改进 PyTorch 中的 `分布式 MNIST 示例 <https://github.com/microsoft/nni/tree/v1.9/examples/trials/mnist-distributed-pytorch>`__

发布 0.4 - 12/6/2018
-----------------------

主要功能
^^^^^^^^^^^^^^


* `Kubeflow 训练平台 <TrainingService/KubeflowMode.rst>`__

  * 支持 tf-operator
  * Kubeflow 上的 `分布式 Trial 示例 <https://github.com/microsoft/nni/tree/v1.9/examples/trials/mnist-distributed/dist_mnist.py>`__ 

* `Grid search tuner <Tuner/GridsearchTuner.rst>`__
* `Hyperband tuner <Tuner/HyperbandAdvisor.rst>`__
* 支持在 MAC 上运行 NNI 实验
* Web 界面

  * 支持 hyperband 调参器
  * 移除 tensorboard 按钮
  * 显示实验的错误消息
  * 显示搜索空间和尝试配置的行号
  * 支持通过指定的尝试 id 来搜索
  * 显示尝试的 hdfsLogPath
  * 下载实验参数

其它
^^^^^^


* 异步调度
* 更新 Docker 文件，增加 pytorch 库
* 重构 'nnictl stop' 过程，发送 SIGTERM 给 NNI 管理器进程，而不是调用停止 Restful API.
* OpenPAI 训练服务修复缺陷

  * 在 NNI 管理器中为 OpenPAI 集群配置文件支持 IP 配置(nniManagerIp)，来修复用户计算机没有 eth0 设备的问题。
  * codeDir 中的文件数量上限改为1000，避免用户无意中填写了 root 目录。
  * 移除 OpenPAI 作业的 stdout 日志中无用的 ‘metrics is empty’。 在新指标被记录时，仅输出有用的消息，来减少用户检查 OpenPAI Trial 输出时的困惑。
  * 在尝试 keeper 的开始增加时间戳。

发布 0.3.0 - 11/2/2018
-------------------------

NNICTL 的新功能和更新
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  支持同时运行多个实验。

  在 v0.3 以前，NNI 仅支持一次运行一个实验。 此版本开始，用户可以同时运行多个 Experiment。 每个实验都需要一个唯一的端口，第一个实验会像以前版本一样使用默认端口。 需要为其它实验指定唯一端口：

  .. code-block:: bash

     nnictl create --port 8081 --config <config file path>

* 
  支持更新最大尝试的数量。
  使用 ``nnictl update --help`` 了解详情。 或参考 `NNICTL Spec <Tutorial/Nnictl.rst>`__ 查看完整帮助。

API 的新功能和更新
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* 
  :raw-html:`<span style="color:red">**不兼容的变化**</span>`\ : nn.get_parameters() 改为 nni.get_next_parameter。 所有以前版本的样例将无法在 v0.3 上运行，需要重新克隆 NNI 代码库获取新样例。 如果在自己的代码中使用了 NNI，也需要相应的更新。

* 
  新 API **nni.get_sequence_id()**。
  每个尝试任务都会被分配一个唯一的序列数字，可通过 nni.get_sequence_id() API 来获取。

  .. code-block:: bash

     git clone -b v0.3 https://github.com/microsoft/nni.git

* 
  **nni.report_final_result(result)** API 对结果参数支持更多的数据类型。

  可用类型：


  * int
  * float
  * 包含有 'default' 键值的 dict，'default' 的值必须为 int 或 float。 dict 可以包含任何其它键值对。

新的内置调参器
^^^^^^^^^^^^^^^^^


**Batch Tuner（批处理调参器）** 会执行所有曹参组合，可被用来批量提交尝试任务。

新样例
^^^^^^^^^^^^


* 
  公共的 NNI Docker 映像：

  .. code-block:: bash

     docker pull msranni/nni:latest

* 
  新的 Trial 示例： `NNI Sklearn Example <https://github.com/microsoft/nni/tree/v0.3/examples/trials/sklearn>`__

* 全新比赛示例 `Kaggle Competition TGS Salt Example <https://github.com/microsoft/nni/tree/v0.2/examples/trials/kaggle-tgs-salt>`__

其它
^^^^^^


* 界面重构，参考 `WebUI 文档 <Tutorial/WebUI.rst>`__，了解如何使用新界面。
* 持续集成：NNI 已切换到 Azure pipelines。

发布 0.2.0 - 9/29/2018
-------------------------

主要功能
^^^^^^^^^^^^^^


* 支持 `OpenPAI <https://github.com/microsoft/pai>`__ (又称 pai) 训练平台 (参考 `这里 <TrainingService/PaiMode.rst>`__ 来了解如何在 OpenPAI 下提交 NNI 任务)

  * 支持 pai 模式的训练服务。 NNI 尝试可发送至 OpenPAI 集群上运行
  * NNI 尝试输出 (包括日志和模型文件) 会被复制到 OpenPAI 的 HDFS 中。

* 支持 `SMAC <https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf>`__ tuner (参考 `这里 <Tuner/SmacTuner.rst>`__ 来了解如何使用 SMAC tuner)

  * `SMAC <https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf>`__ 基于 Sequential Model-Based Optimization (SMBO). 它会利用使用过的突出的模型（高斯随机过程模型），并将随机森林引入到SMBO中，来处理分类参数。 NNI 的 SMAC 通过包装 `SMAC3 <https://github.com/automl/SMAC3>`__ 来支持。

* 支持将 NNI 安装在 `conda <https://conda.io/docs/index.html>`__ 和 Python 虚拟环境中。
* 其它

  * 更新 ga squad 样例与相关文档
  * 用户体验改善及缺陷修复

发布 0.1.0 - 9/10/2018 (首个版本)
-------------------------------------------

首次发布 Neural Network Intelligence (NNI)。

主要功能
^^^^^^^^^^^^^^


* 安装和部署

  * 支持 pip 和源代码安装
  * 支持本机（包括多 GPU 卡）训练和远程多机训练模式

* 调参器，评估器和尝试

  * 支持的自动机器学习算法包括： hyperopt_tpe, hyperopt_annealing, hyperopt_random, 和 evolution_tuner。
  * 支持评估器（提前终止）算法包括：medianstop。
  * 提供 Python API 来自定义调参器和评估器
  * 提供 Python API 来包装尝试代码，以便能在 NNI 中运行

* 实验

  * 提供命令行工具 'nnictl' 来管理实验
  * 提供网页界面来查看并管理实验

* 持续集成

  * 使用 Ubuntu 的 `travis-ci <https://github.com/travis-ci>`__ 来支持持续集成

* 其它

  * 支持简单的 GPU 任务调度
