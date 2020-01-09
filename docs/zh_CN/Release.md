# 更改日志

## 发布 1.3 - 12/30/2019

### 主要功能

#### 支持神经网络架构搜索算法

* [单路径一次性](https://github.com/microsoft/nni/tree/v1.3/examples/nas/spos/)算法和示例

#### 模型压缩算法支持

* [知识蒸馏](https://github.com/microsoft/nni/blob/v1.3/docs/zh_CN/TrialExample/KDExample.md)算法和使用示例
* Pruners 
    * [L2Filter Pruner](https://github.com/microsoft/nni/blob/v1.3/docs/zh_CN/Compressor/Pruner.md#3-l2filter-pruner)
    * [ActivationAPoZRankFilterPruner](https://github.com/microsoft/nni/blob/v1.3/docs/zh_CN/Compressor/Pruner.md#1-activationapozrankfilterpruner)
    * [ActivationMeanRankFilterPruner](https://github.com/microsoft/nni/blob/v1.3/docs/zh_CN/Compressor/Pruner.md#2-activationmeanrankfilterpruner)
* [BNN Quantizer](https://github.com/microsoft/nni/blob/v1.3/docs/zh_CN/Compressor/Quantizer.md#bnn-quantizer)

#### 训练平台

* OpenPAI 的 NFS 支持
    
    从 OpenPAI v0.11开始，HDFS 不再用作默认存储，可将 NFS、AzureBlob 或其他存储用作默认存储。 在本次版本中，NNI 扩展了对 OpenPAI 最近改动的支持，可与 OpenPAI v0.11 及后续版本的默认存储集成。

* Kubeflow 更新适配
    
    适配 Kubeflow 0.7 对 tf-operator 的新支持。

### 工程（代码和生成自动化）

* 启用 [ESLint](https://eslint.org/) 静态代码分析。

### 小改动和 Bug 修复

* 正确识别内置 Tuner 和定制 Tuner
* Dispatcher 基类的日志
* 修复有时 Tuner、Assessor 的失败会终止 Experiment 的 Bug。
* 修复本机作为远程计算机的[问题](https://github.com/microsoft/nni/issues/1852)
* SMAC Tuner 中 Trial 配置的去重 [ticket](https://github.com/microsoft/nni/issues/1364)

## 发布 1.2 - 12/02/2019

### 主要功能

* [特征工程](https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/FeatureEngineering/Overview.md) 
  - 新增特征工程接口
  - 特征选择算法: [Gradient feature selector](https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/FeatureEngineering/GradientFeatureSelector.md) & [GBDT selector](https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/FeatureEngineering/GBDTSelector.md)
  - [特征工程示例](https://github.com/microsoft/nni/tree/v1.2/examples/feature_engineering)
- 神经网络结构搜索在 NNI 上的应用 
  - [新的 NAS 接口](https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/NAS/NasInterface.md)
  - NAS 算法: [ENAS](https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/NAS/Overview.md#enas), [DARTS](https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/NAS/Overview.md#darts), [P-DARTS](https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/NAS/Overview.md#p-darts) (PyTorch)
  - 经典模式下的 NAS（每次 Trial 独立运行）
- 模型压缩 
  - [新增模型剪枝算法](https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/Compressor/Overview.md): lottery ticket 修剪, L1Filter Pruner, Slim Pruner, FPGM Pruner
  - [新增模型量化算法](https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/Compressor/Overview.md): QAT Quantizer, DoReFa Quantizer
  - 支持导出压缩后模型的 API。
- 训练平台 
  - 支持 OpenPAI 令牌身份验证
- 示例： 
  - [使用 NNI 自动调优 rocksdb 配置的示例](https://github.com/microsoft/nni/tree/v1.2/examples/trials/systems/rocksdb-fillrandom)。
  - [新的支持 TensorFlow 2.0 的 Trial 示例](https://github.com/microsoft/nni/tree/v1.2/examples/trials/mnist-tfv2)。
- 改进 
  - 远程训练平台中不需要 GPU 的 Trial 任务改为使用随机调度，不再使用轮询调度。
  - 添加 pylint 规则来检查拉取请求，新的拉取请求需要符合 [pylint 规则](https://github.com/microsoft/nni/blob/v1.2/pylintrc)。
- Web 门户和用户体验 
  - 支持用户添加自定义 Trial。
  - 除了超参外，用户可放大缩小详细图形。
- 文档 
  - 改进了 NNI API 文档，增加了更多的 docstring。

### Bug 修复

- 修复当失败的 Trial 没有指标时，表格的排序问题。 -Issue #1773
- 页面切换时，保留选择的（最大、最小）状态。 -PR#1710
- 使超参数图的默认指标 yAxis 更加精确。 -PR#1736
- 修复 GPU 脚本权限问题。 -Issue #1665

## 发布 1.1 - 10/23/2019

### 主要功能

* 新 Tuner: [PPO Tuner](https://github.com/microsoft/nni/blob/v1.1/docs/zh_CN/Tuner/PPOTuner.md)
* [查看已停止的 Experiment](https://github.com/microsoft/nni/blob/v1.1/docs/zh_CN/Tutorial/Nnictl.md#view)
* Tuner 可使用专门的 GPU 资源（参考[教程](https://github.com/microsoft/nni/blob/v1.1/docs/zh_CN/Tutorial/ExperimentConfig.md)中的 `gpuIndices` 了解详情）
* 改进 WEB 界面 
  - Trial 详情页面可列出每个 Trial 的超参，以及开始结束时间（需要通过 "add column" 添加）
  - 优化大型 Experiment 的显示性能
- 更多示例 
  - [EfficientNet PyTorch 示例](https://github.com/ultmaster/EfficientNet-PyTorch)
  - [Cifar10 NAS 示例](https://github.com/microsoft/nni/blob/v1.1/examples/trials/nas_cifar10/README.md)
- [模型压缩工具包 - Alpha 发布](https://github.com/microsoft/nni/blob/v1.1/docs/zh_CN/Compressor/Overview.md)：我们很高兴的宣布 NNI 的模型压缩工具包发布了。它还处于试验阶段，会根据使用反馈来改进。 诚挚邀请您使用、反馈，或更多贡献

### 修复的 Bug

* 当搜索空间结束后，多阶段任务会死锁 (issue #1204)
* 没有日志时，`nnictl` 会失败 (issue #1548)

## 发布1.0 - 9/2/2019

### 主要功能

* Tuners 和 Assessors
    
    - 支持自动特征生成和选择 -Issue#877 -PR #1387 
        + 提供自动特征接口
        + 基于 Beam 搜索的 Tuner
        + [增加 Pakdd 示例](https://github.com/microsoft/nni/tree/master/examples/trials/auto-feature-engineering)
    + 添加并行算法提高 TPE 在高并发下的性能。 -PR #1052
    + 为 hyperband 支持多阶段 -PR #1257
+ 训练平台
    
    - 支持私有 Docker Registry -PR #755
        
        * 改进
        * 增加 RestFUL API 的 Python 包装，支持通过代码获取指标的值 PR #1318
        * 新的 Python API : get_experiment_id(), get_trial_id() -PR #1353 -Issue #1331 & -Issue#1368
        * 优化 NAS 搜索空间 -PR #1393 
         + 使用 _type 统一 NAS 搜索空间 -- "mutable_type"e
         + 更新随机搜索 Tuner
        + 将 gpuNum 设为可选 -Issue #1365
        + 删除 OpenPAI 模式下的 outputDir 和 dataDir 配置 -Issue #1342
        + 在 Kubeflow 模式下创建 Trial 时，codeDir 不再被拷贝到 logDir -Issue #1224
+ Web 门户和用户体验
    
    - 在 Web 界面的搜索过程中显示最好指标的曲线 -Issue #1218
    - 在多阶段 Experiment 中，显示参数列表的当前值 -Issue1210 -PR #1348
    - 在 AddColumn 中增加 "Intermediate count" 选项。 -Issue #1210
    - 在 Web 界面中支持搜索参数的值 -Issue #1208
    - 在默认指标图中，启用指标轴的自动缩放 -Issue #1360
    - 在命令行中为 nnictl 命令增加详细文档的连接 -Issue #1260
    - 用户体验改进：显示 Error 日志 -Issue #1173
- 文档
    
    - 更新文档结构 -Issue #1231
    - [多阶段文档的改进](AdvancedFeature/MultiPhase.md) -Issue #1233 -PR #1242 
        + 添加配置示例
    + [Web 界面描述改进](Tutorial/WebUI.md) -PR #1419

### Bug 修复

* (Bug 修复)修复 0.9 版本中的链接 -Issue #1236
* (Bug 修复)自动完成脚本
* (Bug 修复) 修复管道中仅检查脚本中最后一个命令退出代码的问题。 -PR #1417
* (Bug 修复) Tuner 的 quniform -Issue #1377
* (Bug fix) 'quniform' 在 GridSearch 和其它 Tuner 之间的含义不同。 -Issue #1335
* (Bug 修复)"nnictl experiment list" 将 "RUNNING" 状态的 Experiment 显示为了 "INITIALIZED" -PR #1388
* (Bug 修复) 在 NNI dev 安装模式下无法安装 SMAC。 -Issue #1376
* (Bug 修复) 无法点击中间结果的过滤按钮 -Issue #1263
* (Bug 修复) API "/api/v1/nni/trial-jobs/xxx" 在多阶段 Experiment 无法显示 Trial 的所有参数 -Issue #1258
* (Bug 修复) 成功的 Trial 没有最终结果，但 Web 界面显示成了 ×××(FINAL) -Issue #1207
* (Bug 修复) nnictl stop -Issue #1298
* (Bug 修复) 修复安全警告
* (Bug 修复) 超参页面损坏 -Issue #1332
* (Bug 修复) 运行 flake8 测试来查找 Python 语法错误和未定义的名称 -PR #1217

## 发布 0.9 - 7/1/2019

### 主要功能

* 生成 NAS 编程接口 
    * 为 NAS 接口添加 `enas-mode` 和 `oneshot-mode`：[PR #1201](https://github.com/microsoft/nni/pull/1201#issue-291094510)
* [有 Matern 核的高斯 Tuner](Tuner/GPTuner.md)

* 支持多阶段 Experiment
    
    * 为多阶段 Experiment 增加新的训练平台：pai 模式从 v0.9 开始支持多阶段 Experiment。
    *     为以下内置 Tuner 增加多阶段的功能： 
            
        
        * TPE, Random Search, Anneal, Naïve Evolution, SMAC, Network Morphism, Metis Tuner。
        
        有关详细信息，参考[实现多阶段的 Tuner](AdvancedFeature/MultiPhase.md)。

* Web 界面
    
    * 在 Web 界面中可比较 Trial。 有关详细信息，参考[查看 Trial 状态](Tutorial/WebUI.md)
    * 允许用户调节 Web 界面的刷新间隔。 有关详细信息，参考[查看概要页面](Tutorial/WebUI.md)
    * 更友好的显示中间结果。 有关详细信息，参考[查看 Trial 状态](Tutorial/WebUI.md)
* [命令行接口](Tutorial/Nnictl.md) 
    * `nnictl experiment delete`：删除一个或多个 Experiment，包括其日志，结果，环境信息核缓存。 用于删除无用的 Experiment 结果，或节省磁盘空间。
    * `nnictl platform clean`：用于清理目标平台的磁盘空间。 所提供的 YAML 文件包括了目标平台的信息，与 NNI 配置文件的格式相同。

### Bug 修复和其它更新

* 改进 Tuner 安装过程：增加 [sklearn](https://scikit-learn.org/stable/) 依赖。
* (Bug 修复) 连接 OpenPAI 失败的 HTTP 代码 - [Issue #1076](https://github.com/microsoft/nni/issues/1076)
* (Bug 修复) 为 OpenPAI 平台验证文件名 - [Issue #1164](https://github.com/microsoft/nni/issues/1164)
* (Bug 修复) 更新 Metis Tunerz 中的 GMM
* (Bug 修复) Web 界面负数的刷新间隔时间 - [Issue #1182](https://github.com/microsoft/nni/issues/1182), [Issue #1185](https://github.com/microsoft/nni/issues/1185)
* (Bug 修复) 当只有一个超参时，Web 界面的超参无法正确显示 - [Issue #1192](https://github.com/microsoft/nni/issues/1192)

## 发布 0.8 - 6/4/2019

### 主要功能

* 在 Windows 上支持 NNI 的 OpenPAI 和远程模式 
  * NNI 可在 Windows 上使用 OpenPAI 模式
  * NNI 可在 Windows 上使用 OpenPAI 模式
* GPU 的高级功能 
  * 在本机或远程模式上，可在同一个 GPU 上运行多个 Trial。
  * 在已经运行非 NNI 任务的 GPU 上也能运行 Trial
* 支持 Kubeflow v1beta2 操作符 
  * 支持 Kubeflow TFJob/PyTorchJob v1beta2
* [通用 NAS 编程接口](https://github.com/microsoft/nni/blob/v0.8/docs/zh_CN/GeneralNasInterfaces.md) 
  * 实现了 NAS 的编程接口，可通过 NNI Annotation 很容易的表达神经网络架构搜索空间
  * 提供新命令 `nnictl trial codegen` 来调试 NAS 代码生成部分
  * 提供 NAS 编程接口教程，NAS 在 MNIST 上的示例，用于 NAS 的可定制的随机 Tuner
* 支持在恢复 Experiment 时，同时恢复 Tuner 和 Advisor 的状态
* 在恢复 Experiment 时，Tuner 和 Advisor 会导入已完成的 Trial 的数据。
* Web 界面 
  * 改进拷贝 Trial 参数的设计
  * 在 hyper-parameter 图中支持 'randint' 类型
  * 使用 ComponentUpdate 来避免不必要的刷新

### Bug 修复和其它更新

* 修复 `nnictl update` 不一致的命令行风格
* SMAC Tuner 支持导入数据
* 支持 Experiment 状态从 ERROR 回到 RUNNING
* 修复表格的 Bug
* 优化嵌套搜索空间
* 优化 'randint' 类型，并支持下限
* [比较不同超参搜索调优算法](CommunitySharings/HpoComparision.md)
* [NAS 算法的对比](CommunitySharings/NasComparision.md)
* [Recommenders 上的实践](CommunitySharings/RecommendersSvd.md)

## 发布 0.7 - 4/29/2018

### 主要功能

* [支持在 Windows 上使用 NNI](Tutorial/NniOnWindows.md) 
  * NNI 可在 Windows 上使用本机模式
* [支持新的 Advisor: BOHB](Tuner/BohbAdvisor.md) 
  * Support a new advisor BOHB, which is a robust and efficient hyperparameter tuning algorithm, combines the advantages of Bayesian optimization and Hyperband
* [Support import and export experiment data through nnictl](Tutorial/Nnictl.md#experiment) 
  * Generate analysis results report after the experiment execution
  * Support import data to tuner and advisor for tuning
* [Designated gpu devices for NNI trial jobs](Tutorial/ExperimentConfig.md#localConfig) 
  * Specify GPU devices for NNI trial jobs by gpuIndices configuration, if gpuIndices is set in experiment configuration file, only the specified GPU devices are used for NNI trial jobs.
* Web Portal enhancement 
  * Decimal format of metrics other than default on the Web UI
  * Hints in WebUI about Multi-phase
  * Enable copy/paste for hyperparameters as python dict
  * Enable early stopped trials data for tuners.
* NNICTL provide better error message 
  * nnictl provide more meaningful error message for YAML file format error

### Bug fix

* Unable to kill all python threads after nnictl stop in async dispatcher mode
* nnictl --version does not work with make dev-install
* All trail jobs status stays on 'waiting' for long time on OpenPAI platform

## Release 0.6 - 4/2/2019

### Major Features

* [Version checking](TrainingService/PaiMode.md) 
  * check whether the version is consistent between nniManager and trialKeeper
* [Report final metrics for early stop job](https://github.com/microsoft/nni/issues/776) 
  * If includeIntermediateResults is true, the last intermediate result of the trial that is early stopped by assessor is sent to tuner as final result. The default value of includeIntermediateResults is false.
* [Separate Tuner/Assessor](https://github.com/microsoft/nni/issues/841) 
  * Adds two pipes to separate message receiving channels for tuner and assessor.
* Make log collection feature configurable
* Add intermediate result graph for all trials

### Bug fix

* [Add shmMB config key for OpenPAI](https://github.com/microsoft/nni/issues/842)
* Fix the bug that doesn't show any result if metrics is dict
* Fix the number calculation issue for float types in hyperband
* Fix a bug in the search space conversion in SMAC tuner
* Fix the WebUI issue when parsing experiment.json with illegal format
* Fix cold start issue in Metis Tuner

## Release 0.5.2 - 3/4/2019

### 改进

* Curve fitting assessor performance improvement.

### 文档

* Chinese version document: https://nni.readthedocs.io/zh/latest/
* Debuggability/serviceability document: https://nni.readthedocs.io/en/latest/Tutorial/HowToDebug.html
* Tuner assessor reference: https://nni.readthedocs.io/en/latest/sdk_reference.html

### Bug 修复和其它更新

* Fix a race condition bug that does not store trial job cancel status correctly.
* Fix search space parsing error when using SMAC tuner.
* Fix cifar10 example broken pipe issue.
* Add unit test cases for nnimanager and local training service.
* Add integration test azure pipelines for remote machine, OpenPAI and kubeflow training services.
* Support Pylon in OpenPAI webhdfs client.

## Release 0.5.1 - 1/31/2018

### Improvements

* Making [log directory](https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig.md) configurable
* 支持[不同级别的日志](https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig_zh_CN.md)，使其更易于调试。

### 文档

* 重新组织文档，新的主页位置：https://nni.readthedocs.io/zh/latest/

### Bug 修复和其它更新

* 修复了 Python 虚拟环境中安装的 Bug，并重构了安装逻辑。
* 修复了在最新的 OpenPAI 下存取 HDFS 失败的问题。
* 修复了有时刷新 stdout 会造成 Experiment 崩溃的问题。

## 发布 0.5.0 - 01/14/2019

### 主要功能

#### 支持新的 Tuner 和 Assessor

* 支持新的 [Metis Tuner](Tuner/MetisTuner.md)。 对于**在线**超参调优的场景，Metis 算法已经被证明非常有效。
* 支持 [ENAS customized tuner](https://github.com/countif/enas_nni)。由 GitHub 社区用户所贡献。它是神经网络的搜索算法，能够通过强化学习来学习神经网络架构，比 NAS 的性能更好。
* 支持 [Curve fitting （曲线拟合）Assessor](Assessor/CurvefittingAssessor.md)，通过曲线拟合的策略来实现提前终止 Trial。
* [权重共享的](https://github.com/microsoft/nni/blob/v0.5/docs/AdvancedNAS.md)高级支持：为 NAS Tuner 提供权重共享，当前支持 NFS。

#### 改进训练平台

* [FrameworkController 训练平台](TrainingService/FrameworkControllerMode.md)：支持使用在 Kubernetes 上使用 FrameworkController 运行。 
  * FrameworkController 是 Kubernetes 上非常通用的控制器（Controller），能用来运行基于各种机器学习框架的分布式作业，如 TensorFlow，Pytorch， MXNet 等。
  * NNI 为作业定义了统一而简单的规范。
  * 如何使用 FrameworkController 的 MNIST 示例。

#### 改进用户体验

* 为 OpenPAI, Kubeflow 和 FrameworkController 模式提供更好的日志支持。 
  * 改进后的日志架构能将尝试的 stdout/stderr 通过 HTTP POST 方式发送给 NNI 管理器。 NNI 管理器将 Trial 的 stdout/stderr 消息存储在本地日志文件中。
  * 在 WEB 界面上显示 Trial 日志的链接。
* 支持将最终结果显示为键值对。

## 发布 0.4.1 - 12/14/2018

### 主要功能

#### 支持新的 Tuner

* 支持新的 [network morphism](Tuner/NetworkmorphismTuner.md) Tuner。

#### 改进训练平台

* 将 [Kubeflow 训练平台](TrainingService/KubeflowMode.md)的依赖从 kubectl CLI 迁移到 [Kubernetes API](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) 客户端。
* Kubeflow 训练平台支持 [Pytorch-operator](https://github.com/kubeflow/pytorch-operator)。
* 改进将本地代码文件上传到 OpenPAI HDFS 的性能。
* 修复 OpenPAI 在 WEB 界面的 Bug：当 OpenPAI 认证过期后，Web 界面无法更新 Trial 作业的状态。

#### 改进 NNICTL

* 在 nnictl 和 WEB 界面中显示 NNI 的版本信息。 可使用 **nnictl -v** 来显示安装的 NNI 版本。

#### 改进 WEB 界面

* 在 Experiment 运行中可修改并发数量
* 增加指向 NNI Github 的反馈链接，可直接创建问题
* 可根据指标，定制选择（最大或最小）的前 10 个 Trial。
* 为 dispatcher 和 nnimanager 提供下载日志的功能
* 为指标数值图提供自动缩放的数轴
* 改进 Annotation，支持在搜索空间中显示实际的选项

### 新示例

* [FashionMnist](https://github.com/microsoft/nni/tree/master/examples/trials/network_morphism)，使用 network morphism Tuner
* 使用 PyTorch 的[分布式 MNIST 示例](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)

## 发布 0.4 - 12/6/2018

### 主要功能

* [Kubeflow 训练平台](TrainingService/KubeflowMode.md) 
  * 支持 tf-operator
  * [Distributed trial example](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-distributed/dist_mnist.py) on Kubeflow
* [Grid search tuner](Tuner/GridsearchTuner.md)
* [Hyperband tuner](Tuner/HyperbandAdvisor.md)
* Support launch NNI experiment on MAC
* WebUI 
  * UI support for hyperband tuner
  * Remove tensorboard button
  * Show experiment error message
  * Show line numbers in search space and trial profile
  * Support search a specific trial by trial number
  * Show trial's hdfsLogPath
  * Download experiment parameters

### Others

* Asynchronous dispatcher
* Docker file update, add pytorch library
* Refactor 'nnictl stop' process, send SIGTERM to nni manager process, rather than calling stop Rest API.
* OpenPAI training service bug fix 
  * Support NNI Manager IP configuration(nniManagerIp) in OpenPAI cluster config file, to fix the issue that user’s machine has no eth0 device
  * File number in codeDir is capped to 1000 now, to avoid user mistakenly fill root dir for codeDir
  * Don’t print useless ‘metrics is empty’ log in OpenPAI job’s stdout. Only print useful message once new metrics are recorded, to reduce confusion when user checks OpenPAI trial’s output for debugging purpose
  * Add timestamp at the beginning of each log entry in trial keeper.

## Release 0.3.0 - 11/2/2018

### NNICTL new features and updates

* Support running multiple experiments simultaneously.
    
    Before v0.3, NNI only supports running single experiment once a time. After this release, users are able to run multiple experiments simultaneously. Each experiment will require a unique port, the 1st experiment will be set to the default port as previous versions. You can specify a unique port for the rest experiments as below:
    
    ```bash
    nnictl create --port 8081 --config <config file path>
    ```

* Support updating max trial number. use `nnictl update --help` to learn more. Or refer to [NNICTL Spec](Tutorial/Nnictl.md) for the fully usage of NNICTL.

### API new features and updates

* <span style="color:red"><strong>breaking change</strong></span>: nn.get_parameters() is refactored to nni.get_next_parameter. All examples of prior releases can not run on v0.3, please clone nni repo to get new examples. If you had applied NNI to your own codes, please update the API accordingly.

* New API **nni.get_sequence_id()**. Each trial job is allocated a unique sequence number, which can be retrieved by nni.get_sequence_id() API.
    
    ```bash
    git clone -b v0.3 https://github.com/microsoft/nni.git
    ```

* **nni.report_final_result(result)** API supports more data types for result parameter.
    
    It can be of following types:
    
  * int
  * float
  * A python dict containing 'default' key, the value of 'default' key should be of type int or float. The dict can contain any other key value pairs.

### New tuner support

* **Batch Tuner** which iterates all parameter combination, can be used to submit batch trial jobs.

### New examples

* 公开的 NNI Docker 映像：
    
    ```bash
    docker pull msranni/nni:latest
    ```

* 新的 Trial 示例：[NNI Sklearn 示例](https://github.com/microsoft/nni/tree/master/examples/trials/sklearn)

* 新的竞赛示例：[Kaggle Competition TGS Salt](https://github.com/microsoft/nni/tree/master/examples/trials/kaggle-tgs-salt)

### 其它

* 界面重构，参考[网页文档](Tutorial/WebUI.md)，了解如何使用新界面。
* 持续集成：NNI 已切换到 Azure pipelines。

## 发布 0.2.0 - 9/29/2018

### 主要功能

* 支持 [OpenPAI](https://github.com/microsoft/pai) (又称 pai) 训练平台 (参考[这里](TrainingService/PaiMode.md)来了解如何在 OpenPAI 下提交 NNI 任务) 
  * 支持 pai 模式的训练平台。 NNI Trial 可发送至 OpenPAI 集群上运行
  * NNI Trial 输出 (包括日志和模型文件) 会被复制到 OpenPAI 的 HDFS 中。
* 支持 [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) Tuner (参考[这里](Tuner/SmacTuner.md)，了解如何使用 SMAC Tuner) 
  * [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) 基于 Sequential Model-Based Optimization (SMBO). 它会利用使用过的结果好的模型（高斯随机过程模型），并将随机森林引入到 SMBO 中，来处理分类参数。 NNI 的 SMAC 通过包装 [SMAC3](https://github.com/automl/SMAC3) 来支持。
* 支持将 NNI 安装在 [conda](https://conda.io/docs/index.html) 和 Python 虚拟环境中。
* 其它 
  * 更新 ga squad 示例与相关文档
  * 用户体验改善及 Bug 修复

## 发布 0.1.0 - 9/10/2018 (首个版本)

首次发布 Neural Network Intelligence (NNI)。

### 主要功能

* 安装和部署 
  * 支持 pip 和源代码安装
  * 支持本机（包括多 GPU 卡）训练和远程多机训练模式
* Tuner ，Assessor 和 Trial 
  * 支持的自动机器学习算法包括： hyperopt_tpe, hyperopt_annealing, hyperopt_random, 和 evolution_tuner。
  * 支持 Assessor（提前终止）算法包括：medianstop。
  * 提供 Python API 来自定义 Tuner 和 Assessor
  * 提供 Python API 来包装 Trial 代码，以便能在 NNI 中运行
* Experiment 
  * 提供命令行工具 'nnictl' 来管理 Experiment
  * 提供网页界面来查看并管理 Experiment
* 持续集成 
  * 使用 Ubuntu 的 [travis-ci](https://github.com/travis-ci) 来支持持续集成
* 其它 
  * 支持简单的 GPU 任务调度