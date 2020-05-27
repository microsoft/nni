# 更改日志

## 发布 1.6 - 5/26/2020

### 主要功能

#### 新功能和改进

* 将 IPC 限制提高至 100W
* 修改非本机训练平台中，将上传代码到存储的逻辑
* SDK 版本支持 `__version__`
* 支持 Windows 下开发模式安装

#### Web 界面

* 显示 Trial 的错误消息
* 完善主页布局
* 重构概述页面的最佳 Trial 模块
* 从 Web 界面中去掉多阶段支持
* 在概述页面为 Trial 并发添加工具提示。
* 在超参图中显示最好的 Trial

#### 超参优化更新

* 改进 PBT 的错误处理，并支持恢复 Experiment

#### NAS 更新

* NAS 支持 TensorFlow 2.0 (预览版) [TF2.0 NAS 示例](https://github.com/microsoft/nni/tree/master/examples/nas/naive-tf)
* LayerChoice 使用 OrderedDict
* 优化导出格式
* 应用固定架构后，将 LayerChoice 替换成选择的模块

#### 模型压缩改进

* 模型压缩支持 PyTorch 1.4

#### 训练平台改进

* 改进 OpenPAI YAML 的合并逻辑
* 支持将 Windows 作为[远程模式](https://github.com/microsoft/nni/blob/master/docs/en_US/TrainingService/RemoteMachineMode.md#windows)中的计算节点

### 修复的 Bug

* 修复开发模式安装
* 当检查点没有 state_dict 时，SPOS 示例会崩溃
* 修复失败 Trial 造成的表格排序问题
* 支持多 Python 环境（如 conda，pyenv 等）

## 发布 1.5 - 4/13/2020

### 新功能和文档

#### 超参优化

* 新 Tuner：[Population Based Training (PBT)](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Tuner/PBTTuner.md)
* Trial 现在可以返回无穷大和 NaN 结果

#### 神经网络架构搜索

* 新 NAS 算法：[TextNAS](https://github.com/microsoft/nni/blob/master/docs/zh_CN/NAS/TextNAS.md)
* ENAS 和 DARTS 现在可通过网页[可视化](https://github.com/microsoft/nni/blob/master/docs/zh_CN/NAS/Visualization.md)。

#### 模型压缩

* 新 Pruner：[GradientRankFilterPruner](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Compressor/Pruner.md#gradientrankfilterpruner)
* 默认情况下，Compressor 会验证配置
* 重构：可将优化器作为 Pruner 的输入参数，从而更容易支持 DataParallel 和其它迭代剪枝方法。 这是迭代剪枝算法用法上的重大改动。
* 重构了模型压缩示例
* 添加了[实现模型压缩算法](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Compressor/Framework.md)的文档

#### 训练平台

* Kubeflow 现已支持 pytorchjob crd v1 (感谢贡献者 @jiapinai)
* 实验性的支持 [DLTS](https://github.com/microsoft/nni/blob/master/docs/zh_CN/TrainingService/DLTSMode.md)

#### 文档的整体改进

* 语法、拼写以及措辞上的修改 (感谢贡献者 @AHartNtkn)

### 修复的 Bug

* ENAS 不能使用多个 LSTM 层 (感谢贡献者 @marsggbo)
* NNI 管理器的计时器无法取消订阅 (感谢贡献者 @guilhermehn)
* NNI 管理器可能会耗尽内存 (感谢贡献者 @Sundrops)
* 批处理 Tuner 不支持自定义 Trial （#2075）
* Experiment 启动失败后，无法终止 (#2080)
* 非数字的指标会破坏网页界面 (#2278)
* lottery ticket Pruner 中的 Bug
* 其它小问题

## 发布 1.4 - 2/19/2020

### 主要功能

#### 神经网络架构搜索

* 支持 [C-DARTS](https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/NAS/CDARTS.md) 算法，并增加对应[示例](https://github.com/microsoft/nni/tree/v1.4/examples/nas/cdarts)。
* 初步支持 [ProxylessNAS](https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/NAS/Proxylessnas.md) 以及对应[示例](https://github.com/microsoft/nni/tree/v1.4/examples/nas/proxylessnas)
* 为 NAS 框架增加单元测试

#### 模型压缩

* 为压缩模型增加 DataParallel，并提供相应的 [示例](https://github.com/microsoft/nni/blob/v1.4/examples/model_compress/multi_gpu.py)
* 支持压缩模型的[加速](https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/Compressor/ModelSpeedup.md)（试用版）

#### 训练平台

* 通过允许指定 OpenPAI 配置文件路径，来支持完整的 OpenPAI 配置
* 为新的 OpenPAI 模式（又称，paiK8S）增加示例配置 YAML 文件
* 支持删除远程模式下使用 sshkey 的 Experiment （感谢外部贡献者 @tyusr）

#### Web 界面

* Web 界面重构：采用 fabric 框架

#### 其它

* 支持[在前台运行 NNI Experiment](https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/Tutorial/Nnictl.md#manage-an-experiment)，即，`nnictl create/resume/view` 的 `--foreground` 参数
* 支持取消 UNKNOWN 状态的 Trial。
* 支持最大 50MB 的搜索空间文件 （感谢外部贡献者 @Sundrops）

### 文档

* 改进 NNI readthedocs 的[目录索引结构](https://nni.readthedocs.io/zh/latest/)
* 改进 [NAS 文档](https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/NAS/NasGuide.md)
* 改进[新的 OpenPAI 模式的文档](https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/TrainingService/PaiMode.md)
* 为 [NAS](https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/NAS/QuickStart.md) 和[模型压缩](https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/Compressor/QuickStart.md)增加入门指南
* 改进支持 [EfficientNet](https://github.com/microsoft/nni/blob/v1.4/docs/zh_CN/TrialExample/EfficientNet.md) 的文档

### 修复的 Bug

* 修复在指标数据和 JSON 格式中对 NaN 的支持
* 修复搜索空间 `randint` 类型的 out-of-range Bug
* 修复模型压缩中导出 ONNX 模型时的错误张量设备的 Bug
* 修复新 OpenPAI 模式（又称，paiK8S）下，错误处理 nnimanagerIP 的 Bug

## 发布 1.3 - 12/30/2019

### 主要功能

#### 支持神经网络架构搜索算法

* [单路径一次性](https://github.com/microsoft/nni/tree/v1.3/examples/nas/spos/)算法和示例

#### 模型压缩算法支持

* [知识蒸馏](https://github.com/microsoft/nni/blob/v1.3/docs/zh_CN/TrialExample/KDExample.md)算法和使用示例
* Pruners 
    * [L2Filter Pruner](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Pruner.md#3-l2filter-pruner)
    * [ActivationAPoZRankFilterPruner](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Pruner.md#1-activationapozrankfilterpruner)
    * [ActivationMeanRankFilterPruner](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Pruner.md#2-activationmeanrankfilterpruner)
* [BNN Quantizer](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Quantizer.md#bnn-quantizer)

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

* [特征工程](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/Overview.md) 
  - 新增特征工程接口
  - 特征选择算法: [Gradient feature selector](https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/FeatureEngineering/GradientFeatureSelector.md) & [GBDT selector](https://github.com/microsoft/nni/blob/v1.2/docs/zh_CN/FeatureEngineering/GBDTSelector.md)
  - [特征工程示例](https://github.com/microsoft/nni/tree/v1.2/examples/feature_engineering)
- 神经网络结构搜索在 NNI 上的应用 
  - [新的 NAS 接口](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/NasInterface.md)
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

### 修复的 Bug

- 修复当失败的 Trial 没有指标时，表格的排序问题。 -Issue #1773
- 页面切换时，保留选择的（最大、最小）状态。 -PR#1710
- Make hyper-parameters graph's default metric yAxis more accurate. -PR#1736
- Fix GPU script permission issue. -Issue #1665

## Release 1.1 - 10/23/2019

### 主要功能

* New tuner: [PPO Tuner](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tuner/PPOTuner.md)
* [View stopped experiments](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tutorial/Nnictl.md#view)
* Tuners can now use dedicated GPU resource (see `gpuIndices` in [tutorial](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tutorial/ExperimentConfig.md) for details)
* Web UI improvements 
  - Trials detail page can now list hyperparameters of each trial, as well as their start and end time (via "add column")
  - Viewing huge experiment is now less laggy
- More examples 
  - [EfficientNet PyTorch example](https://github.com/ultmaster/EfficientNet-PyTorch)
  - [Cifar10 NAS example](https://github.com/microsoft/nni/blob/v1.1/examples/trials/nas_cifar10/README.md)
- [Model compression toolkit - Alpha release](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Compressor/Overview.md): We are glad to announce the alpha release for model compression toolkit on top of NNI, it's still in the experiment phase which might evolve based on usage feedback. We'd like to invite you to use, feedback and even contribute

### Fixed Bugs

* Multiphase job hangs when search space exhuasted (issue #1204)
* `nnictl` fails when log not available (issue #1548)

## Release 1.0 - 9/2/2019

### 主要功能

* Tuners and Assessors
    
    - Support Auto-Feature generator & selection -Issue#877 -PR #1387 
        + Provide auto feature interface
        + Tuner based on beam search
        + [Add Pakdd example](https://github.com/microsoft/nni/tree/master/examples/trials/auto-feature-engineering)
    + Add a parallel algorithm to improve the performance of TPE with large concurrency. -PR #1052
    + Support multiphase for hyperband -PR #1257
+ Training Service
    
    - Support private docker registry -PR #755
        
        * Engineering Improvements
        * Python wrapper for rest api, support retrieve the values of the metrics in a programmatic way PR #1318
        * New python API : get_experiment_id(), get_trial_id() -PR #1353 -Issue #1331 & -Issue#1368
        * Optimized NAS Searchspace -PR #1393 
         + Unify NAS search space with _type -- "mutable_type"e
         + Update random search tuner
        + Set gpuNum as optional -Issue #1365
        + Remove outputDir and dataDir configuration in PAI mode -Issue #1342
        + When creating a trial in Kubeflow mode, codeDir will no longer be copied to logDir -Issue #1224
+ Web Portal & User Experience
    
    - Show the best metric curve during search progress in WebUI -Issue #1218
    - Show the current number of parameters list in multiphase experiment -Issue1210 -PR #1348
    - Add "Intermediate count" option in AddColumn. -Issue #1210
    - Support search parameters value in WebUI -Issue #1208
    - Enable automatic scaling of axes for metric value in default metric graph -Issue #1360
    - Add a detailed documentation link to the nnictl command in the command prompt -Issue #1260
    - UX improvement for showing Error log -Issue #1173
- Documentation
    
    - 更新文档结构 -Issue #1231
    - (deprecated) Multi phase document improvement -Issue #1233 -PR #1242 
        + Add configuration example
    + [WebUI description improvement](Tutorial/WebUI.md) -PR #1419

### Bug fix

* (Bug fix)Fix the broken links in 0.9 release -Issue #1236
* (Bug fix)Script for auto-complete
* (Bug fix)Fix pipeline issue that it only check exit code of last command in a script. -PR #1417
* (Bug fix)quniform fors tuners -Issue #1377
* (Bug fix)'quniform' has different meaning beween GridSearch and other tuner. -Issue #1335
* (Bug fix)"nnictl experiment list" give the status of a "RUNNING" experiment as "INITIALIZED" -PR #1388
* (Bug fix)SMAC cannot be installed if nni is installed in dev mode -Issue #1376
* (Bug fix)The filter button of the intermediate result cannot be clicked -Issue #1263
* (Bug fix)API "/api/v1/nni/trial-jobs/xxx" doesn't show a trial's all parameters in multiphase experiment -Issue #1258
* (Bug fix)Succeeded trial doesn't have final result but webui show ×××(FINAL) -Issue #1207
* (Bug fix)IT for nnictl stop -Issue #1298
* (Bug fix)fix security warning
* (Bug fix)Hyper-parameter page broken -Issue #1332
* (Bug fix)Run flake8 tests to find Python syntax errors and undefined names -PR #1217

## Release 0.9 - 7/1/2019

### 主要功能

* General NAS programming interface 
    * Add `enas-mode` and `oneshot-mode` for NAS interface: [PR #1201](https://github.com/microsoft/nni/pull/1201#issue-291094510)
* [Gaussian Process Tuner with Matern kernel](Tuner/GPTuner.md)

* (deprecated) Multiphase experiment supports
    
    * Added new training service support for multiphase experiment: PAI mode supports multiphase experiment since v0.9.
    * Added multiphase capability for the following builtin tuners: 
        * TPE, Random Search, Anneal, Naïve Evolution, SMAC, Network Morphism, Metis Tuner.
* Web Portal
    
    * Enable trial comparation in Web Portal. For details, refer to [View trials status](Tutorial/WebUI.md)
    * Allow users to adjust rendering interval of Web Portal. For details, refer to [View Summary Page](Tutorial/WebUI.md)
    * show intermediate results more friendly. For details, refer to [View trials status](Tutorial/WebUI.md)
* [Commandline Interface](Tutorial/Nnictl.md) 
    * `nnictl experiment delete`: delete one or all experiments, it includes log, result, environment information and cache. It uses to delete useless experiment result, or save disk space.
    * `nnictl platform clean`: It uses to clean up disk on a target platform. The provided YAML file includes the information of target platform, and it follows the same schema as the NNI configuration file.

### Bug 修复和其它更新

* Tuner Installation Improvements: add [sklearn](https://scikit-learn.org/stable/) to nni dependencies.
* (Bug Fix) Failed to connect to PAI http code - [Issue #1076](https://github.com/microsoft/nni/issues/1076)
* (Bug Fix) Validate file name for PAI platform - [Issue #1164](https://github.com/microsoft/nni/issues/1164)
* (Bug Fix) Update GMM evaluation in Metis Tuner
* (Bug Fix) Negative time number rendering in Web Portal - [Issue #1182](https://github.com/microsoft/nni/issues/1182), [Issue #1185](https://github.com/microsoft/nni/issues/1185)
* (Bug Fix) Hyper-parameter not shown correctly in WebUI when there is only one hyper parameter - [Issue #1192](https://github.com/microsoft/nni/issues/1192)

## Release 0.8 - 6/4/2019

### 主要功能

* Support NNI on Windows for OpenPAI/Remote mode 
  * NNI running on windows for remote mode
  * NNI running on windows for OpenPAI mode
* Advanced features for using GPU 
  * Run multiple trial jobs on the same GPU for local and remote mode
  * Run trial jobs on the GPU running non-NNI jobs
* Kubeflow v1beta2 operator 
  * Support Kubeflow TFJob/PyTorchJob v1beta2
* [General NAS programming interface](https://github.com/microsoft/nni/blob/v0.8/docs/en_US/GeneralNasInterfaces.md) 
  * Provide NAS programming interface for users to easily express their neural architecture search space through NNI annotation
  * Provide a new command `nnictl trial codegen` for debugging the NAS code
  * Tutorial of NAS programming interface, example of NAS on MNIST, customized random tuner for NAS
* Support resume tuner/advisor's state for experiment resume
* For experiment resume, tuner/advisor will be resumed by replaying finished trial data
* Web Portal 
  * Improve the design of copying trial's parameters
  * Support 'randint' type in hyper-parameter graph
  * Use should ComponentUpdate to avoid unnecessary render

### Bug fix and other changes

* Bug fix that `nnictl update` has inconsistent command styles
* Support import data for SMAC tuner
* Bug fix that experiment state transition from ERROR back to RUNNING
* Fix bug of table entries
* Nested search space refinement
* Refine 'randint' type and support lower bound
* [Comparison of different hyper-parameter tuning algorithm](CommunitySharings/HpoComparision.md)
* [Comparison of NAS algorithm](CommunitySharings/NasComparision.md)
* [NNI practice on Recommenders](CommunitySharings/RecommendersSvd.md)

## Release 0.7 - 4/29/2018

### 主要功能

* [Support NNI on Windows](Tutorial/InstallationWin.md) 
  * NNI running on windows for local mode
* [New advisor: BOHB](Tuner/BohbAdvisor.md) 
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

### 修复的 Bug

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

### Improvements

* Curve fitting assessor performance improvement.

### Documentation

* Chinese version document: https://nni.readthedocs.io/zh/latest/
* Debuggability/serviceability document: https://nni.readthedocs.io/en/latest/Tutorial/HowToDebug.html
* Tuner assessor reference: https://nni.readthedocs.io/en/latest/sdk_reference.html

### Bug Fixes and Other Changes

* Fix a race condition bug that does not store trial job cancel status correctly.
* Fix search space parsing error when using SMAC tuner.
* Fix cifar10 example broken pipe issue.
* Add unit test cases for nnimanager and local training service.
* Add integration test azure pipelines for remote machine, OpenPAI and kubeflow training services.
* Support Pylon in OpenPAI webhdfs client.

## Release 0.5.1 - 1/31/2018

### Improvements

* Making [log directory](https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig.md) configurable
* Support [different levels of logs](https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig.md), making it easier for debugging

### Documentation

* Reorganized documentation & New Homepage Released: https://nni.readthedocs.io/en/latest/

### Bug Fixes and Other Changes

* Fix the bug of installation in python virtualenv, and refactor the installation logic
* Fix the bug of HDFS access failure on OpenPAI mode after OpenPAI is upgraded.
* Fix the bug that sometimes in-place flushed stdout makes experiment crash

## Release 0.5.0 - 01/14/2019

### Major Features

#### New tuner and assessor supports

* Support [Metis tuner](Tuner/MetisTuner.md) as a new NNI tuner. Metis algorithm has been proofed to be well performed for **online** hyper-parameter tuning.
* Support [ENAS customized tuner](https://github.com/countif/enas_nni), a tuner contributed by github community user, is an algorithm for neural network search, it could learn neural network architecture via reinforcement learning and serve a better performance than NAS.
* Support [Curve fitting assessor](Assessor/CurvefittingAssessor.md) for early stop policy using learning curve extrapolation.
* Advanced Support of [Weight Sharing](https://github.com/microsoft/nni/blob/v0.5/docs/AdvancedNAS.md): Enable weight sharing for NAS tuners, currently through NFS.

#### Training Service Enhancement

* [FrameworkController Training service](TrainingService/FrameworkControllerMode.md): Support run experiments using frameworkcontroller on kubernetes 
  * FrameworkController is a Controller on kubernetes that is general enough to run (distributed) jobs with various machine learning frameworks, such as tensorflow, pytorch, MXNet.
  * NNI provides unified and simple specification for job definition.
  * MNIST example for how to use FrameworkController.

#### User Experience improvements

* A better trial logging support for NNI experiments in OpenPAI, Kubeflow and FrameworkController mode: 
  * An improved logging architecture to send stdout/stderr of trials to NNI manager via Http post. NNI manager will store trial's stdout/stderr messages in local log file.
  * Show the link for trial log file on WebUI.
* Support to show final result's all key-value pairs.

## Release 0.4.1 - 12/14/2018

### 主要功能

#### New tuner supports

* Support [network morphism](Tuner/NetworkmorphismTuner.md) as a new tuner

#### Training Service improvements

* Migrate [Kubeflow training service](TrainingService/KubeflowMode.md)'s dependency from kubectl CLI to [Kubernetes API](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) client
* [Pytorch-operator](https://github.com/kubeflow/pytorch-operator) support for Kubeflow training service
* Improvement on local code files uploading to OpenPAI HDFS
* Fixed OpenPAI integration WebUI bug: WebUI doesn't show latest trial job status, which is caused by OpenPAI token expiration

#### NNICTL improvements

* Show version information both in nnictl and WebUI. You can run **nnictl -v** to show your current installed NNI version

#### WebUI improvements

* Enable modify concurrency number during experiment
* Add feedback link to NNI github 'create issue' page
* Enable customize top 10 trials regarding to metric numbers (largest or smallest)
* Enable download logs for dispatcher & nnimanager
* Enable automatic scaling of axes for metric number
* Update annotation to support displaying real choice in searchspace

### New examples

* [FashionMnist](https://github.com/microsoft/nni/tree/master/examples/trials/network_morphism), work together with network morphism tuner
* [Distributed MNIST example](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch) written in PyTorch

## Release 0.4 - 12/6/2018

### Major Features

* [Kubeflow Training service](TrainingService/KubeflowMode.md) 
  * Support tf-operator
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

* A NNI Docker image for public usage:
    
    ```bash
    docker pull msranni/nni:latest
    ```

* New trial example: [NNI Sklearn Example](https://github.com/microsoft/nni/tree/master/examples/trials/sklearn)

* New competition example: [Kaggle Competition TGS Salt Example](https://github.com/microsoft/nni/tree/master/examples/trials/kaggle-tgs-salt)

### Others

* UI refactoring, refer to [WebUI doc](Tutorial/WebUI.md) for how to work with the new UI.
* Continuous Integration: NNI had switched to Azure pipelines

## Release 0.2.0 - 9/29/2018

### Major Features

* Support [OpenPAI](https://github.com/microsoft/pai) Training Platform (See [here](TrainingService/PaiMode.md) for instructions about how to submit NNI job in pai mode) 
  * Support training services on pai mode. NNI trials will be scheduled to run on OpenPAI cluster
  * NNI trial's output (including logs and model file) will be copied to OpenPAI HDFS for further debugging and checking
* Support [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) tuner (See [here](Tuner/SmacTuner.md) for instructions about how to use SMAC tuner) 
  * [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO to handle categorical parameters. The SMAC supported by NNI is a wrapper on [SMAC3](https://github.com/automl/SMAC3)
* Support NNI installation on [conda](https://conda.io/docs/index.html) and python virtual environment
* Others 
  * Update ga squad example and related documentation
  * WebUI UX small enhancement and bug fix

## Release 0.1.0 - 9/10/2018 (initial release)

首次发布 Neural Network Intelligence (NNI)。

### Major Features

* Installation and Deployment 
  * Support pip install and source codes install
  * Support training services on local mode(including Multi-GPU mode) as well as multi-machines mode
* Tuners, Assessors and Trial 
  * Support AutoML algorithms including: hyperopt_tpe, hyperopt_annealing, hyperopt_random, and evolution_tuner
  * Support assessor(early stop) algorithms including: medianstop algorithm
  * Provide Python API for user defined tuners and assessors
  * Provide Python API for user to wrap trial code as NNI deployable codes
* Experiments 
  * Provide a command line toolkit 'nnictl' for experiments management
  * Provide a WebUI for viewing experiments details and managing experiments
* Continuous Integration 
  * Support CI by providing out-of-box integration with [travis-ci](https://github.com/travis-ci) on ubuntu
* Others 
  * Support simple GPU job scheduling