# 更改日志

## 发布 1.2 - 12/02/2019

### 主要功能

* [特征工程](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/Overview.md) 
  - 新增特征工程接口
  - 特征选择算法: [Gradient feature selector](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/GradientFeatureSelector.md) & [GBDT selector](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/GBDTSelector.md)
  - [特征工程示例](https://github.com/microsoft/nni/tree/v1.2/examples/feature_engineering)
- 神经网络结构搜索在 NNI 上的应用 
  - [新的 NAS 接口](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/NasInterface.md)
  - NAS 算法: [ENAS](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/Overview.md#enas), [DARTS](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/Overview.md#darts), [P-DARTS](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/Overview.md#p-darts) (PyTorch)
  - 经典模式下的 NAS（每次 Trial 独立运行）
- 模型压缩 
  - [新增模型剪枝算法](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/Compressor/Overview.md): lottery ticket 修剪, L1Filter Pruner, Slim Pruner, FPGM Pruner
  - [新增模型量化算法](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/Compressor/Overview.md): QAT Quantizer, DoReFa Quantizer
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
- 使超参数图的默认指标 yAxis 更加精确。 -PR#1736
- 修复 GPU 脚本权限问题。 -Issue #1665

## 发布 1.1 - 10/23/2019

### 主要功能

* 新 Tuner: [PPO Tuner](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tuner/PPOTuner.md)
* [查看已停止的 Experiment](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tutorial/Nnictl.md#view)
* Tuner 可使用专门的 GPU 资源（参考[教程](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tutorial/ExperimentConfig.md)中的 `gpuIndices` 了解详情）
* 改进 WEB 界面 
  - Trial 详情页面可列出每个 Trial 的超参，以及开始结束时间（需要通过 "add column" 添加）
  - 优化大型 Experiment 的显示性能
- 更多示例 
  - [EfficientNet PyTorch 示例](https://github.com/ultmaster/EfficientNet-PyTorch)
  - [Cifar10 NAS 示例](https://github.com/microsoft/nni/blob/v1.1/examples/trials/nas_cifar10/README.md)
- [模型压缩工具包 - Alpha 发布](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Compressor/Overview.md)：我们很高兴的宣布 NNI 的模型压缩工具包发布了。它还处于试验阶段，会根据使用反馈来改进。 诚挚邀请您使用、反馈，或更多贡献

### 修复的 Bug

* 当搜索空间结束后，多阶段任务会死锁 (issue #1204)
* 没有日志时，`nnictl` 会失败 (issue #1548)

## 发布1.0 - 9/2/2019

### 主要功能

* Tuners 和 Assessors
  
    - 支持自动特征生成和选择 -Issue#877 -PR #1387 + 提供自动特征接口 + 基于 Beam 搜索的 Tuner + [添加 Pakdd 示例](./examples/trials/auto-feature-engineering/README_zh_CN.md)
    - 添加并行算法提高 TPE 在高并发下的性能。 -PR #1052
    - 为 hyperband 支持多阶段 -PR #1257
- 训练平台
  
    -      支持私有 Docker Registry -PR #755
        
    
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
  
    - Update the docs structure -Issue #1231
    - [Multi phase document improvement](./docs/en_US/AdvancedFeature/MultiPhase.md) -Issue #1233 -PR #1242 + Add configuration example
    - [WebUI description improvement](./docs/en_US/Tutorial/WebUI.md) -PR #1419

### 修复的 Bug

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

* 通用 NAS 编程接口 
    * Add `enas-mode` and `oneshot-mode` for NAS interface: [PR #1201](https://github.com/microsoft/nni/pull/1201#issue-291094510)
* [Gaussian Process Tuner with Matern kernel](Tuner/GPTuner.md)

* Multiphase experiment supports
  
    * Added new training service support for multiphase experiment: PAI mode supports multiphase experiment since v0.9.
    * Added multiphase capability for the following builtin tuners: 
      * TPE, Random Search, Anneal, Naïve Evolution, SMAC, Network Morphism, Metis Tuner.
  
    For details, please refer to [Write a tuner that leverages multi-phase](AdvancedFeature/MultiPhase.md)

* Web Portal
  
    * Enable trial comparation in Web Portal. For details, refer to [View trials status](Tutorial/WebUI.md)
    * Allow users to adjust rendering interval of Web Portal. For details, refer to [View Summary Page](Tutorial/WebUI.md)
    * show intermediate results more friendly. For details, refer to [View trials status](Tutorial/WebUI.md)
* [Commandline Interface](Tutorial/Nnictl.md) 
    * `nnictl experiment delete`: delete one or all experiments, it includes log, result, environment information and cache. It uses to delete useless experiment result, or save disk space.
    * `nnictl platform clean`: It uses to clean up disk on a target platform. The provided YAML file includes the information of target platform, and it follows the same schema as the NNI configuration file.

### Bug fix and other changes

* Tuner Installation Improvements: add [sklearn](https://scikit-learn.org/stable/) to nni dependencies.
* (Bug Fix) Failed to connect to PAI http code - [Issue #1076](https://github.com/microsoft/nni/issues/1076)
* (Bug Fix) Validate file name for PAI platform - [Issue #1164](https://github.com/microsoft/nni/issues/1164)
* (Bug 修复) 更新 Metis Tunerz 中的 GMM
* (Bug 修复) Web 界面负数的刷新间隔时间 - [Issue #1182](https://github.com/microsoft/nni/issues/1182), [Issue #1185](https://github.com/microsoft/nni/issues/1185)
* (Bug 修复) 当只有一个超参时，Web 界面的超参无法正确显示 - [Issue #1192](https://github.com/microsoft/nni/issues/1192)

## 发布 0.8 - 6/4/2019

### 主要功能

* 在 Windows 上支持 NNI 的 OpenPAI 和远程模式 
  * NNI 可在 Windows 上使用 OpenPAI 模式
  * NNI running on windows for OpenPAI mode
* Advanced features for using GPU 
  * Run multiple trial jobs on the same GPU for local and remote mode
  * Run trial jobs on the GPU running non-NNI jobs
* Kubeflow v1beta2 operator 
  * Support Kubeflow TFJob/PyTorchJob v1beta2
* [General NAS programming interface](AdvancedFeature/GeneralNasInterfaces.md) 
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

### Major Features

* [Support NNI on Windows](Tutorial/NniOnWindows.md) 
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

#### 支持新的 Tuner 和 Assessor

* Support [Metis tuner](Tuner/MetisTuner.md) as a new NNI tuner. Metis algorithm has been proofed to be well performed for **online** hyper-parameter tuning.
* Support [ENAS customized tuner](https://github.com/countif/enas_nni), a tuner contributed by github community user, is an algorithm for neural network search, it could learn neural network architecture via reinforcement learning and serve a better performance than NAS.
* Support [Curve fitting assessor](Assessor/CurvefittingAssessor.md) for early stop policy using learning curve extrapolation.
* Advanced Support of [Weight Sharing](AdvancedFeature/AdvancedNas.md): Enable weight sharing for NAS tuners, currently through NFS.

#### 改进训练平台

* [FrameworkController Training service](TrainingService/FrameworkControllerMode.md): Support run experiments using frameworkcontroller on kubernetes 
  * FrameworkController is a Controller on kubernetes that is general enough to run (distributed) jobs with various machine learning frameworks, such as tensorflow, pytorch, MXNet.
  * NNI provides unified and simple specification for job definition.
  * MNIST example for how to use FrameworkController.

#### 改进用户体验

* A better trial logging support for NNI experiments in OpenPAI, Kubeflow and FrameworkController mode: 
  * An improved logging architecture to send stdout/stderr of trials to NNI manager via Http post. NNI manager will store trial's stdout/stderr messages in local log file.
  * Show the link for trial log file on WebUI.
* Support to show final result's all key-value pairs.

## Release 0.4.1 - 12/14/2018

### Major Features

#### 支持新的 Tuner

* Support [network morphism](Tuner/NetworkmorphismTuner.md) as a new tuner

#### 改进训练平台

* Migrate [Kubeflow training service](TrainingService/KubeflowMode.md)'s dependency from kubectl CLI to [Kubernetes API](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) client
* [Pytorch-operator](https://github.com/kubeflow/pytorch-operator) support for Kubeflow training service
* Improvement on local code files uploading to OpenPAI HDFS
* Fixed OpenPAI integration WebUI bug: WebUI doesn't show latest trial job status, which is caused by OpenPAI token expiration

#### 改进 NNICTL

* Show version information both in nnictl and WebUI. You can run **nnictl -v** to show your current installed NNI version

#### 改进 WEB 界面

* Enable modify concurrency number during experiment
* Add feedback link to NNI github 'create issue' page
* Enable customize top 10 trials regarding to metric numbers (largest or smallest)
* Enable download logs for dispatcher & nnimanager
* Enable automatic scaling of axes for metric number
* Update annotation to support displaying real choice in searchspace

### 新示例

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