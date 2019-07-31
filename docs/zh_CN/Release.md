# 更改日志

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
* [生成 NAS 编程接口](AdvancedFeature/GeneralNasInterfaces.md) 
  * 实现了 NAS 的编程接口，可通过 NNI Annotation 很容易的表达神经网络架构搜索空间
  * 提供新命令 `nnictl trial codegen` 来调试 NAS 代码生成部分
  * 提供 NAS 编程接口教程，NAS 在 MNIST 上的示例，用于 NAS 的可定制的随机 Tuner
* 支持在恢复 Experiment 时，同时恢复 Tuner 和 Advisor 的状态
* 在恢复 Experiment 时，Tuner 和 Advisor 会导入已完成的 Trial 的数据。
* Web 界面 
  * 改进拷贝 Trial 参数的设计
  * 在 hyper-parameter 图中支持 'randint' 类型
  * 使用 ComponentUpdate 来避免不必要的刷新

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

## 发布 0.5.2 - 3/4/2019

### 改进

* 提升 Curve fitting Assessor 的性能。

### 文档

* 发布中文文档网站：https://nni.readthedocs.io/zh/latest/
* 调试和维护：https://nni.readthedocs.io/zh/latest/Tutorial/HowToDebug.html
* Tuner、Assessor 参考：https://nni.readthedocs.io/zh/latest/sdk_reference.html#tuner

### Bug 修复和其它更新

* 修复了在某些极端条件下，不能正确存储任务的取消状态。
* 修复在使用 SMAC Tuner 时，解析搜索空间的错误。
* 修复 CIFAR-10 样例中的 broken pipe 问题。
* 为本地训练和 NNI 管理器添加单元测试。
* 为远程服务器、OpenPAI 和 Kubeflow 训练平台在 Azure 中增加集成测试。
* 在 OpenPAI 客户端中支持 Pylon 路径。

## 发布 0.5.1 - 1/31/2018

### 改进

* 可配置[日志目录](https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig_zh_CN.md)。
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

* 支持新的 [Metis Tuner](Tuner/MetisTuner.md)。 **在线**超参调优的场景下，Metis 算法已经被证明非常有效。
* 支持 [ENAS customized tuner](https://github.com/countif/enas_nni)。由 GitHub 社区用户所贡献。它是神经网络的搜索算法，能够通过强化学习来学习神经网络架构，比 NAS 的性能更好。
* 支持 [Curve fitting （曲线拟合）Assessor](Assessor/CurvefittingAssessor.md)，通过曲线拟合的策略来实现提前终止 Trial。
* 进一步支持 [Weight Sharing（权重共享）](AdvancedFeature/AdvancedNas.md)：为 NAS Tuner 通过 NFS 来提供权重共享。

#### 改进训练平台

* [FrameworkController 训练平台](TrainingService/FrameworkControllerMode.md)：支持使用在 Kubernetes 上使用 FrameworkController 运行。 
  * FrameworkController 是 Kubernetes 上非常通用的控制器（Controller），能用来运行基于各种机器学习框架的分布式作业，如 TensorFlow，Pytorch， MXNet 等。
  * NNI 为作业定义了统一而简单的规范。
  * 如何使用 FrameworkController 的 MNIST 样例。

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

Initial release of Neural Network Intelligence (NNI).

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