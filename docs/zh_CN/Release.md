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
  * 支持新的 BOHB Advisor，这是一个健壮而有效的超参调优算法，囊括了贝叶斯优化和 Hyperband 的优点
* [支持通过 nnictl 来导入导出 Experiment 数据](Tutorial/Nnictl.md#experiment) 
  * 在 Experiment 执行完后，可生成分析结果报告
  * 支持将先前的调优数据导入到 Tuner 和 Advisor 中
* [可为 NNI Trial 任务指定 GPU](Tutorial/ExperimentConfig.md#localConfig) 
  * 通过 gpuIndices 配置来为 Trial 任务指定GPU。如果 Experiment 配置文件中有 gpuIndices，则只有指定的 GPU 会被用于 NNI 的 Trial 任务。
* 改进 Web 界面 
  * 在 Web 界面上使用十进制格式的指标
  * 添加多阶段训练相关的提示
  * 可将超参复制为 Python dict 格式
  * 可将提前终止的 Trial 数据传入 Tuner。
* 为 nnictl 提供更友好的错误消息 
  * 为 YAML 文件格式错误提供更有意义的错误信息

### Bug 修复

* 运行 nnictl stop 的异步 Dispatcher 模式时，无法杀掉所有的 Python 线程
* nnictl --version 不能在 make dev-install 下使用
* OpenPAI 平台下所有的 Trial 任务状态都是 'WAITING'

## 发布 0.6 - 4/2/2019

### 主要功能

* [版本检查](TrainingService/PaiMode.md) 
  * 检查 nniManager 和 trialKeeper 的版本是否一致
* [提前终止的任务也可返回最终指标](https://github.com/microsoft/nni/issues/776) 
  * 如果 includeIntermediateResults 为 true，最后一个 Assessor 的中间结果会被发送给 Tuner 作为最终结果。 The default value of includeIntermediateResults is false.
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
* 使用 PyTorch 的[分布式 MNIST 样例](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)

## 发布 0.4 - 12/6/2018

### 主要功能

* [Kubeflow 训练平台](TrainingService/KubeflowMode.md) 
  * 支持 tf-operator
  * 使用 Kubeflow 的[分布式 Trial 样例](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-distributed/dist_mnist.py)
* [遍历搜索 Tuner](Tuner/GridsearchTuner.md)
* [Hyperband Tuner](Tuner/HyperbandAdvisor.md)
* 支持在 MAC 上运行 NNI Experiment
* Web 界面 
  * 支持 hyperband Tuner
  * 移除 tensorboard 按钮
  * 显示 Experiment 的错误消息
  * 显示搜索空间和 Trial 配置的行号
  * 支持通过指定的 Trial id 来搜索
  * 显示 Trial 的 hdfsLogPath
  * 下载 Experiment 参数

### 其它

* 异步调度
* 更新 Docker 文件，增加 pytorch 库
* 重构 'nnictl stop' 过程，发送 SIGTERM 给 NNI 管理器进程，而不是调用停止 Restful API.
* 修复 OpenPAI 训练服务的 Bug 
  * 在 NNI 管理器中为 OpenPAI 集群配置文件支持 IP 配置(nniManagerIp)，来修复用户计算机没有 eth0 设备的问题。
  * codeDir 中的文件数量上限改为1000，避免用户无意中填写了 root 目录。
  * 移除 OpenPAI 作业的 stdout 日志中无用的 ‘metrics is empty’。 在新指标被记录时，仅输出有用的消息，来减少用户检查 OpenPAI Trial 输出时的困惑。
  * 在 Trial keeper 的开始增加时间戳。

## 发布 0.3.0 - 11/2/2018

### NNICTL 的新功能和更新

* 支持同时运行多个 Experiment。
    
    在 v0.3 以前，NNI 仅支持一次运行一个 Experiment。 此版本开始，用户可以同时运行多个 Experiment。 每个 Experiment 都需要一个唯一的端口，第一个 Experiment 会像以前版本一样使用默认端口。 需要为其它 Experiment 指定唯一端口：
    
    ```bash
    nnictl create --port 8081 --config <config file path>
    ```

* 支持更新最大 Trial 的数量。 use `nnictl update --help` to learn more. Or refer to [NNICTL Spec](Tutorial/Nnictl.md) for the fully usage of NNICTL.

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