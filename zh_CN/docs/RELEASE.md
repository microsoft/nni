# ChangeLog

## Release 0.5.0 - 01/14/2019

### Major Features

#### New tuner and assessor supports

* 支持 [Metis tuner](./HowToChooseTuner.md#MetisTuner) 作为 NNI 的 Tuner。 **在线**超参调优的场景下，Metis 算法已经被证明非常有效。
* 支持 [ENAS customized tuner](https://github.com/countif/enas_nni)。由 GitHub 社区用户所贡献。它是神经网络的搜索算法，能够通过强化学习来学习神经网络架构，比 NAS 的性能更好。
* 支持 [Curve fitting （曲线拟合）Assessor](./HowToChooseTuner.md#Curvefitting)，通过曲线拟合的策略来实现提前终止 Trial。
* 进一步支持 [Weight Sharing（权重共享）](./AdvancedNAS.md)：为 NAS Tuner 通过 NFS 来提供权重共享。

#### Training Service Enhancement

* [FrameworkController 训练服务](./FrameworkControllerMode.md): 支持使用在 Kubernetes 上使用 FrameworkController。 
  * FrameworkController 是 Kubernetes 上非常通用的控制器（Controller），能用来运行基于各种机器学习框架的分布式作业，如 TensorFlow，Pytorch， MXNet 等。
  * NNI 为作业定义了统一而简单的规范。
  * 如何使用 FrameworkController 的 MNIST 样例。

#### User Experience improvements

* A better trial logging support for NNI experiments in PAI, Kubeflow and FrameworkController mode: 
  * An improved logging architecture to send stdout/stderr of trials to NNI manager via Http post. NNI manager will store trial's stdout/stderr messages in local log file.
  * Show the link for trial log file on WebUI.
* 支持将最终结果显示为键值对。

## Release 0.4.1 - 12/14/2018

### Major Features

#### New tuner supports

* 支持新 Tuner [network morphism](./HowToChooseTuner.md#NetworkMorphism)

#### Training Service improvements

* 将 [Kubeflow 训练服务](https://github.com/Microsoft/nni/blob/master/docs/KubeflowMode.md)的依赖从 kubectl CLI 迁移到 [Kubernetes API](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) 客户端。
* Kubeflow 训练服务支持 [Pytorch-operator](https://github.com/kubeflow/pytorch-operator)。
* 改进将本地代码文件上传到 OpenPAI HDFS 的性能。
* 修复 OpenPAI 在 WEB 界面的 Bug：当 OpenPAI 认证过期后，Web 界面无法更新 Trial 作业的状态。

#### NNICTL improvements

* 在 nnictl 和 WEB 界面中显示 NNI 的版本信息。 可使用 **nnictl -v** 来显示安装的 NNI 版本。

#### WebUI improvements

* 在 Experiment 运行中可修改并发数量
* 增加指向 NNI Github 的反馈链接，可直接创建问题
* 可根据指标，定制选择（最大或最小）的前 10 个 Trial。
* 为 dispatcher 和 nnimanager 提供下载日志的功能
* 为指标数值图提供自动缩放的数轴
* 改进 Annotation，支持在搜索空间中显示实际的选项

### New examples

* [FashionMnist](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism)，使用 network morphism Tuner
* 使用 PyTorch 的[分布式 MNIST 样例](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)

## Release 0.4 - 12/6/2018

### Major Features

* [Kubeflow 训练服务](./KubeflowMode.md) 
  * 支持 tf-operator
  * Kubeflow 上的[分布式 Trial 样例](../examples/trials/mnist-distributed/dist_mnist.py)
* [网格搜索 Tuner](../src/sdk/pynni/nni/README.md#Grid) 
* [Hyperband Tuner](../src/sdk/pynni/nni/README.md#Hyperband)
* 支持在 MAC 上运行 NNI Experiment
* WebUI 
  * 支持 hyperband Tuner
  * 移除 tensorboard 按钮
  * 显示 Experiment 的错误消息
  * 显示搜索空间和 Trial 配置的行号
  * 支持通过指定的 Trial id 来搜索
  * 显示 Trial 的 hdfsLogPath
  * 下载 Experiment 参数

### Others

* 异步调度
* 更新 Docker 文件，增加 pytorch 库 
* 重构 'nnictl stop' 过程，发送 SIGTERM 给 NNI 管理器进程，而不是调用停止 Restful API. 
* 修复 OpenPAI 训练服务的 Bug 
  * 在 NNI 管理器中为 PAI 集群配置文件支持 IP 配置(nniManagerIp)，来修复用户计算机没有 eth0 设备的问题。 
  * codeDir 中的文件数量上限改为1000，避免用户无意中填写了 root 目录。
  * 移除 PAI 作业的 stdout 日志中无用的 ‘metrics is empty’。 在新指标被记录时，仅输出有用的消息，来减少用户检查 PAI Trial 输出时的困惑。
  * 在 Trial keeper 的开始增加时间戳。

## Release 0.3.0 - 11/2/2018

### NNICTL new features and updates

* 支持同时运行多个 Experiment。
  
  在 v0.3 以前，NNI 仅支持一次运行一个 Experiment。 此版本开始，用户可以同时运行多个 Experiment。 每个 Experiment 都需要一个唯一的端口，第一个 Experiment 会像以前版本一样使用默认端口。 需要为其它 Experiment 指定唯一端口：
  
  ```bash
  nnictl create --port 8081 --config <config file path>
  ```

* Support updating max trial number. use `nnictl update --help` to learn more. Or refer to [NNICTL Spec](https://github.com/Microsoft/nni/blob/master/docs/NNICTLDOC.md) for the fully usage of NNICTL.

### API new features and updates

* <span style="color:red"><strong>不兼容的改动</strong></span>：nn.get_parameters() 改为 nni.get_next_parameter。 所有以前版本的样例将无法在 v0.3 上运行，需要重新克隆 NNI 代码库获取新样例。 如果在自己的代码中使用了 NNI，也需要相应的更新。

* 新 API **nni.get_sequence_id()**。 每个 Trial 任务都会被分配一个唯一的序列数字，可通过 nni.get_sequence_id() API 来获取。
  
  ```bash
  git clone -b v0.3 https://github.com/Microsoft/nni.git
  ```

* **nni.report_final_result(result)** API supports more data types for result parameter.
  
  It can be of following types:
  
  * int
  * float
  * 包含有 'default' 键值的 dict，'default' 的值必须为 int 或 float。 dict 可以包含任何其它键值对。

### New tuner support

* **Batch Tuner（批处理调参器）** 会执行所有超参组合，可被用来批量提交 Trial 任务。

### New examples

* A NNI Docker image for public usage:
  
  ```bash
  docker pull msranni/nni:latest
  ```

* New trial example: [NNI Sklearn Example](https://github.com/Microsoft/nni/tree/master/examples/trials/sklearn)

* 新的竞赛样例：[Kaggle Competition TGS Salt](https://github.com/Microsoft/nni/tree/master/examples/trials/kaggle-tgs-salt)

### Others

* 界面重构，参考[网页文档](WebUI.md)，了解如何使用新界面。
* 持续集成：NNI 已切换到 Azure pipelines。
* [0.3.0 的已知问题](https://github.com/Microsoft/nni/labels/nni030knownissues)。

## Release 0.2.0 - 9/29/2018

### Major Features

* Support [OpenPAI](https://github.com/Microsoft/pai) (aka pai) Training Service (See [here](./PAIMode.md) for instructions about how to submit NNI job in pai mode) 
  * Support training services on pai mode. NNI trials will be scheduled to run on OpenPAI cluster
  * NNI trial's output (including logs and model file) will be copied to OpenPAI HDFS for further debugging and checking
* Support [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) tuner (See [here](HowToChooseTuner.md) for instructions about how to use SMAC tuner) 
  * [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO to handle categorical parameters. The SMAC supported by NNI is a wrapper on [SMAC3](https://github.com/automl/SMAC3)
* Support NNI installation on [conda](https://conda.io/docs/index.html) and python virtual environment
* Others 
  * Update ga squad example and related documentation
  * WebUI UX small enhancement and bug fix

### Known Issues

[0.2.0 的已知问题](https://github.com/Microsoft/nni/labels/nni020knownissues)。

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

### Known Issues

[0.1.0 的已知问题](https://github.com/Microsoft/nni/labels/nni010knownissues)。