# 更改日志

## 发布 0.5.1 - 1/31/2018

### 改进

* 可配置[日志目录](ExperimentConfig.md)。
* 支持[不同级别的日志](ExperimentConfig.md)，使其更易于调试。 

### 文档

* 重新组织文档，新的主页位置：https://nni.readthedocs.io/en/latest/

### Bug 修复和其它更新

* 修复了 Python 虚拟环境中安装的 Bug，并重构了安装逻辑。
* 修复了在最新的 OpenPAI 下存取 HDFS 失败的问题。 
* 修复了有时刷新 stdout 会造成 Experiment 崩溃的问题。

## 发布 0.5.0 - 01/14/2019

### 主要功能

#### 支持新的 Tuner 和 Assessor

* 支持[Metis tuner](./Builtin_Tuner.md#MetisTuner) 作为 NNI 的 Tuner。 **在线**超参调优的场景下，Metis 算法已经被证明非常有效。
* 支持 [ENAS customized tuner](https://github.com/countif/enas_nni)。由 GitHub 社区用户所贡献。它是神经网络的搜索算法，能够通过强化学习来学习神经网络架构，比 NAS 的性能更好。
* 支持 [Curve fitting （曲线拟合）Assessor](./Builtin_Tuner.md#Curvefitting)，通过曲线拟合的策略来实现提前终止 Trial。
* 进一步支持 [Weight Sharing（权重共享）](./AdvancedNAS.md)：为 NAS Tuner 通过 NFS 来提供权重共享。

#### 改进训练平台

* [FrameworkController 训练服务](./FrameworkControllerMode.md): 支持使用在 Kubernetes 上使用 FrameworkController。 
  * FrameworkController 是 Kubernetes 上非常通用的控制器（Controller），能用来运行基于各种机器学习框架的分布式作业，如 TensorFlow，Pytorch， MXNet 等。
  * NNI 为作业定义了统一而简单的规范。
  * 如何使用 FrameworkController 的 MNIST 样例。

#### 改进用户体验

* 为 OpenPAI, Kubeflow 和 FrameworkController 模式提供更好的日志支持。 
  * 改进后的日志架构能将尝试的 stdout/stderr 通过 HTTP POST 方式发送给 NNI 管理器。 NNI 管理器将 Trial 的 stdout/stderr 消息存储在本地日志文件中。
  * Show the link for trial log file on WebUI.
* Support to show final result's all key-value pairs.

## Release 0.4.1 - 12/14/2018

### Major Features

#### 支持新的 Tuner

* Support [network morphism](./Builtin_Tuner.md#NetworkMorphism) as a new tuner

#### 改进训练平台

* Migrate [Kubeflow training service](https://github.com/Microsoft/nni/blob/master/docs/KubeflowMode.md)'s dependency from kubectl CLI to [Kubernetes API](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) client
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

### New examples

* [FashionMnist](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism), work together with network morphism tuner
* [Distributed MNIST example](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch) written in PyTorch

## Release 0.4 - 12/6/2018

### Major Features

* [Kubeflow Training service](./KubeflowMode.md) 
  * Support tf-operator
  * [Distributed trial example](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed/dist_mnist.py) on Kubeflow
* [Grid search tuner](Builtin_Tuner.md#GridSearch) 
* [Hyperband tuner](Builtin_Tuner.md#Hyperband)
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
  
  Before v0.3, NNI only supports running single experiment once a time. After this realse, users are able to run multiple experiments simultaneously. Each experiment will require a unique port, the 1st experiment will be set to the default port as previous versions. You can specify a unique port for the rest experiments as below:
  
  ```bash
  nnictl create --port 8081 --config <config file path>
  ```

* Support updating max trial number. use `nnictl update --help` to learn more. Or refer to [NNICTL Spec](https://github.com/Microsoft/nni/blob/master/docs/NNICTLDOC.md) for the fully usage of NNICTL.

### API new features and updates

* <span style="color:red"><strong>breaking change</strong></span>: nn.get_parameters() is refactored to nni.get_next_parameter. All examples of prior releases can not run on v0.3, please clone nni repo to get new examples. If you had applied NNI to your own codes, please update the API accordingly.

* New API **nni.get_sequence_id()**. Each trial job is allocated a unique sequence number, which can be retrieved by nni.get_sequence_id() API.
  
  ```bash
  git clone -b v0.3 https://github.com/Microsoft/nni.git
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

* New trial example: [NNI Sklearn Example](https://github.com/Microsoft/nni/tree/master/examples/trials/sklearn)

* New competition example: [Kaggle Competition TGS Salt Example](https://github.com/Microsoft/nni/tree/master/examples/trials/kaggle-tgs-salt)

### Others

* UI refactoring, refer to [WebUI doc](WebUI.md) for how to work with the new UI.
* Continuous Integration: NNI had switched to Azure pipelines
* [Known Issues in release 0.3.0](https://github.com/Microsoft/nni/labels/nni030knownissues).

## Release 0.2.0 - 9/29/2018

### Major Features

* Support [OpenPAI](https://github.com/Microsoft/pai) Training Platform (See [here](./PAIMode.md) for instructions about how to submit NNI job in pai mode) 
  * Support training services on pai mode. NNI trials will be scheduled to run on OpenPAI cluster
  * NNI trial's output (including logs and model file) will be copied to OpenPAI HDFS for further debugging and checking
* Support [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) tuner (See [here](Builtin_Tuner.md) for instructions about how to use SMAC tuner) 
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

### 已知问题

[0.1.0 的已知问题](https://github.com/Microsoft/nni/labels/nni010knownissues)。