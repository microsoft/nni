# 更改日志

# 发布 0.8 - 6/4/2019

## 主要功能

* 在 Windows 上支持 NNI 的 OpenPAI 和远程模式 
    * NNI 可在 Windows 上使用远程模式
    * NNI 可在 Windows 上使用 OpenPAI 模式
* GPU 的高级功能 
    * 在本机或远程模式上，可在同一个 GPU 上运行多个 Trial。
    * 在已经运行非 NNI 任务的 GPU 上也能运行 Trial
* 支持 Kubeflow v1beta2 操作符 
    * 支持 Kubeflow TFJob/PyTorchJob v1beta2
* [通过 NAS 编程接口](./GeneralNasInterfaces.md) 
    * 实现了 NAS 的编程接口，可通过 NNI Annotation 很容易的表达神经网络架构搜索空间
    * 提供新命令 `nnictl trial codegen` 来调试 NAS 代码生成部分
    * 提供 NAS 编程接口教程，NAS 在 MNIST 上的示例，用于 NAS 的可定制的随机 Tuner
* 支持在恢复 Experiment 时，同时恢复 Tuner 和 Advisor 的状态 
    * 在恢复 Experiment 时，Tuner 和 Advisor 会导入已完成的 Trial 的数据。
* Web 界面 
    * 改进拷贝 Trial 参数的设计
    * 在 hyper-parameter 图中支持 'randint' 类型
    * 使用 ComponentUpdate 来避免不必要的刷新

## Bug 修复和其它更新

* [Bug fix that `nnictl update` has inconsistent command styles]
* [Support import data for SMAC tuner]
* [Bug fix that experiment state transition from ERROR back to RUNNING]
* [Fix bug of table entries]
* [Nested search space refinement]
* [Refine 'randint' type and support lower bound]
* [Comparison of different hyper-parameter tuning algorithm](./CommunitySharings/HpoComparision.md)
* [Comparison of NAS algorithm](./CommunitySharings/NasComparision.md)
* [NNI practice on Recommenders](./CommunitySharings/NniPracticeSharing/RecommendersSvd.md)

## 发布 0.7 - 4/29/2018

### 主要功能

* [支持在 Windows 上使用 NNI](./WindowsLocalMode.md) 
    * NNI 可在 Windows 上使用本机模式
* [支持新的 Advisor: BOHB](./BohbAdvisor.md) 
    * 支持新的 BOHB Advisor，这是一个健壮而有效的超参调优算法，囊括了贝叶斯优化和 Hyperband 的优点
* [支持通过 nnictl 来导入导出 Experiment 数据](./Nnictl.md#experiment) 
    * Generate analysis results report after the experiment execution
    * Support import data to tuner and advisor for tuning
* [Designated gpu devices for NNI trial jobs](./ExperimentConfig.md#localConfig) 
    * Specify GPU devices for NNI trial jobs by gpuIndices configuration, if gpuIndices is set in experiment configuration file, only the specified GPU devices are used for NNI trial jobs.
* Web Portal enhancement 
    * Decimal format of metrics other than default on the Web UI
    * Hints in WebUI about Multi-phase
    * Enable copy/paste for hyperparameters as python dict
    * Enable early stopped trials data for tuners.
* NNICTL provide better error message 
    * nnictl provide more meaningful error message for YAML file format error

### Bug 修复

* Unable to kill all python threads after nnictl stop in async dispatcher mode
* nnictl --version does not work with make dev-install
* All trail jobs status stays on 'waiting' for long time on PAI platform

## Release 0.6 - 4/2/2019

### 主要功能

* [Version checking](https://github.com/Microsoft/nni/blob/master/docs/en_US/PaiMode.md#version-check) 
    * check whether the version is consistent between nniManager and trialKeeper
* [Report final metrics for early stop job](https://github.com/Microsoft/nni/issues/776) 
    * If includeIntermediateResults is true, the last intermediate result of the trial that is early stopped by assessor is sent to tuner as final result. The default value of includeIntermediateResults is false.
* [Separate Tuner/Assessor](https://github.com/Microsoft/nni/issues/841) 
    * Adds two pipes to separate message receiving channels for tuner and assessor.
* Make log collection feature configurable
* Add intermediate result graph for all trials

### Bug 修复

* [Add shmMB config key for PAI](https://github.com/Microsoft/nni/issues/842)
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
* Debuggability/serviceability document: https://nni.readthedocs.io/en/latest/HowToDebug.html
* Tuner assessor reference: https://nni.readthedocs.io/en/latest/sdk_reference.html#tuner

### Bug 修复和其它更新

* Fix a race condition bug that does not store trial job cancel status correctly.
* Fix search space parsing error when using SMAC tuner.
* Fix cifar10 example broken pipe issue.
* Add unit test cases for nnimanager and local training service.
* Add integration test azure pipelines for remote machine, OpenPAI and kubeflow training services.
* Support Pylon in OpenPAI webhdfs client.

## Release 0.5.1 - 1/31/2018

### 改进

* Making [log directory](https://github.com/Microsoft/nni/blob/v0.5.1/docs/en_US/ExperimentConfig.md) configurable
* Support [different levels of logs](https://github.com/Microsoft/nni/blob/v0.5.1/docs/en_US/ExperimentConfig.md), making it easier for debugging 

### 文档

* Reorganized documentation & New Homepage Released: https://nni.readthedocs.io/en/latest/

### Bug 修复和其它更新

* Fix the bug of installation in python virtualenv, and refactor the installation logic
* Fix the bug of HDFS access failure on OpenPAI mode after OpenPAI is upgraded. 
* Fix the bug that sometimes in-place flushed stdout makes experiment crash

## Release 0.5.0 - 01/14/2019

### 主要功能

#### 支持新的 Tuner 和 Assessor

* Support [Metis tuner](MetisTuner.md) as a new NNI tuner. Metis algorithm has been proofed to be well performed for **online** hyper-parameter tuning.
* Support [ENAS customized tuner](https://github.com/countif/enas_nni), a tuner contributed by github community user, is an algorithm for neural network search, it could learn neural network architecture via reinforcement learning and serve a better performance than NAS.
* Support [Curve fitting assessor](CurvefittingAssessor.md) for early stop policy using learning curve extrapolation.
* Advanced Support of [Weight Sharing](./AdvancedNas.md): Enable weight sharing for NAS tuners, currently through NFS.

#### 改进训练平台

* [FrameworkController Training service](./FrameworkControllerMode.md): Support run experiments using frameworkcontroller on kubernetes 
    * FrameworkController is a Controller on kubernetes that is general enough to run (distributed) jobs with various machine learning frameworks, such as tensorflow, pytorch, MXNet.
    * NNI provides unified and simple specification for job definition.
    * MNIST example for how to use FrameworkController.

#### 改进用户体验

* A better trial logging support for NNI experiments in OpenPAI, Kubeflow and FrameworkController mode: 
    * An improved logging architecture to send stdout/stderr of trials to NNI manager via Http post. NNI manager will store trial's stdout/stderr messages in local log file.
    * Show the link for trial log file on WebUI.
* Support to show final result's all key-value pairs.

## Release 0.4.1 - 12/14/2018

### 主要功能

#### 支持新的 Tuner

* Support [network morphism](NetworkmorphismTuner.md) as a new tuner

#### 改进训练平台

* Migrate [Kubeflow training service](KubeflowMode.md)'s dependency from kubectl CLI to [Kubernetes API](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) client
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

### 新样例

* [FashionMnist](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism), work together with network morphism tuner
* [Distributed MNIST example](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch) written in PyTorch

## Release 0.4 - 12/6/2018

### 主要功能

* [Kubeflow Training service](./KubeflowMode.md) 
    * Support tf-operator
    * [Distributed trial example](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed/dist_mnist.py) on Kubeflow
* [Grid search tuner](GridsearchTuner.md) 
* [Hyperband tuner](HyperbandAdvisor.md)
* Support launch NNI experiment on MAC
* WebUI 
    * UI support for hyperband tuner
    * Remove tensorboard button
    * Show experiment error message
    * Show line numbers in search space and trial profile
    * Support search a specific trial by trial number
    * Show trial's hdfsLogPath
    * Download experiment parameters

### 其它

* Asynchronous dispatcher
* Docker file update, add pytorch library 
* Refactor 'nnictl stop' process, send SIGTERM to nni manager process, rather than calling stop Rest API. 
* OpenPAI training service bug fix 
    * Support NNI Manager IP configuration(nniManagerIp) in OpenPAI cluster config file, to fix the issue that user’s machine has no eth0 device 
    * File number in codeDir is capped to 1000 now, to avoid user mistakenly fill root dir for codeDir
    * Don’t print useless ‘metrics is empty’ log in OpenPAI job’s stdout. Only print useful message once new metrics are recorded, to reduce confusion when user checks OpenPAI trial’s output for debugging purpose
    * Add timestamp at the beginning of each log entry in trial keeper.

## Release 0.3.0 - 11/2/2018

### NNICTL 的新功能和更新

* Support running multiple experiments simultaneously.
    
    Before v0.3, NNI only supports running single experiment once a time. After this release, users are able to run multiple experiments simultaneously. Each experiment will require a unique port, the 1st experiment will be set to the default port as previous versions. You can specify a unique port for the rest experiments as below:
    
    ```bash
    nnictl create --port 8081 --config <config file path>
    ```

* Support updating max trial number. use `nnictl update --help` to learn more. Or refer to [NNICTL Spec](Nnictl.md) for the fully usage of NNICTL.

### API 的新功能和更新

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

### 支持新的 Tuner

* **Batch Tuner** which iterates all parameter combination, can be used to submit batch trial jobs.

### 新样例

* A NNI Docker image for public usage:
    
    ```bash
    docker pull msranni/nni:latest
    ```

* New trial example: [NNI Sklearn Example](https://github.com/Microsoft/nni/tree/master/examples/trials/sklearn)

* New competition example: [Kaggle Competition TGS Salt Example](https://github.com/Microsoft/nni/tree/master/examples/trials/kaggle-tgs-salt)

### 其它

* UI refactoring, refer to [WebUI doc](WebUI.md) for how to work with the new UI.
* Continuous Integration: NNI had switched to Azure pipelines
* [Known Issues in release 0.3.0](https://github.com/Microsoft/nni/labels/nni030knownissues).

## Release 0.2.0 - 9/29/2018

### 主要功能

* Support [OpenPAI](https://github.com/Microsoft/pai) Training Platform (See [here](./PaiMode.md) for instructions about how to submit NNI job in pai mode) 
    * Support training services on pai mode. NNI trials will be scheduled to run on OpenPAI cluster
    * NNI trial's output (including logs and model file) will be copied to OpenPAI HDFS for further debugging and checking
* Support [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) tuner (See [here](SmacTuner.md) for instructions about how to use SMAC tuner) 
    * [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO to handle categorical parameters. The SMAC supported by NNI is a wrapper on [SMAC3](https://github.com/automl/SMAC3)
* Support NNI installation on [conda](https://conda.io/docs/index.html) and python virtual environment
* Others 
    * Update ga squad example and related documentation
    * WebUI UX small enhancement and bug fix

### 已知问题

[0.2.0 的已知问题](https://github.com/Microsoft/nni/labels/nni020knownissues)。

## Release 0.1.0 - 9/10/2018 (initial release)

首次发布 Neural Network Intelligence (NNI)。

### 主要功能

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