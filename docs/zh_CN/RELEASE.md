# 更改日志

## Release 0.7 - 4/29/2018

### Major Features

* [Support NNI on Windows](./WindowsLocalMode.md) 
    * NNI running on windows for local mode
* [New advisor: BOHB](./bohbAdvisor.md) 
    * Support a new advisor BOHB, which is a robust and efficient hyperparameter tuning algorithm, combines the advantages of Bayesian optimization and Hyperband
* [Support import and export experiment data through nnictl](./NNICTLDOC.md#experiment) 
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
    * nnictl provide more meaningful error message for yaml file format error

### Bug fix

* Unable to kill all python threads after nnictl stop in async dispatcher mode
* nnictl --version does not work with make dev-instal
* All trail jobs status stays on 'waiting' for long time on PAI platform

## 发布 0.6 - 4/2/2019

### 主要功能

* [版本检查](https://github.com/Microsoft/nni/blob/master/docs/en_US/PAIMode.md#version-check) 
    * check whether the version is consistent between nniManager and trialKeeper
* [Report final metrics for early stop job](https://github.com/Microsoft/nni/issues/776) 
    * If includeIntermediateResults is true, the last intermediate result of the trial that is early stopped by assessor is sent to tuner as final result. The default value of includeIntermediateResults is false.
* [Separate Tuner/Assessor](https://github.com/Microsoft/nni/issues/841) 
    * Adds two pipes to separate message receiving channels for tuner and assessor.
* Make log collection feature configurable
* Add intermediate result graph for all trials

### Bug fix

* [Add shmMB config key for PAI](https://github.com/Microsoft/nni/issues/842)
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
* 调试和维护：https://nni.readthedocs.io/en/latest/HowToDebug.html
* Tuner、Assessor 参考：https://nni.readthedocs.io/en/latest/sdk_reference.html#tuner

### Bug 修复和其它更新

* 修复了在某些极端条件下，不能正确存储任务的取消状态。
* 修复在使用 SMAC Tuner 时，解析搜索空间的错误。
* Fix cifar10 example broken pipe issue.
* Add unit test cases for nnimanager and local training service.
* Add integration test azure pipelines for remote machine, OpenPAI and kubeflow training services.
* Support Pylon in OpenPAI webhdfs client.

## Release 0.5.1 - 1/31/2018

### Improvements

* Making [log directory](https://github.com/Microsoft/nni/blob/v0.5.1/docs/en_US/ExperimentConfig.md) configurable
* Support [different levels of logs](https://github.com/Microsoft/nni/blob/v0.5.1/docs/en_US/ExperimentConfig.md), making it easier for debugging 

### Documentation

* Reorganized documentation & New Homepage Released: https://nni.readthedocs.io/en/latest/

### Bug Fixes and Other Changes

* Fix the bug of installation in python virtualenv, and refactor the installation logic
* Fix the bug of HDFS access failure on OpenPAI mode after OpenPAI is upgraded. 
* Fix the bug that sometimes in-place flushed stdout makes experiment crash

## Release 0.5.0 - 01/14/2019

### Major Features

#### 支持新的 Tuner 和 Assessor

* Support [Metis tuner](metisTuner.md) as a new NNI tuner. Metis algorithm has been proofed to be well performed for **online** hyper-parameter tuning.
* Support [ENAS customized tuner](https://github.com/countif/enas_nni), a tuner contributed by github community user, is an algorithm for neural network search, it could learn neural network architecture via reinforcement learning and serve a better performance than NAS.
* Support [Curve fitting assessor](curvefittingAssessor.md) for early stop policy using learning curve extrapolation.
* Advanced Support of [Weight Sharing](./AdvancedNAS.md): Enable weight sharing for NAS tuners, currently through NFS.

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

### Major Features

#### 支持新的 Tuner

* Support [network morphism](networkmorphismTuner.md) as a new tuner

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

### New examples

* [FashionMnist](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism), work together with network morphism tuner
* [Distributed MNIST example](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch) written in PyTorch

## Release 0.4 - 12/6/2018

### Major Features

* [Kubeflow Training service](./KubeflowMode.md) 
    * Support tf-operator
    * [Distributed trial example](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed/dist_mnist.py) on Kubeflow
* [Grid search tuner](gridsearchTuner.md) 
* [Hyperband tuner](hyperbandAdvisor.md)
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

* Support updating max trial number. use `nnictl update --help` to learn more. Or refer to [NNICTL Spec](NNICTLDOC.md) for the fully usage of NNICTL.

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

### 主要功能

* 支持 [OpenPAI](https://github.com/Microsoft/pai) (又称 pai) 训练服务 (参考[这里](./PAIMode.md)来了解如何在 OpenPAI 下提交 NNI 任务) 
    * 支持 pai 模式的训练服务。 NNI Trial 可发送至 OpenPAI 集群上运行
    * NNI Trial 输出 (包括日志和模型文件) 会被复制到 OpenPAI 的 HDFS 中。
* 支持 [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) Tuner (参考[这里](smacTuner.md)，了解如何使用 SMAC Tuner) 
    * [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) 基于 Sequential Model-Based Optimization (SMBO). 它会利用使用过的结果好的模型（高斯随机过程模型），并将随机森林引入到 SMBO 中，来处理分类参数。 NNI 的 SMAC 通过包装 [SMAC3](https://github.com/automl/SMAC3) 来支持。
* 支持将 NNI 安装在 [conda](https://conda.io/docs/index.html) 和 Python 虚拟环境中。
* 其它 
    * 更新 ga squad 样例与相关文档
    * 用户体验改善及 Bug 修复

### 已知问题

[0.2.0 的已知问题](https://github.com/Microsoft/nni/labels/nni020knownissues)。

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

### 已知问题

[0.1.0 的已知问题](https://github.com/Microsoft/nni/labels/nni010knownissues)。