# Release 0.5.0 - 01/14/2019

## 主要功能

### New tuner and assessor supports

* Support [Metis tuner](./HowToChooseTuner.md#MetisTuner) as a new NNI tuner. Metis algorithm has been proofed to be well performed for **online** hyper-parameter tuning.
* Support [ENAS customized tuner](https://github.com/countif/enas_nni), a tuner contributed by github community user, is an algorithm for neural network search, it could learn neural network architecture via reinforcement learning and serve a better performance than NAS.
* Support [Curve fitting assessor](./HowToChooseTuner.md#Curvefitting) for early stop policy using learning curve extrapolation. 
* Advanced Support of [Weight Sharing](./AdvancedNAS.md): Enable weight sharing for NAS tuners, currently through NFS.

### Training Service Enhancement

* [FrameworkController Training service](./FrameworkControllerMode.md): Support run experiments using frameworkcontroller on kubernetes 
   * FrameworkController is a Controller on kubernetes that is general enough to run (distributed) jobs with various machine learning frameworks, such as tensorflow, pytorch, MXNet.
   * NNI provides unified and simple specification for job definition.
   * MNIST example for how to use FrameworkController.

### User Experience improvements

* A better trial logging support for NNI experiments in PAI, Kubeflow and FrameworkController mode: * An improved logging architecture to send stdout/stderr of trials to NNI manager via Http post. NNI manager will store trial's stdout/stderr messages in local log file. * Show the link for trial log file on WebUI. 
* Support to show final result's all key-value pairs.

# Release 0.4.1 - 12/14/2018

## Major Features

### New tuner supports

* Support [network morphism](./HowToChooseTuner.md#NetworkMorphism) as a new tuner

### Training Service improvements

* Migrate [Kubeflow training service](https://github.com/Microsoft/nni/blob/master/docs/KubeflowMode.md)'s dependency from kubectl CLI to [Kubernetes API](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) client
* [Pytorch-operator](https://github.com/kubeflow/pytorch-operator) support for Kubeflow training service
* Improvement on local code files uploading to OpenPAI HDFS
* Fixed OpenPAI integration WebUI bug: WebUI doesn't show latest trial job status, which is caused by OpenPAI token expiration

### NNICTL improvements

* Show version information both in nnictl and WebUI. You can run **nnictl -v** to show your current installed NNI version

### WebUI improvements

* Enable modify concurrency number during experiment
* Add feedback link to NNI github 'create issue' page
* Enable customize top 10 trials regarding to metric numbers (largest or smallest)
* Enable download logs for dispatcher & nnimanager 
* Enable automatic scaling of axes for metric number
* Update annotation to support displaying real choice in searchspace

## 新样例

* [FashionMnist](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism), work together with network morphism tuner
* [Distributed MNIST example](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch) written in PyTorch

# 发布 0.4 - 12/6/2018

## Major Features

* [Kubeflow 训练服务](./KubeflowMode.md) 
   * 支持 tf-operator
   * Kubeflow 上的[分布式尝试样例](../examples/trials/mnist-distributed/dist_mnist.py)
* [网格搜索调参器](../src/sdk/pynni/nni/README.md#Grid) 
* [Hyperband 调参器](../src/sdk/pynni/nni/README.md#Hyperband)
* 支持在 MAC 上运行 NNI 实验
* WebUI 
   * 支持 hyperband 调参器
   * 移除 tensorboard 按钮 
   * 显示实验的错误消息 
   * 显示搜索空间和尝试配置的行号
   * 支持通过指定的尝试 id 来搜索
   * 显示尝试的 hdfsLogPath
   * 下载实验参数

## 其它

* 异步调度
* 更新 Docker 文件，增加 pytorch 库 
* 重构 'nnictl stop' 过程，发送 SIGTERM 给 NNI 管理器进程，而不是调用停止 Restful API. 
* OpenPAI 训练服务修复缺陷 
   * 在 NNI 管理器中为 PAI 集群配置文件支持 IP 配置(nniManagerIp)，来修复用户计算机没有 eth0 设备的问题。 
   * codeDir 中的文件数量上限改为1000，避免用户无意中填写了 root 目录。
   * 移除 PAI 作业的 stdout 日志中无用的 ‘metrics is empty’。 在新指标被记录时，仅输出有用的消息，来减少用户检查 PAI 尝试输出时的困惑。
   * 在尝试 keeper 的开始增加时间戳。

# 发布 0.3.0 - 11/2/2018

## NNICTL 的新功能和更新

* 支持同时运行多个实验。
   
   在 v0.3 以前，NNI 仅支持一次运行一个实验。 此版本开始，用户可以同时运行多个实验。 每个实验都需要一个唯一的端口，第一个实验会像以前版本一样使用默认端口。 需要为其它实验指定唯一端口：
   
       nnictl create --port 8081 --config <config file path>

* 支持更新最大尝试的数量。 使用 ```nnictl update --help``` 了解更多信息。 或参考 [NNICTL 说明](https://github.com/Microsoft/nni/blob/master/docs/NNICTLDOC.md)来查看完整帮助。

## API 的新功能和更新

* <span style="color:red"><strong>不兼容的改动</strong></span>：nn.get_parameters() 改为 nni.get_next_parameter。 所有以前版本的样例将无法在 v0.3 上运行，需要重新克隆 NNI 代码库获取新样例。 如果在自己的代码中使用了 NNI，也需要相应的更新。

* 新 API **nni.get_sequence_id()**。 每个尝试任务都会被分配一个唯一的序列数字，可通过 nni.get_sequence_id() API 来获取。
   
       git clone -b v0.3 https://github.com/Microsoft/nni.git

* **nni.report_final_result(result)** API 支持了更多结果参数的类型。 可用类型： 
   * int
   * float
   * 包含有 'default' 键值的 dict，'default' 的值必须为 int 或 float。 dict 可以包含任何其它键值对。

## 新的内置调参器

* **Batch Tuner（批处理调参器）** 会执行所有曹参组合，可被用来批量提交尝试任务。

## 新样例

* 公共的 NNI Docker 映像： ```docker pull msranni/nni:latest```
* 新的尝试样例： [NNI Sklearn 样例](https://github.com/Microsoft/nni/tree/master/examples/trials/sklearn)
* 新的竞赛样例：[Kaggle Competition TGS Salt](https://github.com/Microsoft/nni/tree/master/examples/trials/kaggle-tgs-salt)

## 其它

* 界面重构，参考[网页文档](WebUI.md)，了解如何使用新界面。
* 持续集成：NNI 已切换到 Azure pipelines。
* [0.3.0 的已知问题](https://github.com/Microsoft/nni/labels/nni030knownissues)。

# 发布 0.2.0 - 9/29/2018

## Major Features

    * Support [OpenPAI](https://github.com/Microsoft/pai) (aka pai) Training Service (See [here](./PAIMode.md) for instructions about how to submit NNI job in pai mode)
       * Support training services on pai mode. NNI trials will be scheduled to run on OpenPAI cluster
       * NNI trial's output (including logs and model file) will be copied to OpenPAI HDFS for further debugging and checking
    * Support [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) tuner (See [here](HowToChooseTuner.md) for instructions about how to use SMAC tuner)
       * [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) is based on Sequential Model-Based Optimization (SMBO). 它会利用使用过的突出的模型（高斯随机过程模型），并将随机森林引入到SMBO中，来处理分类参数。 The SMAC supported by NNI is a wrapper on [SMAC3](https://github.com/automl/SMAC3)
    * Support NNI installation on [conda](https://conda.io/docs/index.html) and python virtual environment
    * Others
       * Update ga squad example and related documentation
       * WebUI UX small enhancement and bug fix
    

## 已知问题

[0.2.0 的已知问题](https://github.com/Microsoft/nni/labels/nni020knownissues)。

# 发布 0.1.0 - 9/10/2018 (首个版本)

首次发布 Neural Network Intelligence (NNI)。

## Major Features

    * Installation and Deployment
       * Support pip install and source codes install
       * Support training services on local mode(including Multi-GPU mode) as well as multi-machines mode
    * Tuners, Assessors and Trial
       * Support AutoML algorithms including:  hyperopt_tpe, hyperopt_annealing, hyperopt_random, and evolution_tuner
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
    

## 已知问题

[0.1.0 的已知问题](https://github.com/Microsoft/nni/labels/nni010knownissues)。