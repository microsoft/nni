# 发布 0.4 - 12/6/2018

## 主要功能

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

## New examples

* A NNI Docker image for public usage: ```docker pull msranni/nni:latest```
* New trial example: [NNI Sklearn Example](https://github.com/Microsoft/nni/tree/master/examples/trials/sklearn)
* New competition example: [Kaggle Competition TGS Salt Example](https://github.com/Microsoft/nni/tree/master/examples/trials/kaggle-tgs-salt)

## Others

* UI refactoring, refer to [WebUI doc](WebUI.md) for how to work with the new UI.
* Continuous Integration: NNI had switched to Azure pipelines
* [Known Issues in release 0.3.0](https://github.com/Microsoft/nni/labels/nni030knownissues).

# Release 0.2.0 - 9/29/2018

## Major Features

* Support [OpenPAI](https://github.com/Microsoft/pai) (aka pai) Training Service (See [here](./PAIMode.md) for instructions about how to submit NNI job in pai mode) 
    * Support training services on pai mode. NNI trials will be scheduled to run on OpenPAI cluster
    * NNI trial's output (including logs and model file) will be copied to OpenPAI HDFS for further debugging and checking
* Support [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) tuner (See [here](HowToChooseTuner.md) for instructions about how to use SMAC tuner) 
    * [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO to handle categorical parameters. The SMAC supported by NNI is a wrapper on [SMAC3](https://github.com/automl/SMAC3)
* Support NNI installation on [conda](https://conda.io/docs/index.html) and python virtual environment
* Others 
    * Update ga squad example and related documentation
    * WebUI UX small enhancement and bug fix

## Known Issues

[Known Issues in release 0.2.0](https://github.com/Microsoft/nni/labels/nni020knownissues).

# Release 0.1.0 - 9/10/2018 (initial release)

Initial release of Neural Network Intelligence (NNI).

## Major Features

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

## Known Issues

[Known Issues in release 0.1.0](https://github.com/Microsoft/nni/labels/nni010knownissues).