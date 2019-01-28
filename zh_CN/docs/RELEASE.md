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

* 在 OpenPAI，Kubeflow 和 FrameworkController 模式中提供了更好的日志支持： * 通过改进的日志架构来将 Trial 的 stdout/stderr 通过 发送给 NNI 管理器。 NNI 管理器将 Trial 的 stdout/stderr 消息存储在本地日志文件中。 * 在 WEB 界面上显示 Trial 日志的链接。 
* 支持将最终结果显示为键值对。

## Release 0.4.1 - 12/14/2018

### Major Features

#### New tuner supports

* 支持新 Tuner [network morphism](./HowToChooseTuner.md#NetworkMorphism)

#### Training Service improvements

* 将 [Kubeflow 训练服务](https://github.com/Microsoft/nni/blob/master/docs/KubeflowMode.md)的依赖从 kubectl CLI 迁移到 [Kubernetes API](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) 客户端。
* Kubeflow 训练服务支持 [Pytorch-operator](https://github.com/kubeflow/pytorch-operator)。
* 改进将本地代码文件上传到 OpenPAI HDFS 的性能。
* 修复 OpenPAI 在 WEB 界面的缺陷：当 OpenPAI 认证过期后，Web 界面无法更新 Trial 作业的状态。

#### NNICTL improvements

* 在 nnictl 和 WEB 界面中显示 NNI 的版本信息。 可使用 **nnictl -v** 来显示安装的 NNI 版本。

#### WebUI improvements

* 在 Experiment 运行中可修改并发数量
* 增加指向 NNI Github 的反馈链接，可直接创建问题
* 可根据指标，定制选择（最大或最小）的前 10 个 Trial。
* 为 dispatcher 和 nnimanager 提供下载日志的功能 
* 为指标数值图提供自动缩放的数轴
* 改进标记，支持在搜索空间中显示实际的选项

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
* OpenPAI 训练服务修复缺陷 
   * 在 NNI 管理器中为 PAI 集群配置文件支持 IP 配置(nniManagerIp)，来修复用户计算机没有 eth0 设备的问题。 
   * codeDir 中的文件数量上限改为1000，避免用户无意中填写了 root 目录。
   * 移除 PAI 作业的 stdout 日志中无用的 ‘metrics is empty’。 在新指标被记录时，仅输出有用的消息，来减少用户检查 PAI Trial 输出时的困惑。
   * 在 Trial keeper 的开始增加时间戳。

## Release 0.3.0 - 11/2/2018

### NNICTL new features and updates

* 支持同时运行多个 Experiment。
   
   在 v0.3 以前，NNI 仅支持一次运行一个 Experiment。 此版本开始，用户可以同时运行多个 Experiment。 每个 Experiment 都需要一个唯一的端口，第一个 Experiment 会像以前版本一样使用默认端口。 需要为其它 Experiment 指定唯一端口：
   
       nnictl create --port 8081 --config <config file path>

* 支持更新最大 Trial 的数量。 使用 ```nnictl update --help``` 了解更多信息。 或参考 [NNICTL 说明](https://github.com/Microsoft/nni/blob/master/docs/NNICTLDOC.md)来查看完整帮助。

### API new features and updates

* <span style="color:red"><strong>不兼容的改动</strong></span>：nn.get_parameters() 改为 nni.get_next_parameter。 所有以前版本的样例将无法在 v0.3 上运行，需要重新克隆 NNI 代码库获取新样例。 如果在自己的代码中使用了 NNI，也需要相应的更新。

* 新 API **nni.get_sequence_id()**。 每个 Trial 任务都会被分配一个唯一的序列数字，可通过 nni.get_sequence_id() API 来获取。
   
       git clone -b v0.3 https://github.com/Microsoft/nni.git

* **nni.report_final_result(result)** API 支持了更多结果参数的类型。 可用类型： 
   * int
   * float
   * 包含有 'default' 键值的 dict，'default' 的值必须为 int 或 float。 dict 可以包含任何其它键值对。

### New tuner support

* **Batch Tuner（批处理调参器）** 会执行所有超参组合，可被用来批量提交 Trial 任务。

### New examples

* 公开的 NNI Docker 映像： ```docker pull msranni/nni:latest```
* 新的 Trial 样例： [NNI Sklearn 样例](https://github.com/Microsoft/nni/tree/master/examples/trials/sklearn)
* 新的竞赛样例：[Kaggle Competition TGS Salt](https://github.com/Microsoft/nni/tree/master/examples/trials/kaggle-tgs-salt)

### Others

* 界面重构，参考[网页文档](WebUI.md)，了解如何使用新界面。
* 持续集成：NNI 已切换到 Azure pipelines。
* [0.3.0 的已知问题](https://github.com/Microsoft/nni/labels/nni030knownissues)。

## Release 0.2.0 - 9/29/2018

### Major Features

    * 支持 [OpenPAI](https://github.com/Microsoft/pai) (aka pai) 作为训练服务（参考 [!这里](./PAIMode.md)，了解如何在 pai 模式下提交 NNI 作业）。
       * 训练服务支持 pai 模式。 Trial 可发送至 OpenPAI 集群上运行
       * Trial 的输出 (包括日志和模型文件) 会被复制到 OpenPAI 的 HDFS 中。
    * 支持 <a href="https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf">SMAC</a> Tuner (参考[这里](HowToChooseTuner.md)，了解如何使用 SMAC Tuner)
       * <a href="https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf">SMAC</a> 基于 Sequential Model-Based Optimization (SMBO). 它会利用使用过的好模型（高斯随机过程模型），并将随机森林引入到SMBO中，来处理分类参数。 NNI 的 SMAC 通过包装 <a href="https://github.com/automl/SMAC3">SMAC3</a> 来支持。
    * 支持将 NNI 安装在 <a href="https://conda.io/docs/index.html">conda</a> 和 Python 虚拟环境中。
    * 其它
       * 更新 ga squad 样例与相关文档
       * 用户体验改善及缺陷修复
    

### Known Issues

[0.2.0 的已知问题](https://github.com/Microsoft/nni/labels/nni020knownissues)。

## Release 0.1.0 - 9/10/2018 (initial release)

首次发布 Neural Network Intelligence (NNI)。

### Major Features

    * 安装和部署
       * 支持 pip 和源代码安装
       * 支持本机（包括多 GPU 卡）训练和远程多机训练模式
    * Tuner，Assessor 和 Trial
       * 支持自动机器学习算法，包括： hyperopt_tpe, hyperopt_annealing, hyperopt_random, 和 evolution_tuner。
       * 支持 Assessor（提前终止）算法，包括：medianstop。
       * 提供 Python API 来自定义 Tuner 和 Assessor
       * 提供 Python API 来包装 Trial 代码，以便能在 NNI 中运行
    * 实验
       * 提供命令行工具 'nnictl' 来管理 Experiment
       * 提供网页界面来查看并管理 Experiment
    * 持续集成
       * 使用 Ubuntu 的 [travis-ci](https://github.com/travis-ci) 来支持持续集成
    * 其它
       * 支持简单的 GPU 任务调度 
    

### Known Issues

[0.1.0 的已知问题](https://github.com/Microsoft/nni/labels/nni010knownissues)。