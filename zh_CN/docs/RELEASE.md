# 发布 0.5.0 - 01/14/2019

## 主要功能

### 支持新的调参器和评估器

* 支持 [Metis tuner](./HowToChooseTuner.md#MetisTuner) 作为 NNI 的调参器。 **在线**超参调优的场景下，Metis 算法已经被证明非常有效。
* 支持 [ENAS customized tuner](https://github.com/countif/enas_nni)。由 GitHub 社区用户所贡献。它是神经网络的搜索算法，能够通过强化学习来学习神经网络架构，比 NAS 的性能更好。
* 支持 [Curve fitting （曲线拟合）评估器](./HowToChooseTuner.md#Curvefitting)，通过曲线拟合的方式的策略来实现提前终止尝试。 
* 进一步支持 [Weight Sharing（权重共享）](./AdvancedNAS.md)：为 NAS 调参器通过 NFS 来提供权重共享。

### 改进训练服务

* [FrameworkController 训练服务](./FrameworkControllerMode.md): 支持使用在 Kubernetes 上使用 FrameworkController。 
   * FrameworkController 是 Kubernetes 上非常通用的控制器（Controller），能用来运行基于各种机器学习框架的分布式作业，如 TensorFlow，Pytorch， MXNet 等。
   * NNI 为作业定义了统一而简单的规范。
   * 如何使用 FrameworkController 的 MNIST 样例。

### 改进用户体验

* 在 OpenPAI，Kubeflow 和 FrameworkController 模式中提供了更好的日志支持： * 通过改进的日志架构来将尝试的 stdout/stderr 通过 发送给 NNI 管理器。 NNI 管理器将尝试的 stdout/stderr 消息存储在本地日志文件中。 * 在 WEB 界面上显示尝试日志的链接。 
* 支持将最终结果显示为键值对。

# 发布 0.4.1 - 12/14/2018

## 主要功能

### 支持新的调参器

* 支持新调参器 [network morphism](./HowToChooseTuner.md#NetworkMorphism)

### 改进训练服务

* 将 [Kubeflow 训练服务](https://github.com/Microsoft/nni/blob/master/docs/KubeflowMode.md)的依赖从 kubectl CLI 迁移到 [Kubernetes API](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) 客户端。
* Kubeflow 训练服务支持 [Pytorch-operator](https://github.com/kubeflow/pytorch-operator)。
* 改进将本地代码文件上传到 OpenPAI HDFS 的性能。
* 修复 OpenPAI 在 WEB 界面的缺陷：当 OpenPAI 认证过期后，Web 界面无法更新尝试作业的状态。

### 改进 NNICTL

* 在 nnictl 和 WEB 界面中显示 NNI 的版本信息。 可使用 **nnictl -v** 来显示安装的 NNI 版本。

### 改进 WEB 界面

* 在实验运行中可修改并发数量
* 增加指向 NNI Github 的反馈链接，可直接创建问题
* 可根据指标，定制选择（最大或最小）的前 10 个尝试。
* 为 dispatcher 和 nnimanager 提供下载日志的功能 
* 为指标数值图提供自动缩放的数轴
* 改进标记，支持在搜索空间中显示实际的选项

## 新样例

* [FashionMnist](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism)，使用 network morphism 调参器
* 使用 PyTorch 的[分布式 MNIST 样例](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)

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

## 新样例

* 公开的 NNI Docker 映像： ```docker pull msranni/nni:latest```
* 新的尝试样例： [NNI Sklearn 样例](https://github.com/Microsoft/nni/tree/master/examples/trials/sklearn)
* 新的竞赛样例：[Kaggle Competition TGS Salt](https://github.com/Microsoft/nni/tree/master/examples/trials/kaggle-tgs-salt)

## 其它

* 界面重构，参考[网页文档](WebUI.md)，了解如何使用新界面。
* 持续集成：NNI 已切换到 Azure pipelines。
* [0.3.0 的已知问题](https://github.com/Microsoft/nni/labels/nni030knownissues)。

# 发布 0.2.0 - 9/29/2018

## 主要功能

    * 支持 [OpenPAI](https://github.com/Microsoft/pai) (aka pai) 作为训练服务（参考 [!这里](./PAIMode.md)，了解如何在 pai 模式下提交 NNI 作业）。
       * 训练服务支持 pai 模式。 NNI 尝试可发送至 OpenPAI 集群上运行
       * NNI 尝试输出 (包括日志和模型文件) 会被复制到 OpenPAI 的 HDFS 中。
    * 支持 <a href="https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf">SMAC</a> 调参器 (参考[这里](HowToChooseTuner.md)，了解如何使用 SMAC 调参器)
       * <a href="https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf">SMAC</a> 基于 Sequential Model-Based Optimization (SMBO). 它会利用使用过的好模型（高斯随机过程模型），并将随机森林引入到SMBO中，来处理分类参数。 NNI 的 SMAC 通过包装 <a href="https://github.com/automl/SMAC3">SMAC3</a> 来支持。
    * 支持将 NNI 安装在 <a href="https://conda.io/docs/index.html">conda</a> 和 Python 虚拟环境中。
    * 其它
       * 更新 ga squad 样例与相关文档
       * 用户体验改善及缺陷修复
    

## 已知问题

[0.2.0 的已知问题](https://github.com/Microsoft/nni/labels/nni020knownissues)。

# 发布 0.1.0 - 9/10/2018 (首个版本)

首次发布 Neural Network Intelligence (NNI)。

## 主要功能

    * 安装和部署
       * 支持 pip 和源代码安装
       * 支持本机（包括多 GPU 卡）训练和远程多机训练模式
    * 调参器，评估器和尝试
       * 支持自动机器学习算法，包括： hyperopt_tpe, hyperopt_annealing, hyperopt_random, 和 evolution_tuner。
       * 支持评估器（提前终止）算法，包括：medianstop。
       * 提供 Python API 来自定义调参器和评估器
       * 提供 Python API 来包装尝试代码，以便能在 NNI 中运行
    * 实验
       * 提供命令行工具 'nnictl' 来管理实验
       * 提供网页界面来查看并管理实验
    * 持续集成
       * 使用 Ubuntu 的 [travis-ci](https://github.com/travis-ci) 来支持持续集成
    * 其它
       * 支持简单的 GPU 任务调度 
    

## 已知问题

[0.1.0 的已知问题](https://github.com/Microsoft/nni/labels/nni010knownissues)。