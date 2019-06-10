# 更改日志

# 发布 0.8 - 6/4/2019

## 主要功能

* 在 Windows 上支持 NNI 的 OpenPAI 和远程模式 
    * NNI 可在 Windows 上使用 OpenPAI 模式
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

* 修复 `nnictl update` 不一致的命令行风格
* SMAC Tuner 支持导入数据
* 支持 Experiment 状态从 ERROR 回到 RUNNING
* 修复表格的 Bug
* 优化嵌套搜索空间
* 优化 'randint' 类型，并支持下限
* [比较不同超参搜索调优算法](./CommunitySharings/HpoComparision.md)
* [NAS 算法的对比](./CommunitySharings/NasComparision.md)
* [Recommenders 上的实践](./CommunitySharings/NniPracticeSharing/RecommendersSvd.md)

## 发布 0.7 - 4/29/2018

### 主要功能

* [支持在 Windows 上使用 NNI](./WindowsLocalMode.md) 
    * NNI 可在 Windows 上使用本机模式
* [支持新的 Advisor: BOHB](./BohbAdvisor.md) 
    * 支持新的 BOHB Advisor，这是一个健壮而有效的超参调优算法，囊括了贝叶斯优化和 Hyperband 的优点
* [支持通过 nnictl 来导入导出 Experiment 数据](./Nnictl.md#experiment) 
    * 在 Experiment 执行完后，可生成分析结果报告
    * 支持将先前的调优数据导入到 Tuner 和 Advisor 中
* [可为 NNI Trial 任务指定 GPU](./ExperimentConfig.md#localConfig) 
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

* [版本检查](https://github.com/Microsoft/nni/blob/master/docs/en_US/PaiMode.md#version-check) 
    * 检查 nniManager 和 trialKeeper 的版本是否一致
* [提前终止的任务也可返回最终指标](https://github.com/Microsoft/nni/issues/776) 
    * 如果 includeIntermediateResults 为 true，最后一个 Assessor 的中间结果会被发送给 Tuner 作为最终结果。 includeIntermediateResults 的默认值为 false。
* [分离 Tuner/Assessor](https://github.com/Microsoft/nni/issues/841) 
    * 增加两个管道来分离 Tuner 和 Assessor 的消息
* 使日志集合功能可配置
* 为所有 Trial 增加中间结果的视图

### Bug 修复

* [为 OpenPAI 增加 shmMB 配置](https://github.com/Microsoft/nni/issues/842)
* 修复在指标为 dict 时，无法显示任何结果的 Bug。
* 修复 hyperband 中浮点类型的计算问题
* 修复 SMAC Tuner 中搜索空间转换的错误
* 修复 Web 界面中解析 Experiment 的错误格式
* 修复 Metis Tuner 冷启动时的错误

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
* 修复 CIFAR-10 样例中的 broken pipe 问题。
* 为本地训练服务和 NNI 管理器添加单元测试。
* 为远程服务器、OpenPAI 和 Kubeflow 训练平台在 Azure 中增加集成测试。
* 在 OpenPAI 客户端中支持 Pylon 路径。

## 发布 0.5.1 - 1/31/2018

### 改进

* [日志目录](https://github.com/Microsoft/nni/blob/v0.5.1/docs/zh_CN/ExperimentConfig.md)可配置。
* 支持[不同级别的日志](https://github.com/Microsoft/nni/blob/v0.5.1/docs/zh_CN/ExperimentConfig.md)，使其更易于调试。 

### 文档

* 重新组织文档，新的主页位置：https://nni.readthedocs.io/zh/latest/

### Bug 修复和其它更新

* 修复了 Python 虚拟环境中安装的 Bug，并重构了安装逻辑。
* 修复了在最新的 OpenPAI 下存取 HDFS 失败的问题。 
* 修复了有时刷新 stdout 会造成 Experiment 崩溃的问题。

## 发布 0.5.0 - 01/14/2019

### 主要功能

#### 支持新的 Tuner 和 Assessor

* 支持新的 [Metis Tuner](MetisTuner.md)。 **在线**超参调优的场景下，Metis 算法已经被证明非常有效。
* 支持 [ENAS customized tuner](https://github.com/countif/enas_nni)。由 GitHub 社区用户所贡献。它是神经网络的搜索算法，能够通过强化学习来学习神经网络架构，比 NAS 的性能更好。
* 支持 [Curve fitting （曲线拟合）Assessor](CurvefittingAssessor.md)，通过曲线拟合的策略来实现提前终止 Trial。
* 进一步支持 [Weight Sharing（权重共享）](./AdvancedNas.md)：为 NAS Tuner 通过 NFS 来提供权重共享。

#### 改进训练平台

* [FrameworkController 训练平台](./FrameworkControllerMode.md): 支持使用在 Kubernetes 上使用 FrameworkController。 
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

* 支持新的 [network morphism](NetworkmorphismTuner.md) Tuner。

#### 改进训练平台

* 将[Kubeflow 训练平台](KubeflowMode.md)的依赖从 kubectl CLI 迁移到 [Kubernetes API](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) 客户端。
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

### 新样例

* [FashionMnist](https://github.com/Microsoft/nni/tree/master/examples/trials/network_morphism)，使用 network morphism Tuner
* 使用 PyTorch 的[分布式 MNIST 样例](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed-pytorch)

## 发布 0.4 - 12/6/2018

### 主要功能

* [Kubeflow 训练服务](./KubeflowMode.md) 
    * 支持 tf-operator
    * 使用 Kubeflow 的[分布式 Trial 样例](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-distributed/dist_mnist.py)
* [网格搜索 Tuner](GridsearchTuner.md) 
* [Hyperband Tuner](HyperbandAdvisor.md)
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

* 支持更新最大 Trial 的数量。 使用 `nnictl update --help` 了解详情。 或参考 [NNICTL](Nnictl.md) 查看完整帮助。

### API 的新功能和更新

* <span style="color:red"><strong>不兼容的改动</strong></span>：nn.get_parameters() 改为 nni.get_next_parameter。 所有以前版本的样例将无法在 v0.3 上运行，需要重新克隆 NNI 代码库获取新样例。 如果在自己的代码中使用了 NNI，也需要相应的更新。

* 新 API **nni.get_sequence_id()**。 每个 Trial 任务都会被分配一个唯一的序列数字，可通过 nni.get_sequence_id() API 来获取。
    
    ```bash
    git clone -b v0.3 https://github.com/Microsoft/nni.git
    ```

* **nni.report_final_result(result)** API 对结果参数支持更多的数据类型。
    
    可用类型：
    
    * int
    * float
    * 包含有 'default' 键值的 dict，'default' 的值必须为 int 或 float。 dict 可以包含任何其它键值对。

### 支持新的 Tuner

* **Batch Tuner（批处理调参器）** 会执行所有超参组合，可被用来批量提交 Trial 任务。

### 新样例

* 公开的 NNI Docker 映像：
    
    ```bash
    docker pull msranni/nni:latest
    ```

* 新的 Trial 样例： [NNI Sklearn 样例](https://github.com/Microsoft/nni/tree/master/examples/trials/sklearn)

* 新的竞赛样例：[Kaggle Competition TGS Salt](https://github.com/Microsoft/nni/tree/master/examples/trials/kaggle-tgs-salt)

### 其它

* 界面重构，参考[网页文档](WebUI.md)，了解如何使用新界面。
* 持续集成：NNI 已切换到 Azure pipelines。
* [0.3.0 的已知问题](https://github.com/Microsoft/nni/labels/nni030knownissues)。

## 发布 0.2.0 - 9/29/2018

### 主要功能

* 支持 [OpenPAI](https://github.com/Microsoft/pai) (又称 pai) 训练服务 (参考[这里](./PaiMode.md)来了解如何在 OpenPAI 下提交 NNI 任务) 
    * 支持 pai 模式的训练服务。 NNI Trial 可发送至 OpenPAI 集群上运行
    * NNI Trial 输出 (包括日志和模型文件) 会被复制到 OpenPAI 的 HDFS 中。
* 支持 [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) Tuner (参考[这里](SmacTuner.md)，了解如何使用 SMAC Tuner) 
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