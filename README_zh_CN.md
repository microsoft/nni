<p align="center">
<img src="docs/img/nni_logo.png" width="300"/>
</p>

* * *

[![MIT 许可证](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE) [![生成状态](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/Microsoft.nni)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=6) [![问题](https://img.shields.io/github/issues-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen) [![Bug](https://img.shields.io/github/issues/Microsoft/nni/bug.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen+label%3Abug) [![拉取请求](https://img.shields.io/github/issues-pr-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/pulls?q=is%3Apr+is%3Aopen) [![版本](https://img.shields.io/github/release/Microsoft/nni.svg)](https://github.com/Microsoft/nni/releases) [![进入 https://gitter.im/Microsoft/nni 聊天室提问](https://badges.gitter.im/Microsoft/nni.svg)](https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![文档状态](https://readthedocs.org/projects/nni/badge/?version=latest)](https://nni.readthedocs.io/en/latest/?badge=latest)

[English](README.md)

NNI (Neural Network Intelligence) 是自动机器学习（AutoML）的工具包。 它通过多种调优的算法来搜索最好的神经网络结构和（或）超参，并支持单机、本地多机、云等不同的运行环境。

### **NNI [v0.9](https://github.com/Microsoft/nni/releases) 已发布！ &nbsp;[<img width="48" src="docs/img/release_icon.png" />](#nni-released-reminder)**

<p align="center">
  <a href="#nni-has-been-released"><img src="docs/img/overview.svg" /></a>
</p>

<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>支持的框架</b>
        <img src="docs/img/bar.png"/>
      </td>
      <td>
        <b>调优算法</b>
        <img src="docs/img/bar.png"/>
      </td>
      <td>
        <b>训练平台</b>
        <img src="docs/img/bar.png"/>
      </td>
    </tr>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li>PyTorch</li>
          <li>TensorFlow</li>
          <li>Keras</li>
          <li>MXNet</li>
          <li>Caffe2</li>
          <li>CNTK (Python 语言)</li>
          <li>Chainer</li>
          <li>Theano</li>
        </ul>
      </td>
      <td align="left">
        <a href="docs/en_US/Tuner/BuiltinTuner.md">Tuner（调参器）</a>
        <br />
        <ul>
          <b style="margin-left:-20px">通用 Tuner</b>
          <li><a href="docs/en_US/Tuner/BuiltinTuner.md#Random">Random Search（随机搜索）</a></li>
          <li><a href="docs/en_US/Tuner/BuiltinTuner.md#Evolution">Naïve Evolution（进化算法）</a></li>    
          <b style="margin-left:-20px">超参 Tuner</b>
          <li><a href="docs/en_US/Tuner/BuiltinTuner.md#TPE">TPE</a></li>
          <li><a href="docs/en_US/Tuner/BuiltinTuner.md#Anneal">Anneal（退火算法）</a></li>
          <li><a href="docs/en_US/Tuner/BuiltinTuner.md#SMAC">SMAC</a></li>
          <li><a href="docs/en_US/Tuner/BuiltinTuner.md#Batch">Batch（批处理）</a></li>
          <li><a href="docs/en_US/Tuner/BuiltinTuner.md#GridSearch">Grid Search（遍历搜索）</a></li>
          <li><a href="docs/en_US/Tuner/BuiltinTuner.md#Hyperband">Hyperband</a></li>
          <li><a href="docs/en_US/Tuner/BuiltinTuner.md#MetisTuner">Metis Tuner</a></li>
          <li><a href="docs/en_US/Tuner/BuiltinTuner.md#BOHB">BOHB</a></li>
          <li><a href="docs/en_US/Tuner/BuiltinTuner.md#GPTuner">GP Tuner</a></li>
          <b style="margin-left:-20px">网络结构 Tuner</b>
          <li><a href="docs/en_US/Tuner/BuiltinTuner.md#NetworkMorphism">Network Morphism</a></li>
          <li><a href="examples/tuners/enas_nni/README.md">ENAS</a></li>
          </ul>
          <a href="docs/en_US/Assessor/BuiltinAssessor.md">Assessor（评估器）</a>
          <ul>
          <li><a href="docs/en_US/Assessor/BuiltinAssessor.md#Medianstop">Median Stop（中位数终止）</a></li>
          <li><a href="docs/en_US/Assessor/BuiltinAssessor.md#Curvefitting">Curve Fitting（曲线拟合）</a></li>   
          </ul>  
      </td>
      <td>
      <ul>
        <li><a href="docs/en_US/TrainingService/LocalMode.md">本机</a></li>
        <li><a href="docs/en_US/TrainingService/RemoteMachineMode.md">远程计算机</a></li>
        <li><b>基于 Kubernetes 的平台</b></li>
            <ul><li><a href="docs/en_US/TrainingService/PaiMode.md">OpenPAI</a></li>
            <li><a href="docs/en_US/TrainingService/KubeflowMode.md">Kubeflow</a></li>
            <li><a href="docs/en_US/TrainingService/FrameworkControllerMode.md">基于 Kubernetes（AKS 等）的 FrameworkController</a></li>
            </ul>
      </ul>
      </td>
    </tr> 
  </tbody>
</table>

## **使用场景**

* 在本机尝试使用不同的自动机器学习（AutoML）算法来训练模型。
* 在分布式环境中加速自动机器学习（如：远程 GPU 工作站和云服务器）。
* 定制自动机器学习算法，或比较不同的自动机器学习算法。
* 在机器学习平台中支持自动机器学习。

## 相关项目

以开发和先进技术为目标，[Microsoft Research (MSR)](https://www.microsoft.com/en-us/research/group/systems-research-group-asia/) 发布了一些开源项目。

* [OpenPAI](https://github.com/Microsoft/pai)：作为开源平台，提供了完整的 AI 模型训练和资源管理能力，能轻松扩展，并支持各种规模的私有部署、云和混合环境。
* [FrameworkController](https://github.com/Microsoft/frameworkcontroller)：开源的通用 Kubernetes Pod 控制器，通过单个控制器来编排 Kubernetes 上所有类型的应用。
* [MMdnn](https://github.com/Microsoft/MMdnn)：一个完整、跨框架的解决方案，能够转换、可视化、诊断深度神经网络模型。 MMdnn 中的 "MM" 表示model management（模型管理），而 "dnn" 是 deep neural network（深度神经网络）的缩写。
* [SPTAG](https://github.com/Microsoft/SPTAG) : Space Partition Tree And Graph (SPTAG) 是用于大规模向量的最近邻搜索场景的开源库。

我们鼓励研究人员和学生利用这些项目来加速 AI 开发和研究。

## **安装和验证**

**通过 pip 命令安装**

* 当前支持 Linux，MacOS 和 Windows（本机，远程，OpenPAI 模式），在 Ubuntu 16.04 或更高版本，MacOS 10.14.1 以及 Windows 10.1809 上进行了测试。 在 `python >= 3.5` 的环境中，只需要运行 `pip install` 即可完成安装。

Linux 和 macOS

```bash
python3 -m pip install --upgrade nni
```

Windows

```bash
python -m pip install --upgrade nni
```

注意：

* 如果需要将 NNI 安装到自己的 home 目录中，可使用 `--user`，这样也不需要任何特殊权限。
* 目前，Windows 上的 NNI 支持本机，远程和 OpenPAI 模式。 强烈推荐使用 Anaconda 或 Miniconda 在 Windows 上安装 NNI。
* 如果遇到如`Segmentation fault` 这样的任何错误请参考[常见问题](docs/zh_CN/Tutorial/FAQ.md)。

**通过源代码安装**

* 当前支持 Linux（Ubuntu 16.04 或更高版本），MacOS（10.14.1）以及 Windows 10（1809 版）。

Linux 和 macOS

* 在 `python >= 3.5` 的环境中运行命令： `git` 和 `wget`，确保安装了这两个组件。

```bash
    git clone -b v0.9 https://github.com/Microsoft/nni.git
    cd nni
    source install.sh
```

Windows

* 在 `python >=3.5` 的环境中运行命令： `git` 和 `PowerShell`，确保安装了这两个组件。

```bash
  git clone -b v0.9 https://github.com/Microsoft/nni.git
  cd nni
  powershell -ExecutionPolicy Bypass -file install.ps1
```

参考[安装 NNI](docs/zh_CN/Tutorial/Installation.md) 了解系统需求。

Windows 上参考 [Windows 上使用 NNI](docs/zh_CN/Tutorial/NniOnWindows.md)。

**验证安装**

以下示例 Experiment 依赖于 TensorFlow 。 在运行前确保安装了 **TensorFlow**。

* 通过克隆源代码下载示例。

```bash
    git clone -b v0.9 https://github.com/Microsoft/nni.git
```

Linux 和 macOS

* 运行 MNIST 示例。

```bash
    nnictl create --config nni/examples/trials/mnist/config.yml
```

Windows

* 运行 MNIST 示例。

```bash
    nnictl create --config nni\examples\trials\mnist\config_windows.yml
```

* 在命令行中等待输出 `INFO: Successfully started experiment!`。 此消息表明 Experiment 已成功启动。 通过命令行输出的 `Web UI url` 来访问 Experiment 的界面。

```text
INFO: Starting restful server...
INFO: Successfully started Restful server!
INFO: Setting local config...
INFO: Successfully set local config!
INFO: Starting experiment...
INFO: Successfully started experiment!
-----------------------------------------------------------------------
The experiment id is egchD4qy
The Web UI urls are: http://223.255.255.1:8080   http://127.0.0.1:8080
-----------------------------------------------------------------------

You can use these commands to get more information about the experiment
-----------------------------------------------------------------------
         commands                       description

1. nnictl experiment show        show the information of experiments
2. nnictl trial ls               list all of trial jobs
3. nnictl top                    monitor the status of running experiments
4. nnictl log stderr             show stderr log content
5. nnictl log stdout             show stdout log content
6. nnictl stop                   stop an experiment
7. nnictl trial kill             kill a trial job by id
8. nnictl --help                 get help information about nnictl
-----------------------------------------------------------------------
```

* 在浏览器中打开 `Web UI url`，可看到下图的 Experiment 详细信息，以及所有的 Trial 任务。 查看[这里](docs/zh_CN/Tutorial/WebUI.md)的更多页面。

<table style="border: none">
    <th><img src="./docs/img/webui_overview_page.png" alt="drawing" width="395"/></th>
    <th><img src="./docs/img/webui_trialdetail_page.png" alt="drawing" width="410"/></th>
</table>

## **文档**

主要文档都可以在[这里](https://nni.readthedocs.io/cn/latest/Overview.html)找到，文档均从本代码库生成。  
点击阅读：

* [NNI 概述](docs/zh_CN/Overview.md)
* [快速入门](docs/en_US/Tutorial/QuickStart.md)
* [贡献](docs/en_US/Tutorial/Contributing.md)
* [示例](docs/en_US/examples.rst)
* [参考](docs/en_US/reference.rst)
* [Web 界面教程](docs/en_US/Tutorial/WebUI.md)

## **入门**

* [安装 NNI](docs/en_US/Tutorial/Installation.md)
* [使用命令行工具 nnictl](docs/en_US/Tutorial/Nnictl.md)
* [使用 NNIBoard](docs/en_US/Tutorial/WebUI.md)
* [如何定义搜索空间](docs/en_US/Tutorial/SearchSpaceSpec.md)
* [如何实现 Trial 代码](docs/en_US/TrialExample/Trials.md)
* [如何选择 Tuner、搜索算法](docs/en_US/Tuner/BuiltinTuner.md)
* [配置 Experiment](docs/en_US/Tutorial/ExperimentConfig.md)
* [如何使用 Annotation](docs/en_US/TrialExample/Trials.md#nni-python-annotation)

## **教程**

* [在 OpenPAI 上运行 Experiment](docs/en_US/TrainingService/PaiMode.md)
* [在 Kubeflow 上运行 Experiment](docs/en_US/TrainingService/KubeflowMode.md)
* [在本机运行 Experiment (支持多 GPU 卡)](docs/en_US/TrainingService/LocalMode.md)
* [在多机上运行 Experiment](docs/en_US/TrainingService/RemoteMachineMode.md)
* [尝试不同的 Tuner](docs/en_US/Tuner/BuiltinTuner.md)
* [尝试不同的 Assessor](docs/en_US/Assessor/BuiltinAssessor.md)
* [实现自定义 Tuner](docs/en_US/Tuner/CustomizeTuner.md)
* [实现自定义 Assessor](docs/en_US/Assessor/CustomizeAssessor.md)
* [使用进化算法为阅读理解任务找到好模型](docs/en_US/TrialExample/SquadEvolutionExamples.md)

## **贡献**

非常欢迎通过各种方式参与此项目，例如：

* 审查[源代码改动](https://github.com/microsoft/nni/pulls)
* 审查[文档](https://github.com/microsoft/nni/tree/master/docs)中从拼写错误到新内容的任何内容，并提交拉取请求。
* 找到标有 ['good first issue'](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) 或 ['help-wanted'](https://github.com/microsoft/nni/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) 标签的 Issue。这些都是简单的 Issue，新的贡献者可以从这些问题开始。

在提交代码前，需要遵循以下的简单准则：

* [如何调试](docs/en_US/Tutorial/HowToDebug.md)
* [代码风格和命名约定](docs/en_US/Tutorial/Contributing.md)
* 如何设置 [NNI 开发环境](docs/zh_CN/Tutorial/SetupNniDeveloperEnvironment.md)
* 查看[贡献说明](docs/en_US/Tutorial/Contributing.md)并熟悉 NNI 的代码贡献指南

## **外部代码库**

下面是一些贡献者为 NNI 提供的使用示例 谢谢可爱的贡献者！ 欢迎越来越多的人加入我们！

* 在 NNI 中运行 [ENAS](examples/tuners/enas_nni/README_zh_CN.md)
* 在 NNI 中运行 [神经网络架构结构搜索](examples/trials/nas_cifar10/README_zh_CN.md) 

## **反馈**

* [报告 Bug](https://github.com/microsoft/nni/issues/new/choose)。  
    
* [请求新功能](https://github.com/microsoft/nni/issues/new/choose).
* 在 [Gitter](https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) 中参与讨论
* 在 [Stack Overflow](https://stackoverflow.com/questions/tagged/nni?sort=Newest&edited=true) 上使用 nni 的标签提问，或[在 Github 上提交 Issue](https://github.com/microsoft/nni/issues/new/choose)。
* 我们正在实现[如何调试](docs/zh_CN/Tutorial/HowToDebug.md)的页面，欢迎提交建议和问题。

## **许可协议**

代码库遵循 [MIT 许可协议](LICENSE)