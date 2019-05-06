<p align="center">
<img src="docs/img/nni_logo.png" width="300"/>
</p>

* * *

[![MIT 许可证](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE) [![生成状态](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/Microsoft.nni)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=6) [![问题](https://img.shields.io/github/issues-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen) [![Bug](https://img.shields.io/github/issues/Microsoft/nni/bug.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen+label%3Abug) [![拉取请求](https://img.shields.io/github/issues-pr-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/pulls?q=is%3Apr+is%3Aopen) [![版本](https://img.shields.io/github/release/Microsoft/nni.svg)](https://github.com/Microsoft/nni/releases) [![进入 https://gitter.im/Microsoft/nni 聊天室提问](https://badges.gitter.im/Microsoft/nni.svg)](https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![文档状态](https://readthedocs.org/projects/nni/badge/?version=latest)](https://nni.readthedocs.io/en/latest/?badge=latest)

[English](README.md)

NNI (Neural Network Intelligence) 是自动机器学习（AutoML）的工具包。 它通过多种调优的算法来搜索最好的神经网络结构和（或）超参，并支持单机、本地多机、云等不同的运行环境。

### **NNI [v0.7](https://github.com/Microsoft/nni/releases) 已发布！**

<p align="center">
  <a href="#nni-v05-has-been-released"><img src="docs/img/overview.svg" /></a>
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
        <b>训练服务</b>
        <img src="docs/img/bar.png"/>
      </td>
    </tr>
    <tr/>
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
      <td>
        <a href="docs/zh_CN/Builtin_Tuner.md">Tuner（调参器）</a>
        <ul>
          <li><a href="docs/zh_CN/Builtin_Tuner.md#TPE">TPE</a></li>
          <li><a href="docs/zh_CN/Builtin_Tuner.md#Random">Random Search（随机搜索）</a></li>
          <li><a href="docs/zh_CN/Builtin_Tuner.md#Anneal">Anneal（退火算法）</a></li>
          <li><a href="docs/zh_CN/Builtin_Tuner.md#Evolution">Naive Evolution（进化算法）</a></li>
          <li><a href="docs/zh_CN/Builtin_Tuner.md#SMAC">SMAC</a></li>
          <li><a href="docs/zh_CN/Builtin_Tuner.md#Batch">Batch（批处理）</a></li>
          <li><a href="docs/zh_CN/Builtin_Tuner.md#Grid">Grid Search（遍历搜索）</a></li>
          <li><a href="docs/zh_CN/Builtin_Tuner.md#Hyperband">Hyperband</a></li>
          <li><a href="docs/zh_CN/Builtin_Tuner.md#NetworkMorphism">Network Morphism</a></li>
          <li><a href="examples/tuners/enas_nni/README_zh_CN.md">ENAS</a></li>
          <li><a href="docs/zh_CN/Builtin_Tuner.md#NetworkMorphism#MetisTuner">Metis Tuner</a></li>
          <li><a href="docs/zh_CN/Builtin_Tuner.md#BOHB">BOHB</a></li>
        </ul>
          <a href="docs/zh_CN/Builtin_Assessors.md#assessor">Assessor（评估器）</a> 
        <ul>
          <li><a href="docs/zh_CN/Builtin_Assessors.md#Medianstop">Median Stop</a></li>
          <li><a href="docs/zh_CN/Builtin_Assessors.md#Curvefitting">Curve Fitting</a></li>
        </ul>
      </td>
      <td>
      <ul>
        <li><a href="docs/zh_CN/LocalMode.md">本地计算机</a></li>
        <li><a href="docs/zh_CN/RemoteMachineMode.md">远程计算机</a></li>
        <li><a href="docs/zh_CN/PAIMode.md">OpenPAI</a></li>
        <li><a href="docs/zh_CN/KubeflowMode.md">Kubeflow</a></li>
        <li><a href="docs/zh_CN/FrameworkControllerMode.md">基于 Kubernetes（AKS 等等）的 FrameworkController</a></li>
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
* [MMdnn](https://github.com/Microsoft/MMdnn)：一个完整、跨框架的解决方案，能够转换、可视化、诊断深度神经网络模型。 MMdnn 中的 "MM" 表示model management（模型管理），而 "dnn" 是 deep neural network（深度神经网络）的缩写。 我们鼓励研究人员和学生利用这些项目来加速 AI 开发和研究。

## **安装和验证**

在 Windows 本机模式下，并且是第一次使用 PowerShell 来运行脚本，需要**使用管理员权限**运行一次下列命令：

```bash
    Set-ExecutionPolicy -ExecutionPolicy Unrestricted
```

**通过 pip 命令安装**

* 当前支持 Linux，MacOS 和 Windows（本机模式），在 Ubuntu 16.04 或更高版本，MacOS 10.14.1 以及 Windows 10.1809 上进行了测试。 在 `python >= 3.5` 的环境中，只需要运行 `pip install` 即可完成安装。

Linux 和 MacOS

```bash
python3 -m pip install --upgrade nni
```

Windows

```bash
python -m pip install --upgrade nni
```

注意：

* 如果需要将 NNI 安装到自己的 home 目录中，可使用 `--user`，这样也不需要任何特殊权限。
* 当前 NNI 在 Windows 上仅支持本机模式。 强烈推荐使用 Anaconda 在 Windows 上安装 NNI。
* 如果遇到如`Segmentation fault` 这样的任何错误请参考[常见问题](docs/zh_CN/FAQ.md)。

**通过源代码安装**

* 当前支持 Linux（Ubuntu 16.04 或更高版本），MacOS（10.14.1）以及 Windows 10（1809 版）下的本机模式。 

Linux 和 MacOS

* 在 `python >= 3.5` 的环境中运行命令： `git` 和 `wget`，确保安装了这两个组件。

```bash
    git clone -b v0.7 https://github.com/Microsoft/nni.git
    cd nni
    source install.sh
```

Windows

* 在 `python >=3.5` 的环境中运行命令： `git` 和 `PowerShell`，确保安装了这两个组件。

```bash
  git clone -b v0.7 https://github.com/Microsoft/nni.git
  cd nni
  powershell ./install.ps1
```

参考[安装 NNI](docs/zh_CN/Installation.md) 了解系统需求。

参考 [NNI Windows 本机模式](docs/zh_CN/WindowsLocalMode.md)，了解更多信息。

**验证安装**

The following example is an experiment built on TensorFlow. Make sure you have **TensorFlow installed** before running it.

* Download the examples via clone the source code.

```bash
    git clone -b v0.7 https://github.com/Microsoft/nni.git
```

Linux and MacOS

* Run the MNIST example.

```bash
    nnictl create --config nni/examples/trials/mnist/config.yml
```

Windows

* Run the MNIST example.

```bash
    nnictl create --config nni/examples/trials/mnist/config_windows.yml
```

* Wait for the message `INFO: Successfully started experiment!` in the command line. This message indicates that your experiment has been successfully started. You can explore the experiment using the `Web UI url`.

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

* Open the `Web UI url` in your browser, you can view detail information of the experiment and all the submitted trial jobs as shown below. [Here](docs/en_US/WebUI.md) are more Web UI pages.

<table style="border: none">
    <th><img src="./docs/img/webui_overview_page.png" alt="drawing" width="395"/></th>
    <th><img src="./docs/img/webui_trialdetail_page.png" alt="drawing" width="410"/></th>
</table>

## **文档**

* [NNI overview](docs/en_US/Overview.md)
* [Quick start](docs/en_US/QuickStart.md)

## **入门**

* [Install NNI](docs/en_US/Installation.md)
* [Use command line tool nnictl](docs/en_US/NNICTLDOC.md)
* [Use NNIBoard](docs/en_US/WebUI.md)
* [How to define search space](docs/en_US/SearchSpaceSpec.md)
* [How to define a trial](docs/en_US/Trials.md)
* [How to choose tuner/search-algorithm](docs/en_US/Builtin_Tuner.md)
* [Config an experiment](docs/en_US/ExperimentConfig.md)
* [How to use annotation](docs/en_US/Trials.md#nni-python-annotation)

## **教程**

* [Run an experiment on local (with multiple GPUs)?](docs/en_US/LocalMode.md)
* [Run an experiment on multiple machines?](docs/en_US/RemoteMachineMode.md)
* [Run an experiment on OpenPAI?](docs/en_US/PAIMode.md)
* [Run an experiment on Kubeflow?](docs/en_US/KubeflowMode.md)
* [Try different tuners](docs/en_US/tuners.rst)
* [Try different assessors](docs/en_US/assessors.rst)
* [Implement a customized tuner](docs/en_US/Customize_Tuner.md)
* [Implement a customized assessor](docs/en_US/Customize_Assessor.md)
* [Use Genetic Algorithm to find good model architectures for Reading Comprehension task](examples/trials/ga_squad/README.md)

## **贡献**

This project welcomes contributions and suggestions, we use [GitHub issues](https://github.com/Microsoft/nni/issues) for tracking requests and bugs.

Issues with the **good first issue** label are simple and easy-to-start ones that we recommend new contributors to start with.

To set up environment for NNI development, refer to the instruction: [Set up NNI developer environment](docs/en_US/SetupNNIDeveloperEnvironment.md)

Before start coding, review and get familiar with the NNI Code Contribution Guideline: [Contributing](docs/en_US/CONTRIBUTING.md)

We are in construction of the instruction for [How to Debug](docs/en_US/HowToDebug.md), you are also welcome to contribute questions or suggestions on this area.

## **许可协议**

The entire codebase is under [MIT license](LICENSE)