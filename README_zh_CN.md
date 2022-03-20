<p align="center">
<img src="docs/img/nni_logo.png" width="300"/>
</p>

* * *

[![MIT 许可证](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE) [![生成状态](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/full%20test%20-%20linux?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=62&branchName=master) [![问题](https://img.shields.io/github/issues-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen) [![Bug](https://img.shields.io/github/issues/Microsoft/nni/bug.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen+label%3Abug) [![拉取请求](https://img.shields.io/github/issues-pr-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/pulls?q=is%3Apr+is%3Aopen) [![版本](https://img.shields.io/github/release/Microsoft/nni.svg)](https://github.com/Microsoft/nni/releases) [![进入 https://gitter.im/Microsoft/nni 聊天室提问](https://badges.gitter.im/Microsoft/nni.svg)](https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![文档状态](https://readthedocs.org/projects/nni/badge/?version=stable)](https://nni.readthedocs.io/zh/stable/?badge=stable)

[NNI 文档](https://nni.readthedocs.io/zh/stable/) | [English](README.md)

**NNI (Neural Network Intelligence)** 是一个帮助用户**自动**进行[特征工程](docs/zh_CN/FeatureEngineering/Overview.rst)，[神经网络架构搜索](docs/zh_CN/NAS/Overview.rst)，[超参调优](docs/zh_CN/Tuner/BuiltinTuner.rst)以及[模型压缩](docs/zh_CN/Compression/Overview.rst)的轻量且强大的工具包。

NNI 管理自动机器学习 (AutoML) 的 Experiment，**调度运行**由调优算法生成的 Trial 任务来找到最好的神经网络架构和/或超参，支持**各种训练环境**，如[本机](docs/zh_CN/TrainingService/LocalMode.rst)，[远程服务器](docs/zh_CN/TrainingService/RemoteMachineMode.rst)，[OpenPAI](docs/zh_CN/TrainingService/PaiMode.rst)，[Kubeflow](docs/zh_CN/TrainingService/KubeflowMode.rst)，[基于 K8S 的 FrameworkController（如，AKS 等）](docs/zh_CN/TrainingService/FrameworkControllerMode.rst)， [DLWorkspace (又称 DLTS)](docs/zh_CN/TrainingService/DLTSMode.rst), [AML (Azure Machine Learning)](docs/zh_CN/TrainingService/AMLMode.rst), [AdaptDL（又称 ADL）](docs/zh_CN/TrainingService/AdaptDLMode.rst) ，和其他的云平台甚至 [混合模式](docs/zh_CN/TrainingService/HybridMode.rst) 。 DLTS)</a>，[AML (Azure Machine Learning)](https://nni.readthedocs.io/zh/stable/TrainingService/AMLMode.html)[AdaptDL（又称 ADL）](https://nni.readthedocs.io/zh/stable/TrainingService/AdaptDLMode.html) ，和其他的云平台甚至[混合模式](https://nni.readthedocs.io/zh/stable/TrainingService/HybridMode.html) 。

## **使用场景**

* 想要在自己的代码、模型中试验**不同的自动机器学习算法**。
* 想要在**不同的环境中**加速运行自动机器学习。
* 想要更容易**实现或试验新的自动机器学习算法**的研究员或数据科学家，包括：超参调优算法，神经网络搜索算法以及模型压缩算法。
* 在机器学习平台中**支持自动机器学习**。

## **最新消息！** &nbsp;[<img width="48" src="docs/img/release_icon.png" />](#nni-released-reminder)

* **最新版本**：[v2.6 已发布](https://github.com/microsoft/nni/releases/tag/v2.6) - *2022年1月19日*
* **最新视频 demo**：[Youtube 入口](https://www.youtube.com/channel/UCKcafm6861B2mnYhPbZHavw) | [Bilibili 入口](https://space.bilibili.com/1649051673) - *上次更新：2021年5月26日*
* **最新网络研讨会**: [介绍Retiarii：NNI 上的深度学习探索性训练框架](https://note.microsoft.com/MSR-Webinar-Retiarii-Registration-Live.html) - *2021年6月24日*
* **最新互动渠道**: [Discussions](https://github.com/microsoft/nni/discussions)
* **最新粉丝福利表情包上线**： [nnSpider](./docs/en_US/Tutorial/NNSpider.md)
<p align="center">
  <a href="#nni-spider"><img width="100%" src="docs/img/emoicons/home.svg" /></a>
</p>

## **NNI 功能一览**

NNI 提供命令行工具以及友好的 WebUI 来管理训练的 Experiment。 通过可扩展的 API，可定制自动机器学习算法和训练平台。 为了方便新用户，NNI 内置了最新的自动机器学习算法，并为流行的训练平台提供了开箱即用的支持。

下表中，包含了 NNI 的功能，同时在不断地增添新功能，也非常希望您能贡献其中。

<p align="center">
  <a href="#nni-has-been-released"><img src="docs/img/overview.svg" /></a>
</p>

<table>
  <tbody>
    <tr align="center" valign="bottom">
    <td>
      </td>
      <td>
        <b>支持的框架和库</b>
        <img src="docs/img/bar.png"/>
      </td>
      <td>
        <b>算法</b>
        <img src="docs/img/bar.png"/>
      </td>
      <td>
        <b>训练平台</b>
        <img src="docs/img/bar.png"/>
      </td>
    </tr>
    </tr>
    <tr valign="top">
    <td align="center" valign="middle">
    <b>内置</b>
      </td>
      <td>
      <ul><li><b>支持的框架</b></li>
        <ul>
          <li>PyTorch</li>
          <li>Keras</li>
          <li>TensorFlow</li>
          <li>MXNet</li>
          <li>Caffe2</li>
          <a href="https://nni.readthedocs.io/zh/stable/SupportedFramework_Library.html">更多...</a><br/>
        </ul>
        </ul>
      <ul>
        <li><b>支持的库</b></li>
          <ul>
           <li>Scikit-learn</li>
           <li>XGBoost</li>
           <li>LightGBM</li>
           <a href="https://nni.readthedocs.io/zh/stable/SupportedFramework_Library.html">更多...</a><br/>
          </ul>
      </ul>
        <ul>
        <li><b>示例</b></li>
         <ul>
           <li><a href="examples/trials/mnist-pytorch">MNIST-pytorch</li></a>
           <li><a href="examples/trials/mnist-tfv1">MNIST-tensorflow</li></a>
           <li><a href="examples/trials/mnist-keras">MNIST-keras</li></a>
           <li><a href="https://nni.readthedocs.io/zh/stable/TrialExample/GbdtExample.html">Auto-gbdt</a></li>
           <li><a href="https://nni.readthedocs.io/zh/stable/TrialExample/Cifar10Examples.html">Cifar10-pytorch</li></a>
           <li><a href="https://nni.readthedocs.io/zh/stable/TrialExample/SklearnExamples.html">Scikit-learn</a></li>
           <li><a href="https://nni.readthedocs.io/zh/stable/TrialExample/EfficientNet.html">EfficientNet</a></li>
           <li><a href="https://nni.readthedocs.io/zh/stable/TrialExample/OpEvoExamples.html">GPU Kernel 调优</li></a>
              <a href="https://nni.readthedocs.io/zh/stable/SupportedFramework_Library.html">更多...</a><br/>
          </ul>
        </ul>
      </td>
      <td align="left" >
        <a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html">超参调优</a>
        <ul>
          <b>穷举搜索</b>
          <ul>
            <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#Random">Random Search（随机搜索）</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#GridSearch">Grid Search（遍历搜索）</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#Batch">Batch（批处理）</a></li>
            </ul>
          <b>启发式搜索</b>
          <ul>
            <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#Evolution">Naïve Evolution（朴素进化）</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#Anneal">Anneal（退火算法）</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#Hyperband">Hyperband</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#PBTTuner">PBT</a></li>
          </ul>
          <b>贝叶斯优化</b>
            <ul>
              <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#BOHB">BOHB</a></li>
              <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#TPE">TPE</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#SMAC">SMAC</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#MetisTuner">Metis Tuner</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#GPTuner">GP Tuner</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#DNGOTuner">PPO Tuner</a></li>
            </ul>
        </ul>
          <a href="https://nni.readthedocs.io/zh/stable/NAS/Overview.html">神经网络架构搜索</a>
          <ul>
            <li><a href="https://nni.readthedocs.io/zh/stable/NAS/ENAS.html">ENAS</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/NAS/DARTS.html">DARTS</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/NAS/SPOS.html">SPOS</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/NAS/Proxylessnas.html">ProxylessNAS</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/NAS/FBNet.html">FBNet</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/NAS/ExplorationStrategies.html">基于强化学习</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/NAS/ExplorationStrategies.html">正则进化</a></li>
            <li><a href="https://nni.readthedocs.io/zh/stable/NAS/Overview.html">更多...</a></li>
          </ul>
          <a href="https://nni.readthedocs.io/zh/stable/Compression/Overview.html">模型压缩</a>
          <ul>
            <b>剪枝</b>
            <ul>
              <li><a href="https://nni.readthedocs.io/zh/stable/Compression/Pruner.html#agp-pruner">AGP Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/zh/stable/Compression/Pruner.html#slim-pruner">Slim Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/zh/stable/Compression/Pruner.html#fpgm-pruner">FPGM Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/zh/stable/Compression/Pruner.html#netadapt-pruner">NetAdapt Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/zh/stable/Compression/Pruner.html#simulatedannealing-pruner">SimulatedAnnealing Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/zh/stable/Compression/Pruner.html#admm-pruner">ADMM Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/zh/stable/Compression/Pruner.html#autocompress-pruner">AutoCompress Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/zh/stable/Compression/Overview.html">更多...</a></li>
            </ul>
            <b>量化</b>
            <ul>
              <li><a href="https://nni.readthedocs.io/zh/stable/Compression/Quantizer.html#qat-quantizer">QAT Quantizer</a></li>
              <li><a href="https://nni.readthedocs.io/zh/stable/Compression/Quantizer.html#dorefa-quantizer">DoReFa Quantizer</a></li>
              <li><a href="https://nni.readthedocs.io/zh/stable/Compression/Quantizer.html#bnn-quantizer">BNN Quantizer</a></li>
            </ul>
          </ul>
          <a href="https://nni.readthedocs.io/zh/stable/FeatureEngineering/Overview.html">特征工程（测试版）</a>
          <ul>
          <li><a href="https://nni.readthedocs.io/zh/stable/FeatureEngineering/GradientFeatureSelector.html">GradientFeatureSelector</a></li>
          <li><a href="https://nni.readthedocs.io/zh/stable/FeatureEngineering/GBDTSelector.html">GBDTSelector</a></li>
          </ul>
          <a href="https://nni.readthedocs.io/zh/stable/Assessor/BuiltinAssessor.html">提前终止算法</a>
          <ul>
          <li><a href="https://nni.readthedocs.io/zh/stable/Assessor/BuiltinAssessor.html#MedianStop">Median Stop（中位数终止）</a></li>
          <li><a href="https://nni.readthedocs.io/zh/stable/Assessor/BuiltinAssessor.html#Curvefitting">Curve Fitting（曲线拟合）</a></li>
          </ul>
      </td>
      <td>
      <ul>
        <li><a href="https://nni.readthedocs.io/zh/stable/TrainingService/LocalMode.html">本机</a></li>
        <li><a href="https://nni.readthedocs.io/zh/stable/TrainingService/RemoteMachineMode.html">远程计算机</a></li>
        <li><a href="https://nni.readthedocs.io/zh/stable/TrainingService/HybridMode.html">混合模式</a></li>
        <li><a href="https://nni.readthedocs.io/zh/stable/TrainingService/AMLMode.html">AML(Azure Machine Learning)</a></li>
        <li><b>基于 Kubernetes 的平台</b></li>
        <ul>
          <li><a href="https://nni.readthedocs.io/zh/stable/TrainingService/PaiMode.html">OpenPAI</a></li>
          <li><a href="https://nni.readthedocs.io/zh/stable/TrainingService/KubeflowMode.html">Kubeflow</a></li>
          <li><a href="https://nni.readthedocs.io/zh/stable/TrainingService/FrameworkControllerMode.html">基于 Kubernetes（AKS 等）的 FrameworkController</a></li>
          <li><a href="https://nni.readthedocs.io/zh/stable/TrainingService/DLTSMode.html">DLWorkspace（又称  DLTS）</a></li>
          <li><a href="https://nni.readthedocs.io/zh/stable/TrainingService/AdaptDLMode.html">AdaptDL（又称 ADL）</a></li>
        </ul>
      </ul>
      </td>
    </tr>
      <tr align="center" valign="bottom">
      </td>
      </tr>
      <tr valign="top">
       <td valign="middle">
    <b>参考</b>
      </td>
     <td style="border-top:#FF0000 solid 0px;">
      <ul>
        <li><a href="https://nni.readthedocs.io/zh/stable/autotune_ref.html#trial">Python API</a></li>
        <li><a href="https://nni.readthedocs.io/zh/stable/Tutorial/AnnotationSpec.html">NNI Annotation</a></li>
         <li><a href="https://nni.readthedocs.io/zh/stable/installation.html">支持的操作系统</a></li>
      </ul>
      </td>
       <td style="border-top:#FF0000 solid 0px;">
      <ul>
        <li><a href="https://nni.readthedocs.io/zh/stable/Tuner/CustomizeTuner.html">自定义 Tuner</a></li>
        <li><a href="https://nni.readthedocs.io/zh/stable/Assessor/CustomizeAssessor.html">自定义 Assessor</a></li>
        <li><a href="https://nni.readthedocs.io/zh/stable/Tutorial/InstallCustomizedAlgos.html">安装自定义的 Tuner，Assessor，Advisor</a></li>
        <li><a href="https://nni.readthedocs.io/zh/stable/NAS/QuickStart.html#define-your-model-space">定义模型空间</a></li>
        <li><a href="https://nni.readthedocs.io/zh/stable/NAS/ApiReference.html">NAS/Retiarii接口</a></li>
      </ul>
      </td>
        <td style="border-top:#FF0000 solid 0px;">
      <ul>
        <li><a href="https://nni.readthedocs.io/zh/stable/TrainingService/Overview.html">支持训练平台</li>
        <li><a href="https://nni.readthedocs.io/zh/stable/TrainingService/HowToImplementTrainingService.html">实现训练平台</a></li>
      </ul>
      </td>
    </tr>
  </tbody>
</table>

## **安装**

### **安装**

NNI 支持并在 Ubuntu >= 16.04, macOS >= 10.14.1, 和 Windows 10 >= 1809 通过了测试。 在 `python 64-bit >= 3.6` 的环境中，只需要运行 `pip install` 即可完成安装。

Linux 或 macOS

```bash
python3 -m pip install --upgrade nni
```

Windows

```bash
python -m pip install --upgrade nni
```

如果想试试最新代码，可参考从源代码[安装 NNI](https://nni.readthedocs.io/zh/latest/installation.html)。

Linux 和 macOS 下 NNI 系统需求[参考这里](https://nni.readthedocs.io/zh/latest/Tutorial/InstallationLinux.html#system-requirements) ，Windows [参考这里](https://nni.readthedocs.io/zh/latest/Tutorial/InstallationWin.html#system-requirements)。

注意：

* 如果遇到任何权限问题，可添加 `--user` 在用户目录中安装 NNI。
* 目前，Windows 上的 NNI 支持本机，远程和 OpenPAI 模式。 强烈推荐使用 Anaconda 或 Miniconda [在 Windows 上安装 NNI](https://nni.readthedocs.io/zh/stable/Tutorial/InstallationWin.html)。
* 如果遇到如 `Segmentation fault` 等错误参考[常见问题](docs/zh_CN/Tutorial/FAQ.rst)。 Windows 上的 FAQ 参考[在 Windows 上使用 NNI](docs/zh_CN/Tutorial/InstallationWin.rst#faq)。 Windows 上的 FAQ 参考[在 Windows 上使用 NNI](https://nni.readthedocs.io/zh/stable/Tutorial/InstallationWin.html#faq)。

### **验证安装**

* 通过克隆源代码下载示例。
    
    ```bash
    git clone -b v2.6 https://github.com/Microsoft/nni.git
    ```

* 运行 MNIST 示例。
    
    Linux 或 macOS
    
    ```bash
    nnictl create --config nni/examples/trials/mnist-pytorch/config.yml
    ```
    
    Windows
    
    ```powershell
    nnictl create --config nni\examples\trials\mnist-pytorch\config_windows.yml
    ```

* 在命令行中等待输出 `INFO: Successfully started experiment!`。 此消息表明 Experiment 已成功启动。 通过命令行输出的 `Web UI url` 来访问 Experiment 的界面。 此消息表明 Experiment 已成功启动。 通过命令行输出的 `Web UI url` 来访问 Experiment 的界面。

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

* 在浏览器中打开 `Web UI url`，可看到下图的 Experiment 详细信息，以及所有的 Trial 任务。 查看[这里](docs/zh_CN/Tutorial/WebUI.rst)的更多页面。 查看[这里](https://nni.readthedocs.io/zh/stable/Tutorial/WebUI.html)的更多页面。

<img src="docs/static/img/webui.gif" alt="webui" width="100%" />

## **发布和贡献**

NNI 有一个月度发布周期（主要发布）。 如果您遇到问题可以通过 [创建 issue](https://github.com/microsoft/nni/issues/new/choose) 来报告。

我们感谢所有的贡献。 我们感谢所有的贡献。 如果您计划提供任何 Bug 修复，请放手去做，不需要任何顾虑。

如果您计划提供新的功能、新的 Tuner 和 新的训练平台等， 请先创建一个新的 issue 或重用现有 issue，并与我们讨论该功能。 我们会及时与您讨论这个问题，如有需要会安排电话会议。

再次感谢所有的贡献者！

再次感谢所有的贡献者！

<a href="https://github.com/microsoft/nni/graphs/contributors"><img src="docs/img/contributors.png" /></a>

## **反馈**

* [在 GitHub 上提交问题](https://github.com/microsoft/nni/issues/new/choose)。
* 在 [Gitter](https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) 中参与讨论。 
* NNI 有一个月度发布周期（主要发布）。 如果您遇到问题可以通过 [创建 issue](https://github.com/microsoft/nni/issues/new/choose) 来报告。

加入聊天组： 

| Gitter                                                                                                         |   | 微信                                                                      |
| -------------------------------------------------------------------------------------------------------------- | - | ----------------------------------------------------------------------- |
| ![image](https://user-images.githubusercontent.com/39592018/80665738-e0574a80-8acc-11ea-91bc-0836dc4cbf89.png) | 或 | ![image](https://github.com/scarlett2018/nniutil/raw/master/wechat.png) |

## 测试状态

### 必需

|      类型      |                                                                                                                                        状态                                                                                                                                        |
|:------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  Fast test   |                                [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/fast%20test?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=54&branchName=master)                                |
|  Full linux  | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/full%20test%20-%20linux?repoName=microsoft%2Fnni&branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=62&repoName=microsoft%2Fnni&branchName=master) |
| Full windows |                         [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/full%20test%20-%20windows?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=63&branchName=master)                         |

### 训练平台

|            类型             |                                                                                                                                状态                                                                                                                                 |
|:-------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  Remote - linux to linux  |  [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20remote%20-%20linux%20to%20linux?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=64&branchName=master)  |
| Remote - linux to windows | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20remote%20-%20linux%20to%20windows?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=67&branchName=master) |
| Remote - windows to linux | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20remote%20-%20windows%20to%20linux?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=68&branchName=master) |
|          OpenPAI          |        [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20openpai%20-%20linux?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=65&branchName=master)        |
|    Frameworkcontroller    |        [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20frameworkcontroller?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=70&branchName=master)        |
|         Kubeflow          |             [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20kubeflow?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=69&branchName=master)              |
|          Hybrid           |              [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20hybrid?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=79&branchName=master)               |
|          AzureML          |                [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20aml?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=78&branchName=master)                |

## 相关项目

针对开放性和推进最先进的技术，[微软研究院（MSR)](https://www.microsoft.com/en-us/research/group/systems-and-networking-research-group-asia/) 还发布了其他几个开源项目。

* [OpenPAI](https://github.com/Microsoft/pai)：作为开源平台，提供了完整的 AI 模型训练和资源管理能力，能轻松扩展，并支持各种规模的私有部署、云和混合环境。
* [FrameworkController](https://github.com/Microsoft/frameworkcontroller)：开源的通用 Kubernetes Pod 控制器，通过单个控制器来编排 Kubernetes 上所有类型的应用。
* [MMdnn](https://github.com/Microsoft/MMdnn)：一个完整、跨框架的解决方案，能够转换、可视化、诊断深度神经网络模型。 MMdnn 中的 "MM" 表示 model management（模型管理），而 "dnn" 是 deep neural network（深度神经网络）的缩写。
* [SPTAG](https://github.com/Microsoft/SPTAG) : Space Partition Tree And Graph (SPTAG) 是用于大规模向量的最近邻搜索场景的开源库。

我们鼓励研究人员和学生利用这些项目来加速 AI 开发和研究。

## **许可协议**

代码库遵循 [MIT 许可协议](LICENSE)
