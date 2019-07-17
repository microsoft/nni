<p align="center">
  <img src="docs/img/nni_logo.png" width="300" />
</p>

* * *

[![MIT 许可证](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE) [![生成状态](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/Microsoft.nni)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=6) [![问题](https://img.shields.io/github/issues-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen) [![Bug](https://img.shields.io/github/issues/Microsoft/nni/bug.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen+label%3Abug) [![拉取请求](https://img.shields.io/github/issues-pr-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/pulls?q=is%3Apr+is%3Aopen) [![版本](https://img.shields.io/github/release/Microsoft/nni.svg)](https://github.com/Microsoft/nni/releases) [![进入 https://gitter.im/Microsoft/nni 聊天室提问](https://badges.gitter.im/Microsoft/nni.svg)](https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![文档状态](https://readthedocs.org/projects/nni/badge/?version=latest)](https://nni.readthedocs.io/en/latest/?badge=latest)

[English](README.md)

NNI (Neural Network Intelligence) 是自动机器学习（AutoML）的工具包。 它通过多种调优的算法来搜索最好的神经网络结构和（或）超参，并支持单机、本地多机、云等不同的运行环境。

### **NNI [v0.9](https://github.com/Microsoft/nni/releases) 已发布！ &nbsp;[<img width="48" src="docs/img/release_icon.png" />](#nni-released-reminder)**

<p align="center">
  <a href="#nni-has-been-released"><img src="docs/img/overview.svg" /></a>
</p>

<div>
  <table>
    <tr align="center" valign="bottom">
      <td>
        <b>支持的框架</b> <img src="docs/img/bar.png" />
      </td>
      
      <td>
        <b>调优算法</b> <img src="docs/img/bar.png" />
      </td>
      
      <td>
        <b>训练平台</b> <img src="docs/img/bar.png" />
      </td>
    </tr></tr> 
    
    <tr valign="top">
      <td>
        <ul>
          <li>
            PyTorch
          </li>
          <li>
            TensorFlow
          </li>
          <li>
            Keras
          </li>
          <li>
            MXNet
          </li>
          <li>
            Caffe2
          </li>
          <li>
            CNTK (Python 语言)
          </li>
          <li>
            Chainer
          </li>
          <li>
            Theano
          </li>
        </ul>
      </td>
      
      <td align="left">
        <a href="docs/en_US/Tuner/BuiltinTuner.md">Tuner（调参器）</a> <br /> 
        
        <ul>
          <b style="margin-left:-20px"><font size=4 color=#800000>通用 Tuner</font></b> 
          
          <li>
            <a href="docs/en_US/Tuner/BuiltinTuner.md#Random"><font size=2.9>Random Search（随机搜索）</font></a>
          </li>
          <li>
            <a href="docs/en_US/Tuner/BuiltinTuner.md#Evolution"><font size=2.9>Naïve Evolution（进化算法）</font></a>
          </li>
          <b><font size=4 color=#800000 style="margin-left:-20px">超参 Tuner</font></b> 
          
          <li>
            <a href="docs/en_US/Tuner/BuiltinTuner.md#TPE"><font size=2.9>TPE</font></a>
          </li>
          <li>
            <a href="docs/en_US/Tuner/BuiltinTuner.md#Anneal"><font size=2.9>Anneal（退火算法）</font></a>
          </li>
          <li>
            <a href="docs/en_US/Tuner/BuiltinTuner.md#SMAC"><font size=2.9>SMAC</font></a>
          </li>
          <li>
            <a href="docs/en_US/Tuner/BuiltinTuner.md#Batch"><font size=2.9>Batch（批处理）</font></a>
          </li>
          <li>
            <a href="docs/en_US/Tuner/BuiltinTuner.md#GridSearch"><font size=2.9>Grid Search（遍历搜索）</font></a>
          </li>
          <li>
            <a href="docs/en_US/Tuner/BuiltinTuner.md#Hyperband"><font size=2.9>Hyperband</font></a>
          </li>
          <li>
            <a href="docs/en_US/Tuner/BuiltinTuner.md#MetisTuner"><font size=2.9>Metis Tuner</font></a>
          </li>
          <li>
            <a href="docs/en_US/Tuner/BuiltinTuner.md#BOHB"><font size=2.9>BOHB</font></a>
          </li>
          <li>
            <a href="docs/en_US/Tuner/BuiltinTuner.md#GPTuner"><font size=2.9>GP Tuner</font></a>
          </li>
          <b style="margin-left:-20px"><font size=4 color=#800000 style="margin-left:-20px">网络结构 Tuner</font></b> 
          
          <li>
            <a href="docs/en_US/Tuner/BuiltinTuner.md#NetworkMorphism"><font size=2.9>Network Morphism</font></a>
          </li>
          <li>
            <a href="examples/tuners/enas_nni/README.md"><font size=2.9>ENAS</font></a>
          </li>
        </ul>
        
        <a href="docs/en_US/Assessor/BuiltinAssessor.md">Assessor（评估器）</a> 
        
        <ul>
          <li>
            <a href="docs/en_US/Assessor/BuiltinAssessor.md#Medianstop"><font size=2.9>Median Stop（中位数终止）</font></a>
          </li>
          <li>
            <a href="docs/en_US/Assessor/BuiltinAssessor.md#Curvefitting"><font size=2.9>Curve Fitting（曲线拟合）</font></a>
          </li>
        </ul>
      </td>
      
      <td>
        <ul>
          <li>
            <a href="docs/en_US/TrainingService/LocalMode.md">本机</a>
          </li>
          <li>
            <a href="docs/en_US/TrainingService/RemoteMachineMode.md">远程计算机</a>
          </li>
          <li>
            <b>基于 Kubernetes 的平台</b>
          </li>
          <ul>
            <li>
              <a href="docs/en_US/TrainingService/PaiMode.md">OpenPAI</a>
            </li>
            <li>
              <a href="docs/en_US/TrainingService/KubeflowMode.md">Kubeflow</a>
            </li>
            <li>
              <a href="docs/en_US/TrainingService/FrameworkControllerMode.md">基于 Kubernetes（AKS 等）的 FrameworkController</a>
            </li>
          </ul>
        </ul>
      </td>
    </tr>
    
    <tr align="center" valign="bottom">
      <td style="border-top:#FF0000 solid 0px;">
        <b>参考文档</b> <img src="docs/img/bar.png" />
      </td>
      
      <td style="border-top:#FF0000 solid 0px;">
        <b>参考文档</b> <img src="docs/img/bar.png" />
      </td>
      
      <td style="border-top:#FF0000 solid 0px;">
        <b>参考文档</b> <img src="docs/img/bar.png" />
      </td>
    </tr>
    
    <tr valign="top">
      <td style="border-top:#FF0000 solid 0px;">
        <ul>
          <li>
            <a href="docs/en_US/sdk_reference.rst">Python API</a>
          </li>
          <li>
            <a href="docs/en_US/Tutorial/AnnotationSpec.md">NNI Annotation</a>
          </li>
          <li>
            <a href="docs/en_US/TrialExample/Trials.md#nni-python-annotation">Annotation 教程</a>
          </li>
        </ul>
      </td>
      
      <td style="border-top:#FF0000 solid 0px;">
        <ul>
          <li>
            <a href="docs/en_US/tuners.rst">尝试不同的 Tuner</a>
          </li>
          <li>
            <a href="docs/en_US/assessors.rst">尝试不同的 Assessor</a>
          </li>
          <li>
            <a href="docs/en_US/Tuner/CustomizeTuner.md">实现自定义 Tuner</a>
          </li>
          <li>
            <a href="docs/en_US/Tuner/CustomizeAdvisor.md">实现自定义 Advisor</a>
          </li>
          <li>
            <a href="docs/en_US/Assessor/CustomizeAssessor.md">实现自定义 Assessor </a>
          </li>
          <li>
            <a href="docs/en_US/CommunitySharings/HpoComparision.md">超参优化算法的对比</a>
          </li>
          <li>
            <a href="docs/en_US/CommunitySharings/NasComparision.md">网络结构算法的对比</a>
          </li>
          <li>
            <a href="docs/en_US/CommunitySharings/RecommendersSvd.md">在 NNI 上自动调优 SVD</a>
          </li>
        </ul>
      </td>
      
      <td style="border-top:#FF0000 solid 0px;">
        <ul>
          <li>
            <a href="docs/en_US/TrainingService/HowToImplementTrainingService.md">实现 NNI 训练平台</a>
          </li>
          <li>
            <a href="docs/en_US/TrainingService/LocalMode.md">在本机运行 Experiment</a>
          </li>
          <li>
            <a href="docs/en_US/TrainingService/KubeflowMode.md">在 Kubeflow 上运行 Experiment</a>
          </li>
          <li>
            <a href="docs/en_US/TrainingService/PaiMode.md">在 OpenPAI 上运行 Experiment</a>
          </li>
          <li>
            <a href="docs/en_US/TrainingService/RemoteMachineMode.md">在多机上运行 Experiment</a>
          </li>
        </ul>
      </td></tbody> </table> </div> 
      
      <h2>
        <strong>使用场景</strong>
      </h2>
      
      <ul>
        <li>
          在本机尝试使用不同的自动机器学习（AutoML）算法来训练模型。
        </li>
        <li>
          在分布式环境中加速自动机器学习（如：远程 GPU 工作站和云服务器）。
        </li>
        <li>
          定制自动机器学习算法，或比较不同的自动机器学习算法。
        </li>
        <li>
          在机器学习平台中支持自动机器学习。
        </li>
      </ul>
      
      <h2>
        相关项目
      </h2>
      
      <p>
        以开发和先进技术为目标，<a href="https://www.microsoft.com/en-us/research/group/systems-research-group-asia/">Microsoft Research (MSR)</a> 发布了一些开源项目。
      </p>
      
      <ul>
        <li>
          <a href="https://github.com/Microsoft/pai">OpenPAI</a>：作为开源平台，提供了完整的 AI 模型训练和资源管理能力，能轻松扩展，并支持各种规模的私有部署、云和混合环境。
        </li>
        <li>
          <a href="https://github.com/Microsoft/frameworkcontroller">FrameworkController</a>：开源的通用 Kubernetes Pod 控制器，通过单个控制器来编排 Kubernetes 上所有类型的应用。
        </li>
        <li>
          <a href="https://github.com/Microsoft/MMdnn">MMdnn</a>：一个完整、跨框架的解决方案，能够转换、可视化、诊断深度神经网络模型。 MMdnn 中的 "MM" 表示model management（模型管理），而 "dnn" 是 deep neural network（深度神经网络）的缩写。
        </li>
        <li>
          <a href="https://github.com/Microsoft/SPTAG">SPTAG</a> : Space Partition Tree And Graph (SPTAG) 是用于大规模向量的最近邻搜索场景的开源库。
        </li>
      </ul>
      
      <p>
        我们鼓励研究人员和学生利用这些项目来加速 AI 开发和研究。
      </p>
      
      <h2>
        <strong>安装和验证</strong>
      </h2>
      
      <p>
        <strong>通过 pip 命令安装</strong>
      </p>
      
      <ul>
        <li>
          当前支持 Linux，MacOS 和 Windows（本机，远程，OpenPAI 模式），在 Ubuntu 16.04 或更高版本，MacOS 10.14.1 以及 Windows 10.1809 上进行了测试。 在 <code>python &gt;= 3.5</code> 的环境中，只需要运行 <code>pip install</code> 即可完成安装。
        </li>
      </ul>
      
      <p>
        Linux 和 macOS
      </p>
      
      <pre><code class="bash">python3 -m pip install --upgrade nni
</code></pre>
      
      <p>
        Windows
      </p>
      
      <pre><code class="bash">python -m pip install --upgrade nni
</code></pre>
      
      <p>
        注意：
      </p>
      
      <ul>
        <li>
          如果需要将 NNI 安装到自己的 home 目录中，可使用 <code>--user</code>，这样也不需要任何特殊权限。
        </li>
        <li>
          目前，Windows 上的 NNI 支持本机，远程和 OpenPAI 模式。 强烈推荐使用 Anaconda 或 Miniconda 在 Windows 上安装 NNI。
        </li>
        <li>
          如果遇到如<code>Segmentation fault</code> 这样的任何错误请参考<a href="docs/zh_CN/Tutorial/FAQ.md">常见问题</a>。
        </li>
      </ul>
      
      <p>
        <strong>通过源代码安装</strong>
      </p>
      
      <ul>
        <li>
          当前支持 Linux（Ubuntu 16.04 或更高版本），MacOS（10.14.1）以及 Windows 10（1809 版）。
        </li>
      </ul>
      
      <p>
        Linux 和 macOS
      </p>
      
      <ul>
        <li>
          在 <code>python &gt;= 3.5</code> 的环境中运行命令： <code>git</code> 和 <code>wget</code>，确保安装了这两个组件。
        </li>
      </ul>
      
      <pre><code class="bash">    git clone -b v0.9 https://github.com/Microsoft/nni.git
    cd nni
    source install.sh
</code></pre>
      
      <p>
        Windows
      </p>
      
      <ul>
        <li>
          在 <code>python &gt;=3.5</code> 的环境中运行命令： <code>git</code> 和 <code>PowerShell</code>，确保安装了这两个组件。
        </li>
      </ul>
      
      <pre><code class="bash">  git clone -b v0.9 https://github.com/Microsoft/nni.git
  cd nni
  powershell -ExecutionPolicy Bypass -file install.ps1
</code></pre>
      
      <p>
        参考<a href="docs/zh_CN/Tutorial/Installation.md">安装 NNI</a> 了解系统需求。
      </p>
      
      <p>
        Windows 上参考 <a href="docs/zh_CN/Tutorial/NniOnWindows.md">Windows 上使用 NNI</a>。
      </p>
      
      <p>
        <strong>验证安装</strong>
      </p>
      
      <p>
        以下示例 Experiment 依赖于 TensorFlow 。 在运行前确保安装了 <strong>TensorFlow</strong>。
      </p>
      
      <ul>
        <li>
          通过克隆源代码下载示例。
        </li>
      </ul>
      
      <pre><code class="bash">    git clone -b v0.9 https://github.com/Microsoft/nni.git
</code></pre>
      
      <p>
        Linux 和 macOS
      </p>
      
      <ul>
        <li>
          运行 MNIST 示例。
        </li>
      </ul>
      
      <pre><code class="bash">    nnictl create --config nni/examples/trials/mnist/config.yml
</code></pre>
      
      <p>
        Windows
      </p>
      
      <ul>
        <li>
          运行 MNIST 示例。
        </li>
      </ul>
      
      <pre><code class="bash">    nnictl create --config nni\examples\trials\mnist\config_windows.yml
</code></pre>
      
      <ul>
        <li>
          在命令行中等待输出 <code>INFO: Successfully started experiment!</code>。 此消息表明 Experiment 已成功启动。 通过命令行输出的 <code>Web UI url</code> 来访问 Experiment 的界面。
        </li>
      </ul>
      
      <pre><code class="text">INFO: Starting restful server...
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
</code></pre>
      
      <ul>
        <li>
          在浏览器中打开 <code>Web UI url</code>，可看到下图的 Experiment 详细信息，以及所有的 Trial 任务。 查看<a href="docs/zh_CN/Tutorial/WebUI.md">这里</a>的更多页面。
        </li>
      </ul>
      
      <table style="border: none">
        <th>
          <img src="./docs/img/webui_overview_page.png" alt="绘图" width="395" />
        </th>
        
        <th>
          <img src="./docs/img/webui_trialdetail_page.png" alt="绘图" width="410" />
        </th>
      </table>
      
      <h2>
        <strong>文档</strong>
      </h2>
      
      <p>
        主要文档都可以在<a href="https://nni.readthedocs.io/cn/latest/Overview.html">这里</a>找到，文档均从本代码库生成。<br /> 点击阅读：
      </p>
      
      <ul>
        <li>
          <a href="docs/zh_CN/Overview.md">NNI 概述</a>
        </li>
        <li>
          <a href="docs/en_US/Tutorial/QuickStart.md">快速入门</a>
        </li>
        <li>
          <a href="docs/en_US/Tutorial/Contributing.md">贡献</a>
        </li>
        <li>
          <a href="docs/en_US/examples.rst">示例</a>
        </li>
        <li>
          <a href="docs/en_US/reference.rst">参考</a>
        </li>
        <li>
          <a href="docs/en_US/Tutorial/WebUI.md">Web 界面教程</a>
        </li>
      </ul>
      
      <h2>
        <strong>入门</strong>
      </h2>
      
      <ul>
        <li>
          <a href="docs/en_US/Tutorial/Installation.md">安装 NNI</a>
        </li>
        <li>
          <a href="docs/en_US/Tutorial/Nnictl.md">使用命令行工具 nnictl</a>
        </li>
        <li>
          <a href="docs/en_US/Tutorial/WebUI.md">使用 NNIBoard</a>
        </li>
        <li>
          <a href="docs/en_US/Tutorial/SearchSpaceSpec.md">如何定义搜索空间</a>
        </li>
        <li>
          <a href="docs/en_US/TrialExample/Trials.md">如何实现 Trial 代码</a>
        </li>
        <li>
          <a href="docs/en_US/Tuner/BuiltinTuner.md">如何选择 Tuner、搜索算法</a>
        </li>
        <li>
          <a href="docs/en_US/Tutorial/ExperimentConfig.md">配置 Experiment</a>
        </li>
        <li>
          <a href="docs/en_US/TrialExample/Trials.md#nni-python-annotation">如何使用 Annotation</a>
        </li>
      </ul>
      
      <h2>
        <strong>教程</strong>
      </h2>
      
      <ul>
        <li>
          <a href="docs/en_US/PaiMode.md">在 OpenPAI 上运行 Experiment</a>
        </li>
        <li>
          <a href="docs/en_US/KubeflowMode.md">在 Kubeflow 上运行 Experiment</a>
        </li>
        <li>
          <a href="docs/en_US/LocalMode.md">在本机运行 Experiment (支持多 GPU 卡)</a>
        </li>
        <li>
          <a href="docs/en_US/RemoteMachineMode.md">在多机上运行 Experiment</a>
        </li>
        <li>
          <a href="docs/zh_CN/tuners.rst">尝试不同的 Tuner</a>
        </li>
        <li>
          <a href="docs/zh_CN/assessors.rst">尝试不同的 Assessor</a>
        </li>
        <li>
          <a href="docs/en_US/Tuner/CustomizeTuner.md">实现自定义 Tuner</a>
        </li>
        <li>
          <a href="docs/zh_CN/CustomizeAssessor.md">实现自定义 Assessor</a>
        </li>
        <li>
          <a href="examples/trials/ga_squad/README_zh_CN.md">使用进化算法为阅读理解任务找到好模型</a>
        </li>
      </ul>
      
      <h2>
        <strong>贡献</strong>
      </h2>
      
      <p>
        非常欢迎通过各种方式参与此项目，例如：
      </p>
      
      <ul>
        <li>
          审查<a href="https://github.com/microsoft/nni/pulls">源代码改动</a>
        </li>
        <li>
          审查<a href="https://github.com/microsoft/nni/tree/master/docs">文档</a>中从拼写错误到新内容的任何内容，并提交拉取请求。
        </li>
        <li>
          找到标有 <a href="https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">'good first issue'</a> 或 <a href="https://github.com/microsoft/nni/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22">'help-wanted'</a> 标签的 Issue。这些都是简单的 Issue，新的贡献者可以从这些问题开始。
        </li>
      </ul>
      
      <p>
        在提交代码前，需要遵循以下的简单准则：
      </p>
      
      <ul>
        <li>
          <a href="docs/en_US/Tutorial/HowToDebug.md">如何调试</a>
        </li>
        <li>
          <a href="docs/en_US/Tutorial/Contributing.md">代码风格和命名约定</a>
        </li>
        <li>
          如何设置 <a href="docs/zh_CN/Tutorial/SetupNniDeveloperEnvironment.md">NNI 开发环境</a>
        </li>
        <li>
          查看<a href="docs/en_US/Tutorial/Contributing.md">贡献说明</a>并熟悉 NNI 的代码贡献指南
        </li>
      </ul>
      
      <h2>
        <strong>外部代码库</strong>
      </h2>
      
      <p>
        下面是一些贡献者为 NNI 提供的使用示例 谢谢可爱的贡献者！ 欢迎越来越多的人加入我们！
      </p>
      
      <ul>
        <li>
          在 NNI 中运行 <a href="examples/tuners/enas_nni/README_zh_CN.md">ENAS</a>
        </li>
        <li>
          在 NNI 中运行 <a href="examples/trials/nas_cifar10/README_zh_CN.md">神经网络架构结构搜索</a>
        </li>
      </ul>
      
      <h2>
        <strong>反馈</strong>
      </h2>
      
      <ul>
        <li>
          <a href="https://github.com/microsoft/nni/issues/new/choose">报告 Bug</a>。<br />
        </li>
        <li>
          <a href="https://github.com/microsoft/nni/issues/new/choose">请求新功能</a>.
        </li>
        <li>
          在 <a href="https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge">Gitter</a> 中参与讨论
        </li>
        <li>
          Ask a question with NNI tags on <a href="https://stackoverflow.com/questions/tagged/nni?sort=Newest&edited=true">Stack Overflow</a>or <a href="https://github.com/microsoft/nni/issues/new/choose">file an issue</a>on GitHub.
        </li>
        <li>
          We are in construction of the instruction for <a href="docs/en_US/Tutorial/HowToDebug.md">How to Debug</a>, you are also welcome to contribute questions or suggestions on this area.
        </li>
      </ul>
      
      <h2>
        <strong>License</strong>
      </h2>
      
      <p>
        The entire codebase is under <a href="LICENSE">MIT license</a>
      </p>