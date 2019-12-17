<p align="center">
<img src="docs/img/nni_logo.png" width="300"/>
</p>

* * *

[![MIT 许可证](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE) [![生成状态](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/Microsoft.nni)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=6) [![问题](https://img.shields.io/github/issues-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen) [![Bug](https://img.shields.io/github/issues/Microsoft/nni/bug.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen+label%3Abug) [![拉取请求](https://img.shields.io/github/issues-pr-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/pulls?q=is%3Apr+is%3Aopen) [![版本](https://img.shields.io/github/release/Microsoft/nni.svg)](https://github.com/Microsoft/nni/releases) [![进入 https://gitter.im/Microsoft/nni 聊天室提问](https://badges.gitter.im/Microsoft/nni.svg)](https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![文档状态](https://readthedocs.org/projects/nni/badge/?version=latest)](https://nni.readthedocs.io/zh/latest/?badge=latest)

[English](README.md)

**NNI (Neural Network Intelligence)** 是一个轻量但强大的工具包，帮助用户**自动**的进行[特征工程](docs/en_US/FeatureEngineering/Overview.md)，[神经网络架构搜索](docs/en_US/NAS/Overview.md)，[超参调优](docs/en_US/Tuner/BuiltinTuner.md)以及[模型压缩](docs/en_US/Compressor/Overview.md)。

NNI 管理自动机器学习 (AutoML) 的 Experiment，**调度运行**由调优算法生成的 Trial 任务来找到最好的神经网络架构和/或超参，支持**各种训练环境**，如[本机](docs/en_US/TrainingService/LocalMode.md)，[远程服务器](docs/en_US/TrainingService/RemoteMachineMode.md)，[OpenPAI](docs/en_US/TrainingService/PaiMode.md)，[Kubeflow](docs/en_US/TrainingService/KubeflowMode.md)，[基于 K8S 的 FrameworkController（如，AKS 等)](docs/en_US/TrainingService/FrameworkControllerMode.md)，以及其它云服务。

## **使用场景**

* 想要在自己的代码、模型中试验**不同的自动机器学习算法**。
* 想要在**不同的环境中**加速运行自动机器学习。
* 想要更容易**实现或试验新的自动机器学习算法**的研究员或数据科学家，包括：超参调优算法，神经网络搜索算法以及模型压缩算法。
* 在机器学习平台中**支持自动机器学习**。

### **NNI v1.2 已发布！ &nbsp;[<img width="48" src="docs/img/release_icon.png" />](#nni-released-reminder)**

## **NNI 功能一览**

NNI provides CommandLine Tool as well as an user friendly WebUI to manage training experiements. With the extensible API, you can customize your own AutoML algorithms and training services. To make it easy for new users, NNI also provides a set of build-in stat-of-the-art AutoML algorithms and out of box support for popular training platforms.

Within the following table, we summarized the current NNI capabilities, we are gradually adding new capabilities and we'd love to have your contribution.

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
          <a href="docs/zh_CN/SupportedFramework_Library.md">更多...</a><br/>
        </ul>
        </ul>
      <ul>
        <li><b>支持的库</b></li>
          <ul>
           <li>Scikit-learn</li>
           <li>XGBoost</li>
           <li>LightGBM</li>
           <a href="docs/zh_CN/SupportedFramework_Library.md">更多...</a><br/>
          </ul>
      </ul>
        <ul>
        <li><b>示例</b></li>
         <ul>
           <li><a href="examples/trials/mnist-pytorch">MNIST-pytorch</li></a>
           <li><a href="examples/trials/mnist-tfv1">MNIST-tensorflow</li></a>
           <li><a href="examples/trials/mnist-keras">MNIST-keras</li></a>
           <li><a href="docs/zh_CN/TrialExample/GbdtExample.md">Auto-gbdt</a></li>
           <li><a href="docs/zh_CN/TrialExample/Cifar10Examples.md">Cifar10-pytorch</li></a>
           <li><a href="docs/zh_CN/TrialExample/SklearnExamples.md">Scikit-learn</a></li>
              <a href="docs/zh_CN/SupportedFramework_Library.md">更多...</a><br/>
          </ul>
        </ul>
      </td>
      <td align="left" >
        <a href="docs/zh_CN/Tuner/BuiltinTuner.md">Hyperparameter Tuning</a>
        <ul>
          <b>Exhaustive search</b>
          <ul>
            <li><a href="docs/en_US/Tuner/BuiltinTuner.md#Random">Random Search</a></li>
            <li><a href="docs/en_US/Tuner/BuiltinTuner.md#GridSearch">Grid Search</a></li>
            <li><a href="docs/en_US/Tuner/BuiltinTuner.md#Batch">Batch</a></li>
            </ul>
          <b>Heuristic search</b>
          <ul>
            <li><a href="docs/en_US/Tuner/BuiltinTuner.md#Evolution">Naïve Evolution</a></li>
            <li><a href="docs/en_US/Tuner/BuiltinTuner.md#Anneal">Anneal</a></li>  
            <li><a href="docs/en_US/Tuner/BuiltinTuner.md#Hyperband">Hyperband</a></li>
          </ul>
          <b>Bayesian optimization</b>
            <ul>
              <li><a href="docs/en_US/Tuner/BuiltinTuner.md#BOHB">BOHB</a></li>  
              <li><a href="docs/en_US/Tuner/BuiltinTuner.md#TPE">TPE</a></li>
            <li><a href="docs/en_US/Tuner/BuiltinTuner.md#SMAC">SMAC</a></li> 
            <li><a href="docs/en_US/Tuner/BuiltinTuner.md#MetisTuner">Metis Tuner</a></li>
            <li><a href="docs/en_US/Tuner/BuiltinTuner.md#GPTuner">GP Tuner</a> </li>
            </ul>  
          <b>RL Based</b>
          <ul>
            <li><a href="docs/en_US/Tuner/BuiltinTuner.md#PPOTuner">PPO Tuner</a> </li>
          </ul>
        </ul>
          <a href="docs/en_US/NAS/Overview.md">Neural Architecture Search</a>
          <ul>                        
            <ul>
              <li><a href="docs/en_US/NAS/Overview.md#enas">ENAS</a></li>
              <li><a href="docs/en_US/NAS/Overview.md#darts">DARTS</a></li>
              <li><a href="docs/en_US/NAS/Overview.md#p-darts">P-DARTS</a></li>
              <li><a href="docs/en_US/Tuner/BuiltinTuner.md#NetworkMorphism">Network Morphism</a> </li>
            </ul>    
          </ul>
          <a href="docs/en_US/Compressor/Overview.md">Model Compression</a>
          <ul>
            <b>Pruning</b>
            <ul>
              <li><a href="docs/en_US/Compressor/Pruner.md#agp-pruner">AGP Pruner</a></li>
              <li><a href="docs/en_US/Compressor/Pruner.md#slim-pruner">Slim Pruner</a></li>
              <li><a href="docs/en_US/Compressor/Pruner.md#fpgm-pruner">FPGM Pruner</a></li>
            </ul>
            <b>Quantization</b>
            <ul>
              <li><a href="docs/en_US/Compressor/Quantizer.md#qat-quantizer">QAT Quantizer</a></li>
              <li><a href="docs/en_US/Compressor/Quantizer.md#dorefa-quantizer">DoReFa Quantizer</a></li>
            </ul>
          </ul>
          <a href="docs/en_US/FeatureEngineering/Overview.md">Feature Engineering (Beta)</a>
          <ul>
          <li><a href="docs/en_US/FeatureEngineering/GradientFeatureSelector.md">GradientFeatureSelector</a></li>
          <li><a href="docs/en_US/FeatureEngineering/GBDTSelector.md">GBDTSelector</a></li>
          </ul>
          <a href="docs/en_US/Assessor/BuiltinAssessor.md">Early Stop Algorithms</a>
          <ul>
          <li><a href="docs/en_US/Assessor/BuiltinAssessor.md#Medianstop">Median Stop</a></li>
          <li><a href="docs/en_US/Assessor/BuiltinAssessor.md#Curvefitting">Curve Fitting</a></li>   
          </ul>
      </td>
      <td>
      <ul>
        <li><a href="docs/zh_CN/TrainingService/LocalMode.md">本机</a></li>
        <li><a href="docs/zh_CN/TrainingService/RemoteMachineMode.md">远程计算机</a></li>
        <li><b>基于 Kubernetes 的平台</b></li>
            <ul><li><a href="docs/zh_CN/TrainingService/PaiMode.md">OpenPAI</a></li>
            <li><a href="docs/zh_CN/TrainingService/KubeflowMode.md">Kubeflow</a></li>
            <li><a href="docs/zh_CN/TrainingService/FrameworkControllerMode.md">基于 Kubernetes（AKS 等）的 FrameworkController</a></li>
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
        <li><a href="docs/zh_CN/sdk_reference.rst">Python API</a></li>
        <li><a href="docs/zh_CN/Tutorial/AnnotationSpec.md">NNI Annotation</a></li>
         <li><a href="docs/zh_CN/Tutorial/Installation.md">支持的操作系统</a></li>
      </ul>
      </td>
       <td style="border-top:#FF0000 solid 0px;">
      <ul>
        <li><a href="docs/zh_CN/Tuner/CustomizeTuner.md">自定义 Tuner</a></li>
        <li><a href="docs/zh_CN/Assessor/CustomizeAssessor.md">自定义 Assessor</a></li>
      </ul>
      </td>
        <td style="border-top:#FF0000 solid 0px;">
      <ul>
        <li><a href="docs/zh_CN/TrainingService/SupportTrainingService.md">支持训练平台</li>
        <li><a href="docs/zh_CN/TrainingService/HowToImplementTrainingService.md">实现训练平台</a></li>
      </ul>
      </td>     
    </tr> 
  </tbody>
</table>

## **安装和验证**

**Install through pip**

* We support Linux, MacOS and Windows (local, remote and pai mode) in current stage, Ubuntu 16.04 or higher, MacOS 10.14.1 along with Windows 10.1809 are tested and supported. Simply run the following `pip install` in an environment that has `python >= 3.5`.

Linux and MacOS

```bash
python3 -m pip install --upgrade nni
```

Windows

```bash
python -m pip install --upgrade nni
```

Note:

* `--user` can be added if you want to install NNI in your home directory, which does not require any special privileges.
* Currently NNI on Windows support local, remote and pai mode. Anaconda or Miniconda is highly recommended to install NNI on Windows.
* If there is any error like `Segmentation fault`, please refer to [FAQ](docs/en_US/Tutorial/FAQ.md)

**Install through source code**

* We support Linux (Ubuntu 16.04 or higher), MacOS (10.14.1) and Windows (10.1809) in our current stage.

Linux and MacOS

* Run the following commands in an environment that has `python >= 3.5`, `git` and `wget`.

```bash
    git clone -b v1.2 https://github.com/Microsoft/nni.git
    cd nni
    source install.sh
```

Windows

* Run the following commands in an environment that has `python >=3.5`, `git` and `PowerShell`

```bash
  git clone -b v1.2 https://github.com/Microsoft/nni.git
  cd nni
  powershell -ExecutionPolicy Bypass -file install.ps1
```

For the system requirements of NNI, please refer to [Install NNI](docs/en_US/Tutorial/Installation.md)

For NNI on Windows, please refer to [NNI on Windows](docs/en_US/Tutorial/NniOnWindows.md)

**Verify install**

The following example is an experiment built on TensorFlow. Make sure you have **TensorFlow 1.x installed** before running it. Note that **currently Tensorflow 2.0 is NOT supported**.

* Download the examples via clone the source code.

```bash
    git clone -b v1.2 https://github.com/Microsoft/nni.git
```

Linux and MacOS

* Run the MNIST example.

```bash
    nnictl create --config nni/examples/trials/mnist-tfv1/config.yml
```

Windows

* 运行 MNIST 示例。

```bash
    nnictl create --config nni\examples\trials\mnist-tfv1\config_windows.yml
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

* Open the `Web UI url` in your browser, you can view detail information of the experiment and all the submitted trial jobs as shown below. [Here](docs/en_US/Tutorial/WebUI.md) are more Web UI pages.

<table style="border: none">
    <th><img src="./docs/img/webui_overview_page.png" alt="drawing" width="395"/></th>
    <th><img src="./docs/img/webui_trialdetail_page.png" alt="drawing" width="410"/></th>
</table>

## **文档**

* To learn about what's NNI, read the [NNI Overview](https://nni.readthedocs.io/en/latest/Overview.html). 
* To get yourself familiar with how to use NNI, read the [documentation](https://nni.readthedocs.io/en/latest/index.html). 
* To get started and install NNI on your system, please refer to [Install NNI](docs/en_US/Tutorial/Installation.md).

## **Contributing**

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the Code of [Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact opencode@microsoft.com with any additional questions or comments.

After getting familiar with contribution agreements, you are ready to create your first PR =), follow the NNI developer tutorials to get start:

* We recommend new contributors to start with ['good first issue'](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or ['help-wanted'](https://github.com/microsoft/nni/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22), these issues are simple and easy to start.
* [NNI developer environment installation tutorial](docs/en_US/Tutorial/SetupNniDeveloperEnvironment.md)
* [How to debug](docs/en_US/Tutorial/HowToDebug.md)
* [Customize your own Tuner](docs/en_US/Tuner/CustomizeTuner.md)
* [Implement customized TrainingService](docs/en_US/TrainingService/HowToImplementTrainingService.md)
* [Implement a new NAS trainer on NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/NAS/NasInterface.md#implement-a-new-nas-trainer-on-nni)
* [Customize your own Advisor](docs/en_US/Tuner/CustomizeAdvisor.md)

## **External Repositories and References**

With authors' permission, we listed a set of NNI usage examples and relevant articles.

* ### **External Repositories**
   
   * Run [ENAS](examples/tuners/enas_nni/README.md) with NNI
   * Run [Neural Network Architecture Search](examples/trials/nas_cifar10/README.md) with NNI 
   * [Automatic Feature Engineering](examples/feature_engineering/auto-feature-engineering/README.md) with NNI 
   * [Hyperparameter Tuning for Matrix Factorization](https://github.com/microsoft/recommenders/blob/master/notebooks/04_model_select_and_optimize/nni_surprise_svd.ipynb) with NNI
   * [scikit-nni](https://github.com/ksachdeva/scikit-nni) Hyper-parameter search for scikit-learn pipelines using NNI
* ### **Relevant Articles**
   
   * [Hyper Parameter Optimization Comparison](docs/en_US/CommunitySharings/HpoComparision.md)
   * [Neural Architecture Search Comparison](docs/en_US/CommunitySharings/NasComparision.md)
   * [Parallelizing a Sequential Algorithm TPE](docs/en_US/CommunitySharings/ParallelizingTpeSearch.md)
   * [Automatically tuning SVD with NNI](docs/en_US/CommunitySharings/RecommendersSvd.md)
   * [Automatically tuning SPTAG with NNI](docs/en_US/CommunitySharings/SptagAutoTune.md)
   * [Find thy hyper-parameters for scikit-learn pipelines using Microsoft NNI](https://towardsdatascience.com/find-thy-hyper-parameters-for-scikit-learn-pipelines-using-microsoft-nni-f1015b1224c1)
   * **Blog (in Chinese)** - [AutoML tools (Advisor, NNI and Google Vizier) comparison](http://gaocegege.com/Blog/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/katib-new#%E6%80%BB%E7%BB%93%E4%B8%8E%E5%88%86%E6%9E%90) by [@gaocegege](https://github.com/gaocegege) - 总结与分析 section of design and implementation of kubeflow/katib

## **Feedback**

* Discuss on the NNI [Gitter](https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) in NNI.
* [File an issue](https://github.com/microsoft/nni/issues/new/choose) on GitHub.
* Ask a question with NNI tags on [Stack Overflow](https://stackoverflow.com/questions/tagged/nni?sort=Newest&edited=true).

## Related Projects

Targeting at openness and advancing state-of-art technology, [Microsoft Research (MSR)](https://www.microsoft.com/en-us/research/group/systems-research-group-asia/) had also released few other open source projects.

* [OpenPAI](https://github.com/Microsoft/pai) : an open source platform that provides complete AI model training and resource management capabilities, it is easy to extend and supports on-premise, cloud and hybrid environments in various scale.
* [FrameworkController](https://github.com/Microsoft/frameworkcontroller) : an open source general-purpose Kubernetes Pod Controller that orchestrate all kinds of applications on Kubernetes by a single controller.
* [MMdnn](https://github.com/Microsoft/MMdnn) : A comprehensive, cross-framework solution to convert, visualize and diagnose deep neural network models. The "MM" in MMdnn stands for model management and "dnn" is an acronym for deep neural network.
* [SPTAG](https://github.com/Microsoft/SPTAG) : Space Partition Tree And Graph (SPTAG) is an open source library for large scale vector approximate nearest neighbor search scenario.

We encourage researchers and students leverage these projects to accelerate the AI development and research.

## **License**

The entire codebase is under [MIT license](LICENSE)