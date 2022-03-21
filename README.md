<p align="center">
<img src="docs/img/nni_logo.png" width="300"/>
</p>

[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
[![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/full%20test%20-%20linux?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=62&branchName=master)
[![Issues](https://img.shields.io/github/issues-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen)
[![Bugs](https://img.shields.io/github/issues/Microsoft/nni/bug.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen+label%3Abug)
[![Pull Requests](https://img.shields.io/github/issues-pr-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/pulls?q=is%3Apr+is%3Aopen)
[![Version](https://img.shields.io/github/release/Microsoft/nni.svg)](https://github.com/Microsoft/nni/releases) [![Join the chat at https://gitter.im/Microsoft/nni](https://badges.gitter.im/Microsoft/nni.svg)](https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Documentation Status](https://readthedocs.org/projects/nni/badge/?version=stable)](https://nni.readthedocs.io/en/stable/?badge=stable)

[NNI Doc](https://nni.readthedocs.io/) | [简体中文](README_zh_CN.md)

**NNI (Neural Network Intelligence)** is a lightweight but powerful toolkit to help users **automate** <a href="https://nni.readthedocs.io/en/stable/FeatureEngineering/Overview.html">Feature Engineering</a>, <a href="https://nni.readthedocs.io/en/stable/NAS/Overview.html">Neural Architecture Search</a>, <a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html">Hyperparameter Tuning</a> and <a href="https://nni.readthedocs.io/en/stable/Compression/Overview.html">Model Compression</a>.

The tool manages automated machine learning (AutoML) experiments, **dispatches and runs** experiments' trial jobs generated by tuning algorithms to search the best neural architecture and/or hyper-parameters in **different training environments** like <a href="https://nni.readthedocs.io/en/stable/TrainingService/LocalMode.html">Local Machine</a>, <a href="https://nni.readthedocs.io/en/stable/TrainingService/RemoteMachineMode.html">Remote Servers</a>, <a href="https://nni.readthedocs.io/en/stable/TrainingService/PaiMode.html">OpenPAI</a>, <a href="https://nni.readthedocs.io/en/stable/TrainingService/KubeflowMode.html">Kubeflow</a>, <a href="https://nni.readthedocs.io/en/stable/TrainingService/FrameworkControllerMode.html">FrameworkController on K8S (AKS etc.)</a>, <a href="https://nni.readthedocs.io/en/stable/TrainingService/DLTSMode.html">DLWorkspace (aka. DLTS)</a>, <a href="https://nni.readthedocs.io/en/stable/TrainingService/AMLMode.html">AML (Azure Machine Learning)</a>, <a href="https://nni.readthedocs.io/en/stable/TrainingService/AdaptDLMode.html">AdaptDL (aka. ADL)</a> , other cloud options and even <a href="https://nni.readthedocs.io/en/stable/TrainingService/HybridMode.html">Hybrid mode</a>.

## **Who should consider using NNI**

* Those who want to **try different AutoML algorithms** in their training code/model.
* Those who want to run AutoML trial jobs **in different environments** to speed up search.
* Researchers and data scientists who want to easily **implement and experiment new AutoML algorithms**, may it be: hyperparameter tuning algorithm, neural architect search algorithm or model compression algorithm.
* ML Platform owners who want to **support AutoML in their platform**.

## **What's NEW!** &nbsp;<a href="#nni-released-reminder"><img width="48" src="docs/img/release_icon.png"></a>

* **New release**: [v2.6 is available](https://github.com/microsoft/nni/releases/tag/v2.6) - _released on Jan-19-2022_
* **New demo available**: [Youtube entry](https://www.youtube.com/channel/UCKcafm6861B2mnYhPbZHavw) | [Bilibili 入口](https://space.bilibili.com/1649051673) - _last updated on May-26-2021_
* **New webinar**: [Introducing Retiarii: A deep learning exploratory-training framework on NNI](https://note.microsoft.com/MSR-Webinar-Retiarii-Registration-Live.html) - _scheduled on June-24-2021_
* **New community channel**: [Discussions](https://github.com/microsoft/nni/discussions)
* **New emoticons release**: [nnSpider](./docs/source/Tutorial/NNSpider.md)
<p align="center">
  <a href="#nni-spider"><img width="100%" src="docs/img/emoicons/home.svg" /></a>
</p>

## **NNI capabilities in a glance**

NNI provides CommandLine Tool as well as an user friendly WebUI to manage training experiments. With the extensible API, you can customize your own AutoML algorithms and training services. To make it easy for new users, NNI also provides a set of build-in state-of-the-art AutoML algorithms and out of box support for popular training platforms.

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
        <b>Frameworks & Libraries</b>
        <img src="docs/img/bar.png"/>
      </td>
      <td>
        <b>Algorithms</b>
        <img src="docs/img/bar.png"/>
      </td>
      <td>
        <b>Training Services</b>
        <img src="docs/img/bar.png"/>
      </td>
    </tr>
    </tr>
    <tr valign="top">
    <td align="center" valign="middle">
    <b>Built-in</b>
      </td>
      <td>
      <ul><li><b>Supported Frameworks</b></li>
        <ul>
          <li>PyTorch</li>
          <li>Keras</li>
          <li>TensorFlow</li>
          <li>MXNet</li>
          <li>Caffe2</li>
          <a href="https://nni.readthedocs.io/en/stable/SupportedFramework_Library.html">More...</a><br/>
        </ul>
        </ul>
      <ul>
        <li><b>Supported Libraries</b></li>
          <ul>
           <li>Scikit-learn</li>
           <li>XGBoost</li>
           <li>LightGBM</li>
           <a href="https://nni.readthedocs.io/en/stable/SupportedFramework_Library.html">More...</a><br/>
          </ul>
      </ul>
        <ul>
        <li><b>Examples</b></li>
         <ul>
           <li><a href="examples/trials/mnist-pytorch">MNIST-pytorch</li></a>
           <li><a href="examples/trials/mnist-tfv1">MNIST-tensorflow</li></a>
           <li><a href="examples/trials/mnist-keras">MNIST-keras</li></a>
           <li><a href="https://nni.readthedocs.io/en/stable/TrialExample/GbdtExample.html">Auto-gbdt</a></li>
           <li><a href="https://nni.readthedocs.io/en/stable/TrialExample/Cifar10Examples.html">Cifar10-pytorch</li></a>
           <li><a href="https://nni.readthedocs.io/en/stable/TrialExample/SklearnExamples.html">Scikit-learn</a></li>
           <li><a href="https://nni.readthedocs.io/en/stable/TrialExample/EfficientNet.html">EfficientNet</a></li>
           <li><a href="https://nni.readthedocs.io/en/stable/TrialExample/OpEvoExamples.html">Kernel Tunning</li></a>
              <a href="https://nni.readthedocs.io/en/stable/SupportedFramework_Library.html">More...</a><br/>
          </ul>
        </ul>
      </td>
      <td align="left" >
        <a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html">Hyperparameter Tuning</a>
        <ul>
          <b>Exhaustive search</b>
          <ul>
            <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#Random">Random Search</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#GridSearch">Grid Search</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#Batch">Batch</a></li>
            </ul>
          <b>Heuristic search</b>
          <ul>
            <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#Evolution">Naïve Evolution</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#Anneal">Anneal</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#Hyperband">Hyperband</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#PBTTuner">PBT</a></li>
          </ul>
          <b>Bayesian optimization</b>
            <ul>
              <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#BOHB">BOHB</a></li>
              <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#TPE">TPE</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#SMAC">SMAC</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#MetisTuner">Metis Tuner</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#GPTuner">GP Tuner</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/Tuner/BuiltinTuner.html#DNGOTuner">DNGO Tuner</a></li>
            </ul>
        </ul>
          <a href="https://nni.readthedocs.io/en/stable/NAS/Overview.html">Neural Architecture Search (Retiarii)</a>
          <ul>
            <li><a href="https://nni.readthedocs.io/en/stable/NAS/ENAS.html">ENAS</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/NAS/DARTS.html">DARTS</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/NAS/SPOS.html">SPOS</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/NAS/Proxylessnas.html">ProxylessNAS</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/NAS/FBNet.html">FBNet</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/NAS/ExplorationStrategies.html">Reinforcement Learning</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/NAS/ExplorationStrategies.html">Regularized Evolution</a></li>
            <li><a href="https://nni.readthedocs.io/en/stable/NAS/Overview.html">More...</a></li>
          </ul>
          <a href="https://nni.readthedocs.io/en/stable/Compression/Overview.html">Model Compression</a>
          <ul>
            <b>Pruning</b>
            <ul>
              <li><a href="https://nni.readthedocs.io/en/stable/Compression/Pruner.html#agp-pruner">AGP Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/en/stable/Compression/Pruner.html#slim-pruner">Slim Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/en/stable/Compression/Pruner.html#fpgm-pruner">FPGM Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/en/stable/Compression/Pruner.html#netadapt-pruner">NetAdapt Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/en/stable/Compression/Pruner.html#simulatedannealing-pruner">SimulatedAnnealing Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/en/stable/Compression/Pruner.html#admm-pruner">ADMM Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/en/stable/Compression/Pruner.html#autocompress-pruner">AutoCompress Pruner</a></li>
              <li><a href="https://nni.readthedocs.io/en/stable/Compression/Overview.html">More...</a></li>
            </ul>
            <b>Quantization</b>
            <ul>
              <li><a href="https://nni.readthedocs.io/en/stable/Compression/Quantizer.html#qat-quantizer">QAT Quantizer</a></li>
              <li><a href="https://nni.readthedocs.io/en/stable/Compression/Quantizer.html#dorefa-quantizer">DoReFa Quantizer</a></li>
              <li><a href="https://nni.readthedocs.io/en/stable/Compression/Quantizer.html#bnn-quantizer">BNN Quantizer</a></li>
            </ul>
          </ul>
          <a href="https://nni.readthedocs.io/en/stable/FeatureEngineering/Overview.html">Feature Engineering (Beta)</a>
          <ul>
          <li><a href="https://nni.readthedocs.io/en/stable/FeatureEngineering/GradientFeatureSelector.html">GradientFeatureSelector</a></li>
          <li><a href="https://nni.readthedocs.io/en/stable/FeatureEngineering/GBDTSelector.html">GBDTSelector</a></li>
          </ul>
          <a href="https://nni.readthedocs.io/en/stable/Assessor/BuiltinAssessor.html">Early Stop Algorithms</a>
          <ul>
          <li><a href="https://nni.readthedocs.io/en/stable/Assessor/BuiltinAssessor.html#MedianStop">Median Stop</a></li>
          <li><a href="https://nni.readthedocs.io/en/stable/Assessor/BuiltinAssessor.html#Curvefitting">Curve Fitting</a></li>
          </ul>
      </td>
      <td>
      <ul>
        <li><a href="https://nni.readthedocs.io/en/stable/TrainingService/LocalMode.html">Local Machine</a></li>
        <li><a href="https://nni.readthedocs.io/en/stable/TrainingService/RemoteMachineMode.html">Remote Servers</a></li>
        <li><a href="https://nni.readthedocs.io/en/stable/TrainingService/HybridMode.html">Hybrid mode</a></li>
        <li><a href="https://nni.readthedocs.io/en/stable/TrainingService/AMLMode.html">AML(Azure Machine Learning)</a></li>
        <li><b>Kubernetes based services</b></li>
        <ul>
          <li><a href="https://nni.readthedocs.io/en/stable/TrainingService/PaiMode.html">OpenPAI</a></li>
          <li><a href="https://nni.readthedocs.io/en/stable/TrainingService/KubeflowMode.html">Kubeflow</a></li>
          <li><a href="https://nni.readthedocs.io/en/stable/TrainingService/FrameworkControllerMode.html">FrameworkController on K8S (AKS etc.)</a></li>
          <li><a href="https://nni.readthedocs.io/en/stable/TrainingService/DLTSMode.html">DLWorkspace (aka. DLTS)</a></li>
          <li><a href="https://nni.readthedocs.io/en/stable/TrainingService/AdaptDLMode.html">AdaptDL (aka. ADL)</a></li>
        </ul>
      </ul>
      </td>
    </tr>
      <tr align="center" valign="bottom">
      </td>
      </tr>
      <tr valign="top">
       <td valign="middle">
    <b>References</b>
      </td>
     <td style="border-top:#FF0000 solid 0px;">
      <ul>
        <li><a href="https://nni.readthedocs.io/en/stable/autotune_ref.html#trial">Python API</a></li>
        <li><a href="https://nni.readthedocs.io/en/stable/Tutorial/AnnotationSpec.html">NNI Annotation</a></li>
         <li><a href="https://nni.readthedocs.io/en/stable/installation.html">Supported OS</a></li>
      </ul>
      </td>
       <td style="border-top:#FF0000 solid 0px;">
      <ul>
        <li><a href="https://nni.readthedocs.io/en/stable/Tuner/CustomizeTuner.html">CustomizeTuner</a></li>
        <li><a href="https://nni.readthedocs.io/en/stable/Assessor/CustomizeAssessor.html">CustomizeAssessor</a></li>
        <li><a href="https://nni.readthedocs.io/en/stable/Tutorial/InstallCustomizedAlgos.html">Install Customized Algorithms as Builtin Tuners/Assessors/Advisors</a></li>
        <li><a href="https://nni.readthedocs.io/en/stable/NAS/QuickStart.html#define-your-model-space">Define NAS Model Space</a></li>
        <li><a href="https://nni.readthedocs.io/en/stable/NAS/ApiReference.html">NAS/Retiarii APIs</a></li>
      </ul>
      </td>
        <td style="border-top:#FF0000 solid 0px;">
      <ul>
        <li><a href="https://nni.readthedocs.io/en/stable/TrainingService/Overview.html">Support TrainingService</li>
        <li><a href="https://nni.readthedocs.io/en/stable/TrainingService/HowToImplementTrainingService.html">Implement TrainingService</a></li>
      </ul>
      </td>
    </tr>
  </tbody>
</table>

## **Installation**

### **Install**

NNI supports and is tested on Ubuntu >= 18.04, Windows 10 >= 21H2, and macOS >= 11.
Simply run the following `pip install` in an environment that has `python 64-bit >= 3.7`.

Linux or macOS

```bash
python3 -m pip install --upgrade nni
```

Windows

```bash
python -m pip install --upgrade nni
```

If you want to try latest code, please [install NNI](https://nni.readthedocs.io/en/stable/installation.html) from source code.

For detail system requirements of NNI, please refer to [here](https://nni.readthedocs.io/en/stable/Tutorial/InstallationLinux.html#system-requirements) for Linux & macOS, and [here](https://nni.readthedocs.io/en/stable/Tutorial/InstallationWin.html#system-requirements) for Windows.

Note:

* If there is any privilege issue, add `--user` to install NNI in the user directory.
* Currently NNI on Windows supports local, remote and pai mode. Anaconda or Miniconda is highly recommended to install [NNI on Windows](https://nni.readthedocs.io/en/stable/Tutorial/InstallationWin.html).
* If there is any error like `Segmentation fault`, please refer to [FAQ](https://nni.readthedocs.io/en/stable/Tutorial/FAQ.html). For FAQ on Windows, please refer to [NNI on Windows](https://nni.readthedocs.io/en/stable/Tutorial/InstallationWin.html#faq).

### **Verify installation**

* Download the examples via clone the source code.

  ```bash
  git clone -b v2.6 https://github.com/Microsoft/nni.git
  ```

* Run the MNIST example.

  Linux or macOS

  ```bash
  nnictl create --config nni/examples/trials/mnist-pytorch/config.yml
  ```

  Windows

  ```powershell
  nnictl create --config nni\examples\trials\mnist-pytorch\config_windows.yml
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

* Open the `Web UI url` in your browser, you can view detailed information of the experiment and all the submitted trial jobs as shown below. [Here](https://nni.readthedocs.io/en/stable/Tutorial/WebUI.html) are more Web UI pages.

<img src="docs/static/img/webui.gif" alt="webui" width="100%"/>

## **Releases and Contributing**
NNI has a monthly release cycle (major releases). Please let us know if you encounter a bug by [filling an issue](https://github.com/microsoft/nni/issues/new/choose).

We appreciate all contributions. If you are planning to contribute any bug-fixes, please do so without further discussions.

If you plan to contribute new features, new tuners, new training services, etc. please first open an issue or reuse an exisiting issue, and discuss the feature with us. We will discuss with you on the issue timely or set up conference calls if needed.

To learn more about making a contribution to NNI, please refer to our [How-to contribution page](https://nni.readthedocs.io/en/stable/contribution.html). 

We appreciate all contributions and thank all the contributors!

<a href="https://github.com/microsoft/nni/graphs/contributors"><img src="https://contrib.rocks/image?repo=microsoft/nni&max=240&columns=18" /></a>


## **Feedback**
* [File an issue](https://github.com/microsoft/nni/issues/new/choose) on GitHub.
* Open or participate in a [discussion](https://github.com/microsoft/nni/discussions). 
* Discuss on the NNI [Gitter](https://gitter.im/Microsoft/nni?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) in NNI.

Join IM discussion groups:
|Gitter||WeChat|
|----|----|----|
|![image](https://user-images.githubusercontent.com/39592018/80665738-e0574a80-8acc-11ea-91bc-0836dc4cbf89.png)| OR |![image](https://github.com/scarlett2018/nniutil/raw/master/wechat.png)|


## Test status

### Essentials

| Type | Status |
| :---: | :---: |
| Fast test | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/fast%20test?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=54&branchName=master) |
| Full linux | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/full%20test%20-%20linux?repoName=microsoft%2Fnni&branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=62&repoName=microsoft%2Fnni&branchName=master) |
| Full windows | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/full%20test%20-%20windows?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=63&branchName=master) |

### Training services

| Type | Status |
| :---: | :---: |
| Remote - linux to linux | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20remote%20-%20linux%20to%20linux?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=64&branchName=master) |
| Remote - linux to windows | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20remote%20-%20linux%20to%20windows?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=67&branchName=master) |
| Remote - windows to linux | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20remote%20-%20windows%20to%20linux?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=68&branchName=master) |
| OpenPAI | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20openpai%20-%20linux?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=65&branchName=master) |
| Frameworkcontroller | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20frameworkcontroller?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=70&branchName=master) |
| Kubeflow | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20kubeflow?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=69&branchName=master) |
| Hybrid | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20hybrid?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=79&branchName=master) |
| AzureML | [![Build Status](https://msrasrg.visualstudio.com/NNIOpenSource/_apis/build/status/integration%20test%20-%20aml?branchName=master)](https://msrasrg.visualstudio.com/NNIOpenSource/_build/latest?definitionId=78&branchName=master) |

## Related Projects

Targeting at openness and advancing state-of-art technology, [Microsoft Research (MSR)](https://www.microsoft.com/en-us/research/group/systems-and-networking-research-group-asia/) had also released few other open source projects.

* [OpenPAI](https://github.com/Microsoft/pai) : an open source platform that provides complete AI model training and resource management capabilities, it is easy to extend and supports on-premise, cloud and hybrid environments in various scale.
* [FrameworkController](https://github.com/Microsoft/frameworkcontroller) : an open source general-purpose Kubernetes Pod Controller that orchestrate all kinds of applications on Kubernetes by a single controller.
* [MMdnn](https://github.com/Microsoft/MMdnn) : A comprehensive, cross-framework solution to convert, visualize and diagnose deep neural network models. The "MM" in MMdnn stands for model management and "dnn" is an acronym for deep neural network.
* [SPTAG](https://github.com/Microsoft/SPTAG) : Space Partition Tree And Graph (SPTAG) is an open source library for large scale vector approximate nearest neighbor search scenario.
* [nn-Meter](https://github.com/microsoft/nn-Meter) : An accurate inference latency predictor for DNN models on diverse edge devices.

We encourage researchers and students leverage these projects to accelerate the AI development and research.

## **License**

The entire codebase is under [MIT license](LICENSE)
