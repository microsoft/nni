<div align="center">
<img src="docs/img/nni_logo.png" width="100%"/>
</div>

[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
[![Issues](https://img.shields.io/github/issues-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen)
[![Bugs](https://img.shields.io/github/issues/Microsoft/nni/bug.svg)](https://github.com/Microsoft/nni/issues?q=is%3Aissue+is%3Aopen+label%3Abug)
[![Pull Requests](https://img.shields.io/github/issues-pr-raw/Microsoft/nni.svg)](https://github.com/Microsoft/nni/pulls?q=is%3Apr+is%3Aopen)
[![Version](https://img.shields.io/github/release/Microsoft/nni.svg)](https://github.com/Microsoft/nni/releases)
[![Documentation Status](https://readthedocs.org/projects/nni/badge/?version=stable)](https://nni.readthedocs.io/en/stable/?badge=stable)

NNI automates feature engineering, neural architecture search, hyperparameter tuning, and model compression for deep learning. Find the latest features, API, examples and tutorials in our **[official documentation](https://nni.readthedocs.io/)** ([简体中文](https://nni.readthedocs.io/zh/)). Quick links:

* [Documentation homepage](https://nni.readthedocs.io/)
* [Installation guide](https://nni.readthedocs.io/en/stable/installation.html)
* [Tutorials](https://nni.readthedocs.io/en/stable/tutorials.html)
* [Python API reference](https://nni.readthedocs.io/en/stable/reference/python_api.html)
* [Releases](https://nni.readthedocs.io/en/stable/Release.html)

## What's NEW! &nbsp;<a href="#nni-released-reminder"><img width="48" src="docs/img/release_icon.png"></a>

* **New release**: [v2.6 is available](https://github.com/microsoft/nni/releases/tag/v2.6) - _released on Jan-19-2022_
* **New demo available**: [Youtube entry](https://www.youtube.com/channel/UCKcafm6861B2mnYhPbZHavw) | [Bilibili 入口](https://space.bilibili.com/1649051673) - _last updated on May-26-2021_
* **New webinar**: [Introducing Retiarii: A deep learning exploratory-training framework on NNI](https://note.microsoft.com/MSR-Webinar-Retiarii-Registration-Live.html) - _scheduled on June-24-2021_
* **New community channel**: [Discussions](https://github.com/microsoft/nni/discussions)
* **New emoticons release**: [nnSpider](./docs/source/Tutorial/NNSpider.md)
<div align="center">
<a href="#nni-spider"><img width="100%" src="docs/img/emoicons/home.svg" /></a>
</div>

## NNI capabilities in a glance

(TBD: figures and tables)

## Installation

See the [NNI installation guide](https://nni.readthedocs.io/en/stable/installation.html) to install from pip, or build from source.

To install the current release:

```
$ pip install nni
```

To update NNI to the latest version, add `--upgrade` flag to the above commands.

(TBD: build from soure link)
  
## Run your first experiment

To run this experiment, you need to have [XXX](link) installed.

```shell
$ nnictl hello-world
```

Wait for the message `INFO: Successfully started experiment!` in the command line. This message indicates that your experiment has been successfully started. You can explore the experiment using the `Web UI url` shown in the console.

```text
TBD
```

<img src="docs/static/img/webui.gif" alt="webui" width="100%"/>

For more usages, please see [NNI tutorials](https://nni.readthedocs.io/en/stable/tutorials.html).

## Contribution guidelines

If you want to contribute to NNI, be sure to review the [contribution guidelines](https://nni.readthedocs.io/en/stable/notes/contributing.html), which includes instructions of submitting feedbacks, best coding practices, and code of conduct.

We use [GitHub issues](https://github.com/microsoft/nni/issues) to track tracking requests and bugs.
Please use [NNI Discussion](https://github.com/microsoft/nni/discussions) for general questions and new ideas.
For questions of specific use cases, please go to [Stack Overflow](https://stackoverflow.com/questions/tagged/nni).

Participating discussions via the following IM groups is also welcomed.

|Gitter||WeChat|
|----|----|----|
|![image](https://user-images.githubusercontent.com/39592018/80665738-e0574a80-8acc-11ea-91bc-0836dc4cbf89.png)| OR |![image](https://github.com/scarlett2018/nniutil/raw/master/wechat.png)|

Over the past few years, NNI has received thousands of feedbacks on GitHub issues, and pull requests from hundreds of contributors.
We appreciate all contributions from community to make this tool thrive.

<a href="https://github.com/microsoft/nni/graphs/contributors"><img src="https://contrib.rocks/image?repo=microsoft/nni&max=240" width="60%" /></a>

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
