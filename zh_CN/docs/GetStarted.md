# **开始使用 NNI**

## **安装**

* **依赖项**
    
    python >= 3.5 git wget
    
    python pip should also be correctly installed. You could use "python3 -m pip -v" to check in Linux.
    
    * 注意：当前版本不支持虚拟环境。

* **通过 pip 命令安装 NNI**
    
    python3 -m pip install --user --upgrade nni

* **通过源代码安装 NNI**
    
    git clone -b v0.4.1 https://github.com/Microsoft/nni.git cd nni source install.sh

## **快速入门：运行自定义的实验**

实验会运行多个尝试任务，每个尝试任务会使用特定的神经网络（或模型）结构以及超参的值。 运行 NNI 实验，需要如下准备：

* 可运行的尝试的代码
* 实现或选择调参器
* 准备 yaml 的实验配置文件
* (可选) 实现或选择评估器

**准备尝试**: 先从简单样例开始，如：NNI 样例中的 mnist。 NNI 样例在代码目录的 examples 中，运行 `ls ~/nni/examples/trials` 可以看到所有实验的样例。 执行下面的命令可轻松运行 NNI 的 mnist 样例：

      python3 ~/nni/examples/trials/mnist-annotation/mnist.py
    

上面的命令会写在 yaml 文件中。 参考[这里](howto_1_WriteTrial.md)来写出自己的实验代码。

**准备调参器**: NNI 支持多种流行的自动机器学习算法，包括：Random Search（随机搜索），Tree of Parzen Estimators (TPE)，Evolution（进化算法）等等。 也可以实现自己的调参器（参考[这里](howto_2_CustomizedTuner.md)）。下面使用了 NNI 内置的调参器：

      tuner:
        builtinTunerName: TPE
        classArgs:
          optimize_mode: maximize
    

*builtinTunerName* 用来指定 NNI 中的调参器，*classArgs* 是传入到调参器的参数，*optimization_mode* 表明需要最大化还是最小化尝试的结果。

**准备配置文件**：实现尝试的代码，并选择或实现自定义的调参器后，就要准备 yaml 配置文件了。 NNI 为每个尝试样例都提供了演示的配置文件，用命令`cat ~/nni/examples/trials/mnist-annotation/config.yml` 来查看其内容。 大致内容如下：

    authorName: your_name
    experimentName: auto_mnist
    
    # 并发运行数量
    trialConcurrency: 2
    
    # 实验运行时间
    maxExecDuration: 3h
    
    # 可为空，即数量不限
    maxTrialNum: 100
    
    # 可选值为: local, remote, pai
    trainingServicePlatform: local
    
    # 可选值为: true, false  
    useAnnotation: true
    tuner:
      builtinTunerName: TPE
      classArgs:
        optimize_mode: maximize
    trial:
      command: python mnist.py
      codeDir: ~/nni/examples/trials/mnist-annotation
      gpuNum: 0
    

因为此尝试代码使用了 NNI 标记的方法（参考[这里](../tools/annotation/README.md) ），所以 *useAnnotation* 为 true。 *command* 是运行尝试代码所需要的命令，*codeDir* 是尝试代码的相对位置。 命令会在此目录中执行。 同时，也需要提供每个尝试进程所需的 GPU 数量。

完成上述步骤后，可通过下列命令来启动实验：

      nnictl create --config ~/nni/examples/trials/mnist-annotation/config.yml
    

参考[这里](NNICTLDOC.md)来了解 *nnictl* 命令行工具的更多用法。

## 查看实验结果

实验开始运行后，可以通过 NNI 的网页来查看实验进程，并进行控制等。 网页界面默认会通过 `nnictl create` 命令打开。

## 更多内容

* [NNI 最新版本支持的调参器](./HowToChooseTuner.md)
* [概述](Overview.md)
* [安装](Installation.md)
* [使用命令行工具 nnictl](NNICTLDOC.md)
* [使用 NNIBoard](WebUI.md)
* [定制搜索空间](SearchSpaceSpec.md)
* [配置实验](ExperimentConfig.md)
* [如何在本机运行实验 (支持多 GPU 卡)？](tutorial_1_CR_exp_local_api.md)
* [如何在多机上运行实验？](tutorial_2_RemoteMachineMode.md)
* [如何在 OpenPAI 上运行实验？](PAIMode.md)