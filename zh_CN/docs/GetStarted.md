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
    

This command will be filled in the yaml configure file below. Please refer to [here](howto_1_WriteTrial.md) for how to write your own trial.

**Prepare tuner**: NNI supports several popular automl algorithms, including Random Search, Tree of Parzen Estimators (TPE), Evolution algorithm etc. Users can write their own tuner (refer to [here](howto_2_CustomizedTuner.md), but for simplicity, here we choose a tuner provided by NNI as below:

      tuner:
        builtinTunerName: TPE
        classArgs:
          optimize_mode: maximize
    

*builtinTunerName* is used to specify a tuner in NNI, *classArgs* are the arguments pass to the tuner, *optimization_mode* is to indicate whether you want to maximize or minimize your trial's result.

**Prepare configure file**: Since you have already known which trial code you are going to run and which tuner you are going to use, it is time to prepare the yaml configure file. NNI provides a demo configure file for each trial example, `cat ~/nni/examples/trials/mnist-annotation/config.yml` to see it. Its content is basically shown below:

    authorName: your_name
    experimentName: auto_mnist
    
    # how many trials could be concurrently running
    trialConcurrency: 2
    
    # maximum experiment running duration
    maxExecDuration: 3h
    
    # empty means never stop
    maxTrialNum: 100
    
    # choice: local, remote, pai
    trainingServicePlatform: local
    
    # choice: true, false  
    useAnnotation: true
    tuner:
      builtinTunerName: TPE
      classArgs:
        optimize_mode: maximize
    trial:
      command: python mnist.py
      codeDir: ~/nni/examples/trials/mnist-annotation
      gpuNum: 0
    

Here *useAnnotation* is true because this trial example uses our python annotation (refer to [here](../tools/annotation/README.md) for details). For trial, we should provide *trialCommand* which is the command to run the trial, provide *trialCodeDir* where the trial code is. The command will be executed in this directory. We should also provide how many GPUs a trial requires.

With all these steps done, we can run the experiment with the following command:

      nnictl create --config ~/nni/examples/trials/mnist-annotation/config.yml
    

You can refer to [here](NNICTLDOC.md) for more usage guide of *nnictl* command line tool.

## 查看实验结果

The experiment has been running now, NNI provides WebUI for you to view experiment progress, to control your experiment, and some other appealing features. The WebUI is opened by default by `nnictl create`.

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