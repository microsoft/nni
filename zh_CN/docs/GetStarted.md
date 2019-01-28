# **开始使用 NNI**

## **安装**

* **依赖项**
    
    ```bash
    python >= 3.5
    git
    wget
    ```
    
    需要正确安装 Python 的 pip。 可以用 "python3 -m pip -v" 来检查 pip 版本。
    
    * 注意：当前版本不支持虚拟环境。

* **通过 pip 命令安装 NNI**
    
    ```bash
    python3 -m pip install --user --upgrade nni
    ```

* **通过源代码安装 NNI**
    
    ```bash
    git clone -b v0.5 https://github.com/Microsoft/nni.git
    cd nni
    source install.sh
    ```

## **快速入门：运行自定义的 Experiment**

Experiment 会运行多个 Trial 任务，每个 Trial 任务会使用特定的神经网络（或模型）结构以及超参的值。 运行 NNI Experiment，需要如下准备：

* 可运行的 Trial 的代码
* 实现或选择 Tuner
* 准备 yaml 的 Experiment 配置文件
* (可选) 实现或选择 Assessor

**准备 Trial**: 先从简单样例开始，如：NNI 样例中的 mnist。 NNI 样例在代码目录的 examples 中，运行 `ls ~/nni/examples/trials` 可以看到所有 Experiment 的样例。 执行下面的命令可轻松运行 NNI 的 mnist 样例：

```bash
python3 ~/nni/examples/trials/mnist-annotation/mnist.py
```

上面的命令会写在 yml 文件中。 参考[这里](howto_1_WriteTrial.md)来写出自己的 Experiment 代码。

**准备 Tuner**: NNI 支持多种流行的自动机器学习算法，包括：Random Search（随机搜索），Tree of Parzen Estimators (TPE)，Evolution（进化算法）等等。 也可以实现自己的 Tuner（参考[这里](howto_2_CustomizedTuner.md)）。下面使用的是 NNI 内置 Tuner：

```yaml
tuner:
  builtinTunerName: TPE
    classArgs:
      optimize_mode: maximize
```

*builtinTunerName* 用来指定 NNI 中的 Tuner，*classArgs* 是传入到 Tuner 的参数，*optimization_mode* 表明需要最大化还是最小化 Trial 的结果。

**准备配置文件**：实现 Trial 的代码，并选择或实现自定义的 Tuner 后，就要准备 yml 配置文件了。 NNI 为每个 Trial 样例都提供了演示的配置文件，用命令`cat ~/nni/examples/trials/mnist-annotation/config.yml` 来查看其内容。 大致内容如下：

```yaml
authorName: your_name
experimentName: auto_mnist

# 并发运行数量
trialConcurrency: 2

# Experiment 运行时间
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
```

因为此 Trial 代码使用了 NNI 标记的方法（参考[这里](../tools/annotation/README.md) ），所以 *useAnnotation* 为 true。 *command* 是运行 Trial 代码所需要的命令，*codeDir* 是 Trial 代码的相对位置。 命令会在此目录中执行。 同时，也需要提供每个 Trial 进程所需的 GPU 数量。

完成上述步骤后，可通过下列命令来启动 Experiment：

      nnictl create --config ~/nni/examples/trials/mnist-annotation/config.yml
    

参考[这里](NNICTLDOC.md)来了解 *nnictl* 命令行工具的更多用法。

## 查看 Experiment 结果

Experiment 开始运行后，可以通过 NNI 的网页来查看Experiment 进程，并进行控制等。 网页界面默认会通过 `nnictl create` 命令打开。

## 更多内容

* [NNI 最新版本支持的 Tuner](./HowToChooseTuner.md)
* [概述](Overview.md)
* [安装](Installation.md)
* [使用命令行工具 nnictl](NNICTLDOC.md)
* [使用 NNIBoard](WebUI.md)
* [定制搜索空间](SearchSpaceSpec.md)
* [配置 Experiment](ExperimentConfig.md)
* [如何在本机运行 Experiment (支持多 GPU 卡)？](tutorial_1_CR_exp_local_api.md)
* [如何在多机上运行 Experiment？](tutorial_2_RemoteMachineMode.md)
* [如何在 OpenPAI 上运行 Experiment？](PAIMode.md)
* [如何创建多阶段的 Experiment](multiPhase.md)