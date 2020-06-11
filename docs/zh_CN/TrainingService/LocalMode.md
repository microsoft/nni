# **教程：使用 NNI API 在本地创建和运行 Experiment**

本教程会使用 [~/examples/trials/mnist-tfv1] 示例来解释如何在本地使用 NNI API 来创建并运行 Experiment。

> 在开始前

要有一个使用卷积层对 MNIST 分类的代码，如 `mnist_before.py`。

> 第一步：更新模型代码

对代码进行以下改动来启用 NNI API：

    1.1 声明 NNI API
        在 Trial 代码中通过 `import nni` 来导入 NNI API。
    
    1.2 获取预定义的参数
        参考下列代码片段： 
    
            RECEIVED_PARAMS = nni.get_next_parameter()
    
        来获得 Tuner 分配的超参值。 `RECEIVED_PARAMS` 是一个对象，例如：
    
            {"conv_size": 2, "hidden_size": 124, "learning_rate": 0.0307, "dropout_rate": 0.2029}
    
    1.3 返回结果
        使用 API：
    
            `nni.report_intermediate_result(accuracy)`
    
        返回 `accuracy` 的值给 Assessor。
    
        使用 API:
    
            `nni.report_final_result(accuracy)`
    
        返回 `accuracy` 的值给 Tuner。
    

将改动保存到 `mnist.py` 文件中。

**注意**：

    accuracy - 如果使用 NNI 内置的 Tuner/Assessor，那么 `accuracy` 必须是数值（如 float, int）。在定制 Tuner/Assessor 时 `accuracy` 可以是任何类型的 Python 对象。
    Assessor（评估器）- 会根据 Trial 的历史值（即其中间结果），来决定这次 Trial 是否应该提前终止。
    Tuner（调参器） - 会根据探索的历史（所有 Trial 的最终结果）来生成下一组参数、架构。
    

> 第二步：定义搜索空间

在 `Step 1.2 获取预定义的参数` 中使用的超参定义在 `search_space.json` 文件中：

    {
        "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
        "conv_size":{"_type":"choice","_value":[2,3,5,7]},
        "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
        "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
    }
    

参考[定义搜索空间](../Tutorial/SearchSpaceSpec.md)进一步了解。

> 第三步：定义 Experiment
> 
> > 3.1 启用 NNI API 模式

要启用 NNI 的 API 模式，需要将 useAnnotation 设置为 *false*，并提供搜索空间文件的路径（即第一步中定义的文件）：

    useAnnotation: false
    searchSpacePath: /path/to/your/search_space.json
    

在 NNI 中运行 Experiment，只需要：

* 可运行的 Trial 的代码
* 实现或选择 Tuner
* 准备 YAML 的 Experiment 配置文件
* (可选) 实现或选择 Assessor

**准备 Trial**：

> 在克隆代码后，可以在 ~/nni/examples 中找到一些示例，运行 `ls examples/trials` 查看所有 Trial 示例。

先从 NNI 提供的简单 Trial 示例，如 MNIST 开始。 NNI 示例在代码目录的 examples 中，运行 `ls ~/nni/examples/trials` 可以看到所有 Experiment 的示例。 执行下面的命令可轻松运行 NNI 的 mnist 示例：

      python ~/nni/examples/trials/mnist-annotation/mnist.py
    

上面的命令会写在 YAML 文件中。 参考[这里](../TrialExample/Trials.md)来写出自己的 Experiment 代码。

**准备 Tuner**: NNI 支持多种流行的自动机器学习算法，包括：Random Search（随机搜索），Tree of Parzen Estimators (TPE)，Evolution（进化算法）等等。 也可以实现自己的 Tuner（参考[这里](../Tuner/CustomizeTuner.md)）。下面使用了 NNI 内置的 Tuner：

      tuner:
        builtinTunerName: TPE
        classArgs:
          optimize_mode: maximize
    

*builtinTunerName* 用来指定 NNI 中的 Tuner，*classArgs* 是传入到 Tuner的参数（内置 Tuner 在[这里](../Tuner/BuiltinTuner.md)），*optimization_mode* 表明需要最大化还是最小化 Trial 的结果。

**准备配置文件**：实现 Trial 的代码，并选择或实现自定义的 Tuner 后，就要准备 YAML 配置文件了。 NNI 为每个 Trial 示例都提供了演示的配置文件，用命令`cat ~/nni/examples/trials/mnist-annotation/config.yml` 来查看其内容。 大致内容如下：

```yaml
authorName: your_name
experimentName: auto_mnist

# 并发运行数量
trialConcurrency: 2

# Experiment 运行时间
maxExecDuration: 3h

# 可为空，即数量不限
maxTrialNum: 100

# 可选值为: local, remote
trainingServicePlatform: local

# 搜索空间文件
searchSpacePath: search_space.json

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

因为这个 Trial 代码使用了 NNI Annotation 的方法（参考[这里](../Tutorial/AnnotationSpec.md) ），所以*useAnnotation* 为 true。 *command* 是运行 Trial 代码所需要的命令，*codeDir* 是 Trial 代码的相对位置。 命令会在此目录中执行。 同时，也需要提供每个 Trial 进程所需的 GPU 数量。

完成上述步骤后，可通过下列命令来启动 Experiment：

      nnictl create --config ~/nni/examples/trials/mnist-annotation/config.yml
    

参考[这里](../Tutorial/Nnictl.md)来了解 *nnictl* 命令行工具的更多用法。

## 查看 Experiment 结果

Experiment 应该一直在运行。 除了 *nnictl* 以外，还可以通过 NNI 的网页来查看 Experiment 进程，进行控制和其它一些有意思的功能。

## 使用多个本地 GPU 加快搜索速度

下列步骤假设本机有 4 块 NVIDIA GPUs，参考 [tensorflow with GPU support](https://www.tensorflow.org/install/gpu)。 演示启用了 4 个并发的 Trial 任务，每个 Trial 任务使用了 1 块 GPU。

**准备配置文件**：NNI 提供了演示用的配置文件，使用 `cat examples/trials/mnist-annotation/config_gpu.yml` 来查看。 trailConcurrency 和 gpuNum 与基本配置文件不同：

    ...
    
    # 可同时运行的 Trial 数量
    trialConcurrency: 4
    
    ...
    
    trial:
      command: python mnist.py
      codeDir: ~/nni/examples/trials/mnist-annotation
      gpuNum: 1
    

用下列命令运行 Experiment：

      nnictl create --config ~/nni/examples/trials/mnist-annotation/config_gpu.yml
    

可以用 *nnictl* 命令行工具或网页界面来跟踪训练过程。 *nvidia_smi* 命令行工具能在训练过程中查看 GPU 使用情况。