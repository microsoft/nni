# **教程：使用 NNI API 在本地创建和运行实验**

本教程会使用 [~/examples/trials/mnist] 样例来解释如何在本地使用 NNI API 来创建并运行实验。

> 在开始前

要有一个使用卷积层对 MNIST 分类的代码，如 `mnist_before.py`。

> 第一步：更新模型代码

对代码进行以下改动来启用 NNI API：

    1.1 声明 NNI API
        在尝试代码中通过 `import nni` 来导入 NNI API。 
    
    1.2 获取预定义的参数
        参考下列代码片段： 
    
            RECEIVED_PARAMS = nni.get_next_parameter()
    
        来获得调参器分配的超参值。 `RECEIVED_PARAMS` 是一个对象，例如： 
    
            {"conv_size": 2, "hidden_size": 124, "learning_rate": 0.0307, "dropout_rate": 0.2029}
    
    1.3 向 NNI 返回结果
        使用 API：
    
            `nni.report_intermediate_result(accuracy)` 
    
        返回 `accuracy` 的值给评估器。
    
        使用 API:
    
            `nni.report_final_result(accuracy)` 
    
        返回 `accuracy` 的值给调参器。 
    

将改动保存到 `mnist.py` 文件中。

**注意**：

    accuracy - 如果使用 NNI 内置的调参器/评估器，那么 `accuracy` 必须是数值（如 float, int）。在定制调参器/评估器时 `accuracy` 可以是任何类型的 Python 对象。
    评估器 - 会根据尝试的历史值（即其中间结果），来决定这次尝试是否应该提前终止。
    调参器 - 会根据探索的历史（所有尝试的最终结果）来生成下一组参数、架构。
    

> 第二步：定义搜索空间

在 `Step 1.2 获取预定义的参数` 中使用的超参定义在 `search_space.json` 文件中：

    {
        "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
        "conv_size":{"_type":"choice","_value":[2,3,5,7]},
        "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
        "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
    }
    

参考 [SearchSpaceSpec.md](./SearchSpaceSpec.md) 进一步了解搜索空间。

> 第三步：定义实验
> 
> > 3.1 启用 NNI API 模式

要启用 NNI 的 API 模式，需要将 useAnnotation 设置为 *false*，并提供搜索空间文件的路径（即第一步中定义的文件）：

    useAnnotation: false
    searchSpacePath: /path/to/your/search_space.json
    

在 NNI 中运行实验，只需要：

* 可运行的尝试的代码
* 实现或选择调参器
* 准备 yaml 的实验配置文件
* (可选) 实现或选择评估器

**准备尝试**：

> 在克隆代码后，可以在 ~/nni/examples 中找到一些样例，运行 `ls examples/trials` 查看所有尝试样例。

先从 NNI 提供的简单尝试样例，如 MNIST 开始。 NNI 样例在代码目录的 examples 中，运行 `ls ~/nni/examples/trials` 可以看到所有实验的样例。 执行下面的命令可轻松运行 NNI 的 mnist 样例：

      python ~/nni/examples/trials/mnist-annotation/mnist.py
    

上面的命令会写在 yaml 文件中。 参考[这里](./howto_1_WriteTrial.md)来写出自己的实验代码。

**准备调参器**: NNI 支持多种流行的自动机器学习算法，包括：Random Search（随机搜索），Tree of Parzen Estimators (TPE)，Evolution（进化算法）等等。 也可以实现自己的调参器（参考[这里](./CustomizedTuner.md)）。下面使用了 NNI 内置的调参器：

      tuner:
        builtinTunerName: TPE
        classArgs:
          optimize_mode: maximize
    

*builtinTunerName* is used to specify a tuner in NNI, *classArgs* are the arguments pass to the tuner (the spec of builtin tuners can be found [here]()), *optimization_mode* is to indicate whether you want to maximize or minimize your trial's result.

**Prepare configure file**: Since you have already known which trial code you are going to run and which tuner you are going to use, it is time to prepare the yaml configure file. NNI provides a demo configure file for each trial example, `cat ~/nni/examples/trials/mnist-annotation/config.yml` to see it. Its content is basically shown below:

    authorName: your_name
    experimentName: auto_mnist
    
    # how many trials could be concurrently running
    trialConcurrency: 1
    
    # maximum experiment running duration
    maxExecDuration: 3h
    
    # empty means never stop
    maxTrialNum: 100
    
    # choice: local, remote  
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

## View experiment results

The experiment has been running now. Oher than *nnictl*, NNI also provides WebUI for you to view experiment progress, to control your experiment, and some other appealing features.

## Using multiple local GPUs to speed up search

The following steps assume that you have 4 NVIDIA GPUs installed at local and [tensorflow with GPU support](https://www.tensorflow.org/install/gpu). The demo enables 4 concurrent trail jobs and each trail job uses 1 GPU.

**Prepare configure file**: NNI provides a demo configuration file for the setting above, `cat ~/nni/examples/trials/mnist-annotation/config_gpu.yml` to see it. The trailConcurrency and gpuNum are different from the basic configure file:

    ...
    
    # how many trials could be concurrently running
    trialConcurrency: 4
    
    ...
    
    trial:
      command: python mnist.py
      codeDir: ~/nni/examples/trials/mnist-annotation
      gpuNum: 1
    

We can run the experiment with the following command:

      nnictl create --config ~/nni/examples/trials/mnist-annotation/config_gpu.yml
    

You can use *nnictl* command line tool or WebUI to trace the training progress. *nvidia_smi* command line tool can also help you to monitor the GPU usage during training.