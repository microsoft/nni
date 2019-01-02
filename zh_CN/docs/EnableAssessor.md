# **使用评估器**

评估器模块用于评估正在运行的尝试。 最常用的情况是提前中止尝试。如果尝试的中间结果不够好，则可提前终止。

## 使用 NNI 内置的评估器

以下样例代码在 `examples/trials/mnist-annotation` 目录中。 此实验使用了 `Medianstop` 评估器。 yaml 配置文件如下：

    authorName: your_name
    experimentName: auto_mnist
    # 并发运行数量
    trialConcurrency: 2
    # 实验运行时间
    maxExecDuration: 3h
    # 可为空，即数量不限
    maxTrialNum: 100
    # 可选值为: local, remote  
    trainingServicePlatform: local
    # 可选值为: true, false  
    useAnnotation: true
    tuner:
      builtinTunerName: TPE
      classArgs:
        optimize_mode: maximize
    assessor:
      builtinAssessorName: Medianstop
      classArgs:
        optimize_mode: maximize
    trial:
      command: python mnist.py
      codeDir: /usr/share/nni/examples/trials/mnist-annotation
      gpuNum: 0
    

如使用内置的评估器，需要填写两个字段: `builtinAssessorName`，即所选择的评估器 (参考[这里]())，`optimize_mode` 可选项为 maximize 和 minimize (即需要最大化或最小化的结果)。

## 使用自定义的评估器

可参考[这里]()，来自定义评估器。 例如，为样例代码 `examples/trials/mnist-annotation` 写一个定制的评估器。 需要准备如下的 yaml 配置文件：

    authorName: your_name
    experimentName: auto_mnist
    # 并发运行数量
    trialConcurrency: 2
    # 实验运行时间
    maxExecDuration: 3h
    # 可为空，即数量不限
    maxTrialNum: 100
    # 可选值为: local, remote  
    trainingServicePlatform: local
    # 可选值为: true, false  
    useAnnotation: true
    tuner:
      # 可选值为: TPE, Random, Anneal, Evolution
      builtinTunerName: TPE
      classArgs:
        optimize_mode: maximize
    assessor:
      # 评估器代码目录
      codeDir: 
      # 评估器类的文件名
      classFileName: 
      # 评估器类名，必须继承于 nni.Assessor
      className: 
      # 参数名和需要输入给评估器 __init__ 构造函数的值。
      classArgs:
        arg1: value1
      gpuNum: 0
    trial:
      command: python mnist.py
      codeDir: /usr/share/nni/examples/trials/mnist-annotation
      gpuNum: 0
    

必填项: `codeDir`, `classFileName`, `className`。如果评估器的 `__init__` 构造函数有必填参数，需要用 `classArgs` 传入。

**注意** 如果需要访问 assessor 目录中的文件 （如： ```data.txt```），不能使用 ```open('data.txt', 'r')```。 要使用：

    _pwd = os.path.dirname(__file__)
    _fd = open(os.path.join(_pwd, 'data.txt'), 'r')
    

因为评估器不是在其自己的目录中执行的。（也就是说 ```pwd``` 不是评估器自己的目录）。