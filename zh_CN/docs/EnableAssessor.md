# **使用评估器**

评估器模块用于评估正在运行的尝试。 最常用的情况是提前中止尝试。如果尝试的中间结果不够好，则可提前终止。

## 使用 NNI 内置的评估器

以下样例代码在 `examples/trials/mnist-annotation` 目录中。 We use `Medianstop` assessor for this experiment. The yaml configure file is shown below:

    authorName: your_name
    experimentName: auto_mnist
    # how many trials could be concurrently running
    trialConcurrency: 2
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
    assessor:
      builtinAssessorName: Medianstop
      classArgs:
        optimize_mode: maximize
    trial:
      command: python mnist.py
      codeDir: /usr/share/nni/examples/trials/mnist-annotation
      gpuNum: 0
    

For our built-in assessors, you need to fill two fields: `builtinAssessorName` which chooses NNI provided assessors (refer to [here]() for built-in assessors), `optimize_mode` which includes maximize and minimize (you want to maximize or minimize your trial result).

## Using user customized Assessor

You can also write your own assessor following the guidance [here](). For example, you wrote an assessor for `examples/trials/mnist-annotation`. You should prepare the yaml configure below:

    authorName: your_name
    experimentName: auto_mnist
    # how many trials could be concurrently running
    trialConcurrency: 2
    # maximum experiment running duration
    maxExecDuration: 3h
    # empty means never stop
    maxTrialNum: 100
    # choice: local, remote  
    trainingServicePlatform: local
    # choice: true, false  
    useAnnotation: true
    tuner:
      # Possible values: TPE, Random, Anneal, Evolution
      builtinTunerName: TPE
      classArgs:
        optimize_mode: maximize
    assessor:
      # Your assessor code directory
      codeDir: 
      # Name of the file which contains your assessor class
      classFileName: 
      # Your assessor class name, must be a subclass of nni.Assessor
      className: 
      # Parameter names and literal values you want to pass to
      # the __init__ constructor of your assessor class
      classArgs:
        arg1: value1
      gpuNum: 0
    trial:
      command: python mnist.py
      codeDir: /usr/share/nni/examples/trials/mnist-annotation
      gpuNum: 0
    

You need to fill: `codeDir`, `classFileName`, `className`, and pass parameters to *\_init__ constructor through `classArgs` field if the *\_init__ constructor of your assessor class has required parameters.

**Note that** if you want to access a file (e.g., ```data.txt```) in the directory of your own assessor, you cannot use ```open('data.txt', 'r')```. Instead, you should use the following:

    _pwd = os.path.dirname(__file__)
    _fd = open(os.path.join(_pwd, 'data.txt'), 'r')
    

This is because your assessor is not executed in the directory of your assessor (i.e., ```pwd``` is not the directory of your own assessor).