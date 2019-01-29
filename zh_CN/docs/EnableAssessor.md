# **使用 Assessor**

Assessor 模块用于评估正在运行的 Trial。 最常用的情况是提前中止 Trial。如果 Trial 的中间结果不够好，则可提前终止。

## 使用内置的 Assessor 

以下样例代码在 `examples/trials/mnist-annotation` 目录中。 此 Experiment 使用了 `Medianstop` Assessor。 YAML 配置文件如下：

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
```

如使用内置的 Assessor，需要填写两个字段: `builtinAssessorName`，即所选择的Assessor (参考[这里]())，`optimize_mode` 可选项为 maximize 和 minimize (即需要最大化或最小化的结果)。

## 使用自定义的 Assessor

可参考[这里]()，来自定义 Assessor。 例如，为样例代码 `examples/trials/mnist-annotation` 写一个定制的 Assessor。 需要准备如下的 YAML 配置文件：

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
# 可选值为: true, false  
useAnnotation: true
tuner:
  # 可选值为: TPE, Random, Anneal, Evolution
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
assessor:
  # Assessor 代码目录
  codeDir: 
  # Assessor 类的文件名
  classFileName: 
  # Assessor 类名，必须继承于 nni.Assessor
  className: 
  # 参数名和需要输入给 Assessor __init__ 构造函数的值。
  classArgs:
    arg1: value1
  gpuNum: 0
trial:
  command: python mnist.py
  codeDir: /usr/share/nni/examples/trials/mnist-annotation
  gpuNum: 0
```

必填项: `codeDir`, `classFileName`, `className`。如果 Assessor 的 `__init__` 构造函数有必填参数，需要用 `classArgs` 传入。

**Note that** if you want to access a file (e.g., `data.txt`) in the directory of your own assessor, you cannot use `open('data.txt', 'r')`. Instead, you should use the following:

```python
_pwd = os.path.dirname(__file__)
_fd = open(os.path.join(_pwd, 'data.txt'), 'r')
```

This is because your assessor is not executed in the directory of your assessor (i.e., `pwd` is not the directory of your own assessor).