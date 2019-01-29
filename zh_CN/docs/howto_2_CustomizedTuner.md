# **指南** - 自定义 Tuner

*Tuner 从 Trial 接收指标结果，来评估一组超参或网络结构的性能。 然后 Tuner 会将下一组超参或网络结构的配置发送给新的 Trial。*

因此，如果要自定义 Tuner，需要：

1. 从基类 Tuner 继承，创建新的 Tuner 子类
2. 实现 receive_trial_result 和 generate_parameter 函数
3. Configure your customized tuner in experiment YAML config file

样例如下：

**1) 从基类 Tuner 继承，创建新的 Tuner 子类**

```python
from nni.tuner import Tuner

class CustomizedTuner(Tuner):
    def __init__(self, ...):
        ...
```

**2) 实现 receive_trial_result 和 generate_parameter 函数**

```python
from nni.tuner import Tuner

class CustomizedTuner(Tuner):
    def __init__(self, ...):
        ...

    def receive_trial_result(self, parameter_id, parameters, value):
    '''
    Record an observation of the objective function and Train
    parameter_id: int
    parameters: object created by 'generate_parameters()'
    value: final metrics of the trial, including reward
    '''
    # your code implements here.
    ...

    def generate_parameters(self, parameter_id):
    '''
    Returns a set of trial (hyper-)parameters, as a serializable object
    parameter_id: int
    '''
    # your code implements here.
    return your_parameters
    ...
```

`receive_trial_result` 从输入中会接收 `parameter_id, parameters, value` 参数。 Tuner 会收到 Trial 进程发送的完全一样的 `value` 值。

`generate_parameters` 函数返回的 `your_parameters`，会被 NNI SDK 打包为 json。 然后 SDK 会将 json 对象解包给 Trial 进程。因此，Trial 进程会收到来自 Tuner 的完全相同的 `your_parameters`。

例如， 如下实现了 `generate_parameters`：

```python
    def generate_parameters(self, parameter_id):
        '''
        返回 Trial 的超参组合的序列化对象
        parameter_id: int
        '''
        # 代码实现位置
        return {"dropout": 0.3, "learning_rate": 0.4}
```

这表示 Tuner 会一直生成参数 `{"dropout": 0.3, "learning_rate": 0.4}`。 而 Trial 进程也会在调用 API `nni.get_next_parameter()` 时得到 `{"dropout": 0.3, "learning_rate": 0.4}`。 Trial 结束后的返回值（通常是某个指标），通过调用 API `nni.report_final_result()` 返回给 Tuner。如： `nni.report_final_result(0.93)`。 而 Tuner 的 `receive_trial_result` 函数会收到如下结果：

```python
parameter_id = 82347
parameters = {"dropout": 0.3, "learning_rate": 0.4}
value = 0.93
```

**注意** 如果需要存取自定义的 Tuner 目录里的文件 (如, `data.txt`)，不能使用 `open('data.txt', 'r')`。 要使用：

```python
_pwd = os.path.dirname(__file__)
_fd = open(os.path.join(_pwd, 'data.txt'), 'r')
```

这是因为自定义的 Tuner 不是在自己的目录里执行的。（即，`pwd` 返回的目录不是 Tuner 的目录）。

**3) Configure your customized tuner in experiment YAML config file**

NNI 需要定位到自定义的 Tuner 类，并实例化它，因此需要指定自定义 Tuner 类的文件位置，并将参数值传给 \_\_init__ 构造函数。

```yml
tuner:
  codeDir: /home/abc/mytuner
  classFileName: my_customized_tuner.py
  className: CustomizedTuner
  # 任何传入 __init__ 构造函数的参数
  # 都需要声明在 classArgs 字段中，如：
  classArgs:
    arg1: value1
```

更多样例，可参考：

> - [evolution-tuner](../../src/sdk/pynni/nni/evolution_tuner)
> - [hyperopt-tuner](../../src/sdk/pynni/nni/hyperopt_tuner)
> - [evolution-based-customized-tuner](../../examples/tuners/ga_customer_tuner)

## 实现更高级的自动机器学习算法

上述内容足够写出通用的 Tuner。 但有时可能需要更多的信息，例如，中间结果， Trial 的状态等等，从而能够实现更强大的自动机器学习算法。 因此，有另一个叫做 `advisor` 的类，直接继承于 `MsgDispatcherBase`，它位于 [`src/sdk/pynni/nni/msg_dispatcher_base.py`](../../src/sdk/pynni/nni/msg_dispatcher_base.py)。 参考[这里](./howto_3_CustomizedAdvisor.md)来了解如何实现自定义的 advisor。