# **指南** - 自定义 Advisor

*Advisor 用于同时需要 Tuner 和 Assessor 方法的自动机器学习算法。 Advisor 与 Tuner 类似，它接收 Trial 的参数请求、最终结果，并生成 Trial 的参数。 另外，它也能像 Assessor 一样接收中间结果、Trial 的最终状态，并可以发送终止 Trial 的命令。 注意，在使用 Advisor 时，不能同时使用 Tuner 和 Assessor。*

如果要自定义 Advisor，需要：

1. 从 MsgDispatcherBase 类继承并创建新的 Advisor 类
2. 实现所有除了 `handle_request` 外的，以 `handle_` 前缀开始的方法
3. 在 Experiment 的 YAML 文件中配置好自定义的 Advisor

样例如下：

**1) 从 MsgDispatcherBase 类继承并创建新的 Advisor 类**

```python
from nni.msg_dispatcher_base import MsgDispatcherBase

class CustomizedAdvisor(MsgDispatcherBase):
    def __init__(self, ...):
        ...
```

**2) 实现所有除了 `handle_request` 外的，以 `handle_` 前缀开始的方法**

Please refer to the implementation of Hyperband ([src/sdk/pynni/nni/hyperband_advisor/hyperband_advisor.py](https://github.com/Microsoft/nni/tree/master/src/sdk/pynni/nni/hyperband_advisor/hyperband_advisor.py)) for how to implement the methods.

**3) Configure your customized Advisor in experiment YAML config file**

Similar to tuner and assessor. NNI needs to locate your customized Advisor class and instantiate the class, so you need to specify the location of the customized Advisor class and pass literal values as parameters to the \_\_init__ constructor.

```yaml
advisor:
  codeDir: /home/abc/myadvisor
  classFileName: my_customized_advisor.py
  className: CustomizedAdvisor
  # Any parameter need to pass to your advisor class __init__ constructor
  # can be specified in this optional classArgs field, for example
  classArgs:
    arg1: value1
```