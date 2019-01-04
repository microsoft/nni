# **指南** - 自定义 advisor

*Advisor 用于同时需要调参器和评估器方法的自动机器学习算法。 Advisor 与调参器类似，它接收尝试的参数请求，最终结果，并生成尝试的参数。 另外，它也能像评估器一样接收中间结果，尝试的最终状态，并可以发送终止尝试的命令。 注意，在使用 Advisor 时，不能同时使用调参器和评估器。*

如果要自定义 Advisor，需要：

1. 从 MsgDispatcherBase 类继承并创建新的 Advisor 类
2. 实现所有除了 `handle_request` 外的，以 `handle_` 前缀开始的方法
3. 在实验的 yaml 文件中配置好自定义的 Advisor

样例如下：

**1) 从 MsgDispatcherBase 类继承并创建新的 Advisor 类**

```python
from nni.msg_dispatcher_base import MsgDispatcherBase

class CustomizedAdvisor(MsgDispatcherBase):
    def __init__(self, ...):
        ...
```

**2) 实现所有除了 `handle_request` 外的，以 `handle_` 前缀开始的方法**

参考 Hyperband 的实现 ([src/sdk/pynni/nni/hyperband_advisor/hyperband_advisor.py](../../src/sdk/pynni/nni/hyperband_advisor/hyperband_advisor.py)) 来学习如何实现这些方法。

**3) 在实验的 yaml 文件中配置好自定义的 Advisor**

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