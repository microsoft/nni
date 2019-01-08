# **指南** - 自定义 advisor

*Advisor 用于同时需要调参器和评估器方法的自动机器学习算法。 Advisor is similar to tuner on that it receives trial configuration request, final results, and generate trial configurations. 另外，它也能像评估器一样接收中间结果，尝试的最终状态，并可以发送终止尝试的命令。 注意，在使用 Advisor 时，不能同时使用调参器和评估器。*

如果要自定义 Advisor，需要：

1) Define an Advisor inheriting from the MsgDispatcherBase class 2) Implement the methods with prefix `handle_` except `handle_request` 3) Configure your customized Advisor in experiment yaml config file

Here ia an example:

**1) Define an Advisor inheriting from the MsgDispatcherBase class**

```python
from nni.msg_dispatcher_base import MsgDispatcherBase

class CustomizedAdvisor(MsgDispatcherBase):
    def __init__(self, ...):
        ...
```

**2) Implement the methods with prefix `handle_` except `handle_request`**

Please refer to the implementation of Hyperband ([src/sdk/pynni/nni/hyperband_advisor/hyperband_advisor.py](../src/sdk/pynni/nni/hyperband_advisor/hyperband_advisor.py)) for how to implement the methods.

**3) Configure your customized Advisor in experiment yaml config file**

Similar to tuner and assessor. NNI needs to locate your customized Advisor class and instantiate the class, so you need to specify the location of the customized Advisor class and pass literal values as parameters to the \_\_init__ constructor.

```yaml
advisor:
  codeDir: /home/abc/myadvisor
  classFileName: my_customized_advisor.py
  className: CustomizedAdvisor
  # 任何传入 __init__ 构造函数的参数
  # 都需要声明在 classArgs 字段中，如：
  classArgs:
    arg1: value1
```