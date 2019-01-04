# **指南** - 自定义 advisor

*Advisor 用于同时需要调参器和评估器方法的自动机器学习算法。 Advisor 与调参器类似，它接收尝试的参数请求，最终结果，并生成尝试的参数。 另外，它也能像评估器一样接收中间结果，尝试的最终状态，并可以发送终止尝试的命令。 Note that, if you use Advisor, tuner and assessor are not allowed to be used at the same time.*

So, if user want to implement a customized Advisor, she/he only need to:

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
  # Any parameter need to pass to your advisor class __init__ constructor
  # can be specified in this optional classArgs field, for example 
  classArgs:
    arg1: value1
```